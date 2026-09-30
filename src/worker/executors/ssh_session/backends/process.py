"""Serves SSH sessions on a root worker with no Docker socket.

* **One interactive session per worker.** Sessions share the worker's filesystem
  and process namespace, so a second concurrent session is refused, as is a
  non-interactive task, which needs a container runtime to run its image. Workers
  that share a root filesystem share the mount root and session accounts, so only
  one of them serves sessions.
* **Isolation by account.** Each session logs in as its own throwaway account,
  denied every root of the worker's state (see ``session_identity``), and sshd and
  its helpers start from a scrubbed environment.
* **Links into the session's own directory.** The session's inputs and output live
  in a root-owned directory of its own, and its mount paths are links to them
  under a mount root emptied before and after every session, created one component
  at a time, never through a link and never across a mount. Its output is read
  back by a child running as its account.
* **The worker's size is the cap.** ``SSH_MAX_*`` and the GPU subset need cgroup
  and device control over the worker itself, so a session runs niced, first in
  line for the OOM killer, and under a subreaper that reaps what it orphans.
"""

import errno
import fcntl
import json
import logging
import os
import pwd
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import psutil

from shared.content.config import BACKEND_FILESYSTEM
from shared.schemas.worker import SSHBackendName
from shared.tasks.worker_message import WorkerHardware
from worker.config import WorkerConfig

from ...base_executor import ExecutionError, RunSignals
from .. import acl
from ..base import (
    SessionInterrupted,
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    extract_output_archive,
    is_ssh_ready,
    path_size_bytes,
    read_local_proc_net_tcp,
    resolve_tailnet_address,
)
from ..config import SAFE_MOUNT_ROOT, normalize_mount_path
from ..inputs import stage_inputs_locally
from ..session_identity import (
    PRIVSEP_DIR,
    SessionAccount,
    account_name_for,
    exec_as,
    kill_processes,
    process_identity_available,
    reap_stale_accounts,
    remove_tree,
    retire_account,
)

logger = logging.getLogger(__name__)

# On disk, beside the ACL ledger: staged inputs and output can be large.
SESSIONS_ROOT = acl.STATE_DIR / "ssh-sessions"
_MANIFEST_NAME = "manifest.json"
_SSHD_CANDIDATES = ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "sshd")
_KEYGEN_BINARY = "ssh-keygen"
_TAR_BINARY = "tar"
_TINI_BINARY = "tini"
_KEYGEN_TIMEOUT_SEC = 30.0
_TERMINATE_GRACE_SEC = 5.0
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VALUE_FORBIDDEN = ('"', "\\", "\n", "\r")
_PORT_ATTEMPTS = 3
_READY_PROBE_SEC = 2.0
_READY_POLL_SEC = 0.5
_BIND_FAILURE_MARKERS = ("cannot bind", "address already in use", "bind to port")
_SPAWN_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ")
# sshd's own default; the session's PATH prepends the per-session bin dir.
_DEFAULT_SESSION_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/games"
_COPY_CHUNK = 64 * 1024
_SESSION_NICENESS = 10
_SESSION_OOM_SCORE_ADJ = 1000
# Lowers its own priority, then execs the session's sshd, which every process of
# the session inherits it from.
_LAUNCH_SCRIPT = (
    "import os, sys\n"
    "os.nice(int(sys.argv[1]))\n"
    "with open('/proc/self/oom_score_adj', 'w') as fh:\n"
    "    fh.write(sys.argv[2])\n"
    "os.execv(sys.argv[3], sys.argv[3:])\n"
)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_BACKEND_LOCK_NAME = "flowmesh-ssh-process.lock"
# A lookup that follows more links than this loops; the kernel stops at 40.
_MAX_LOOKUP_STEPS = 4096
# Paths every session needs; a denied root covering one would break it.
_SESSION_REQUIRED_PATHS = (
    Path(SAFE_MOUNT_ROOT),
    SESSIONS_ROOT,
    PRIVSEP_DIR,
    Path("/usr"),
    Path("/bin"),
    Path("/etc"),
    Path("/lib"),
)
# Every path-typed ``WorkerConfig`` field belongs to exactly one of these.
DENIED_CONFIG_FIELDS = (
    "results_dir",
    "private_state_dir",
    "content_dir",
    "hb_file",
    "state_dirs",
)
ALLOWED_CONFIG_FIELDS: tuple[str, ...] = ()
_MOUNTINFO = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{3})")
_backend_lock_fd: int | None = None
_backend_lock_mutex = threading.Lock()
_missing_tini_logged = False


def find_sshd() -> str | None:
    for candidate in _SSHD_CANDIDATES:
        if candidate.startswith("/"):
            if os.access(candidate, os.X_OK):
                return candidate
        elif resolved := shutil.which(candidate):
            return resolved
    return None


def find_ssh_keygen() -> str | None:
    return shutil.which(_KEYGEN_BINARY)


def find_tar() -> str | None:
    return shutil.which(_TAR_BINARY)


class ProcessSessionBackend(SSHSessionBackend):
    name = SSHBackendName.PROCESS
    supports_noninteractive = False

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        super().__init__(config, hardware)
        self._lock = threading.Lock()
        self._active: ProcessSession | None = None
        # False while an earlier session's account still runs a process.
        self._clean = True

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        if not process_identity_available():
            logger.info(
                "Process SSH backend unavailable: this worker cannot give a session "
                "an account of its own (it is not root, or lacks useradd/userdel)"
            )
            return False
        if find_sshd() is None or find_ssh_keygen() is None or find_tar() is None:
            logger.info(
                "Process SSH backend unavailable: sshd, ssh-keygen or tar is missing "
                "(install openssh-server in the worker image)"
            )
            return False
        if not (config.ssh_relay_host or resolve_tailnet_address()):
            logger.info(
                "Process SSH backend unavailable: the worker has no tailnet address "
                "and SSH_RELAY_HOST is unset, so its supervisor cannot reach a session"
            )
            return False
        if not _acl_ready(config):
            return False
        if not _acquire_backend_lock():
            logger.info(
                "Process SSH backend unavailable: another worker sharing this root "
                "filesystem already serves process-mode sessions, and they would "
                "share %s",
                SAFE_MOUNT_ROOT.as_posix(),
            )
            return False
        return True

    def prepare(self) -> None:
        if self._config.ssh_limits is not None:
            logger.warning(
                "SSH resource caps are configured but the process backend cannot "
                "enforce them; the size of this worker is the cap"
            )
        if self._config.enable_ssh_gpu_limit:
            logger.warning(
                "The process backend hands a session a CUDA_VISIBLE_DEVICES value it "
                "is free to unset; the GPU subset is advisory here, not enforced"
            )
        with self._lock:
            idle = self._active is None
        if idle:
            self.reap_stale()

    def relay_host(self) -> str:
        if override := self._config.ssh_relay_host:
            return override
        if (address := resolve_tailnet_address()) is None:
            raise ExecutionError(
                "Cannot publish a relay target for this SSH session: the worker has "
                "no tailnet address and SSH_RELAY_HOST is unset"
            )
        return address

    def session_host(self) -> str:
        if override := self._config.ssh_relay_host:
            return override
        return resolve_tailnet_address() or socket.getfqdn()

    def start_session(self, request: SessionRequest) -> "ProcessSession":
        if not request.cfg.interactive:
            raise ExecutionError(
                "Non-interactive SSH tasks need a container runtime to run the "
                "requested image; this worker runs SSH sessions as processes"
            )
        with self._lock:
            if self._active is not None:
                raise ExecutionError(
                    "This worker already has an SSH session, and process-mode "
                    "sessions share a filesystem, so only one may run at a time"
                )
            if not self._clean:
                raise ExecutionError(
                    "An earlier SSH session on this worker still has an account it "
                    "could not be rid of; the worker serves no session until a reap "
                    "removes it",
                    retryable=True,
                )
            session = self._create_session(request)
            self._active = session
        return session

    def teardown(self, worker_name: str) -> None:
        with self._lock:
            session = self._active
        if session is None:
            return
        session.stop(self._config.ssh_stop_timeout_sec)
        session.cleanup()

    def _release(self, session: "ProcessSession", clean: bool) -> None:
        with self._lock:
            if self._active is session:
                self._active = None
            self._clean = self._clean and clean

    # ------------------------------------------------------------------ #
    # Session construction
    # ------------------------------------------------------------------ #

    def _create_session(self, request: SessionRequest) -> "ProcessSession":
        cfg = request.cfg
        signals = request.signals
        sshd_path = find_sshd()
        keygen_path = find_ssh_keygen()
        if sshd_path is None or keygen_path is None:
            raise ExecutionError(
                "SSH session cannot start: sshd or ssh-keygen is missing from this "
                "worker image"
            )
        plan = _plan_mounts(request)
        if cfg.requested_image:
            logger.info(
                "Ignoring SSH spec image %s: process-mode sessions run in the "
                "worker's own root filesystem",
                cfg.requested_image,
            )

        session_dir = SESSIONS_ROOT / request.session_id
        manifest = SessionManifest(
            session_dir=session_dir, account=account_name_for(request.session_id)
        )
        account: SessionAccount | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            _make_private_dir(SESSIONS_ROOT.parent, 0o711)
            _make_private_dir(SESSIONS_ROOT, 0o711)
            session_dir.mkdir(mode=0o711)
            manifest.write()
            account = SessionAccount.create(
                manifest.account, session_dir / "home", ensure_state_roots(self._config)
            )
            if cfg.requested_user:
                logger.info(
                    "Ignoring SSH spec user %s: this session logs in as its own "
                    "account %s",
                    cfg.requested_user,
                    account.name,
                )

            staged_inputs: Path | None = None
            if request.resolved_inputs:
                staged_inputs = stage_inputs_locally(
                    request.resolved_inputs,
                    request.session_id,
                    signals,
                    parent=session_dir,
                )
                _hand_over_tree(staged_inputs, account)
            if signals.interrupted:
                raise SessionInterrupted
            output_path: Path | None = None
            if plan.output is not None:
                output_path = session_dir / "output"
                output_path.mkdir(mode=0o700)
                os.lchown(output_path, account.uid, account.gid)
            _link_mounts(plan, staged_inputs, output_path)

            finish_sentinel = account.home / ".flowmesh_finish"
            host_key = session_dir / "ssh_host_ed25519_key"
            _generate_host_key(keygen_path, host_key)
            environment = self._build_environment(
                cfg.user,
                cfg.authorized_keys,
                cfg.extra_env,
                [],
                [],
                bootstrap_entrypoint=False,
                gpu_device_ids=cfg.gpu_device_ids,
            )
            environment["FLOWMESH_FINISH_SENTINEL"] = finish_sentinel.as_posix()
            bin_dir = _install_finish_helper(session_dir, finish_sentinel)
            environment["PATH"] = f"{bin_dir.as_posix()}:{_DEFAULT_SESSION_PATH}"
            authorized_keys = session_dir / "authorized_keys"
            rendered, exported = _render_authorized_keys(
                cfg.authorized_keys, environment
            )
            authorized_keys.write_text(rendered, encoding="utf-8")
            # It carries the session's env; sshd reads it as the session user.
            authorized_keys.chmod(0o600)
            account.grant_read(authorized_keys)
            process, port, log_path = _start_sshd(
                sshd_path, session_dir, host_key, authorized_keys, account, exported
            )
            manifest.sshd_pid = process.pid
            manifest.write()
        except BaseException:
            if not _discard_session(process, account, session_dir):
                self._clean = False
            raise
        return ProcessSession(
            backend=self,
            process=process,
            port=port,
            session_dir=session_dir,
            log_path=log_path,
            output_path=output_path,
            finish_sentinel=finish_sentinel,
            account=account,
            signals=signals,
        )

    def reap_stale(self) -> None:
        """Remove the sshd, account, ACL entries and paths of every session an
        earlier worker left, and record whether any account stays."""
        if os.getuid() != 0:
            return
        clean = True
        if SESSIONS_ROOT.is_dir() and not SESSIONS_ROOT.is_symlink():
            for session_dir in SESSIONS_ROOT.iterdir():
                clean = reap_session(session_dir) and clean
        _clear_mount_root()
        clean = reap_stale_accounts() and clean
        with self._lock:
            self._clean = clean


def ensure_state_roots(config: WorkerConfig) -> list[Path]:
    """Return :func:`denied_roots`, creating each missing one root-owned ``0700`` so
    a session's deny entry lands on it; raise if one cannot be denied safely."""
    for path in _state_paths(config):
        if link := _lookup(path).shared_link:
            raise ExecutionError(
                f"Refusing the SSH session: worker state {path} is reached through "
                f"{link}, a link in a shared directory",
                retryable=True,
            )
    roots = denied_roots(config)
    for root in roots:
        if problem := _root_problem(root):
            raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
        if os.path.lexists(root):
            continue
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(root, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ExecutionError(
                f"Cannot create worker state {root}: {exc}", retryable=True
            ) from exc
        if problem := _root_problem(root):
            raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
    return roots


def denied_roots(config: WorkerConfig) -> list[Path]:
    """Return the worker state a session is denied, as the paths ``setfacl`` acts on.

    Each path in ``DENIED_CONFIG_FIELDS``, and a filesystem content store's root, is
    resolved, with the heartbeat file replaced by its directory. Each world-writable
    directory without the sticky bit that resolving it looks an entry up in is
    denied too, the highest of those below one another, since a session could swap
    that entry for one of its own; a path below a denied directory is covered by it.
    """
    roots: dict[Path, None] = {}
    for path in _state_paths(config):
        open_dirs = [
            directory for directory in _lookup(path).dirs if _is_open_dir(directory)
        ]
        resolved = Path(os.path.realpath(path))
        for candidate in (*open_dirs, resolved):
            if not any(other in candidate.parents for other in open_dirs):
                roots[candidate] = None
    return list(roots)


def _state_paths(config: WorkerConfig) -> list[Path]:
    """Return the absolute paths of the worker state a session is denied."""
    paths: list[Path] = []
    for field_name in DENIED_CONFIG_FIELDS:
        value = getattr(config, field_name)
        for path in value if isinstance(value, tuple) else (value,):
            path = Path(os.path.abspath(path))
            if field_name == "hb_file":
                # Denying only the file would still let a session list its name,
                # which contains the worker token.
                path = path.parent
            paths.append(path)
    if config.object_store.backend == BACKEND_FILESYSTEM:
        paths.append(Path(os.path.abspath(config.object_store.filesystem_root)))
    return paths


@dataclass(slots=True)
class _Lookup:
    """The resolved directories a path's lookup reads an entry from, and the first
    link it follows out of a shared directory."""

    dirs: list[Path] = field(default_factory=list)
    shared_link: Path | None = None


def _lookup(path: Path) -> _Lookup:
    """Resolve the absolute ``path`` one component at a time, as the kernel does."""
    lookup = _Lookup()
    pending = list(PurePosixPath(path).parts[1:])
    current = Path("/")
    for _ in range(_MAX_LOOKUP_STEPS):
        if not pending:
            break
        name = pending.pop(0)
        if name == "..":
            current = current.parent
            continue
        lookup.dirs.append(current)
        candidate = current / name
        try:
            is_link = stat.S_ISLNK(os.lstat(candidate).st_mode)
        except OSError:
            is_link = False
        if not is_link:
            current = candidate
            continue
        if lookup.shared_link is None and _is_shared_dir(current):
            lookup.shared_link = candidate
        target = PurePosixPath(os.readlink(candidate))
        if target.is_absolute():
            current = Path("/")
        pending[:0] = [part for part in target.parts if part not in ("/", ".")]
    return lookup


def _root_problem(root: Path) -> str | None:
    """Why ``root`` cannot be denied to a session safely, if it cannot."""
    if blocked := _required_path_under(root):
        return f"denying {root} would also deny {blocked}"
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"cannot inspect worker state {root}: {exc}"
    if stat.S_ISLNK(info.st_mode):
        return f"worker state {root} is a link"
    if _is_shared_dir(root.parent) and (
        info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        return (
            f"worker state {root} sits in a shared directory but is not a root-owned "
            "path only root can write"
        )
    return None


def _acl_ready(config: WorkerConfig) -> bool:
    if not acl.tools_available():
        logger.info(
            "Process SSH backend unavailable: setfacl/getfacl are missing, so a "
            "session could not be denied this worker's state (install the acl "
            "package in the worker image)"
        )
        return False
    try:
        roots = ensure_state_roots(config)
    except ExecutionError as exc:
        logger.info("Process SSH backend unavailable: %s", exc)
        return False
    for root in roots:
        try:
            acl.probe(_probe_dir(root))
        except (OSError, ExecutionError) as exc:
            logger.info(
                "Process SSH backend unavailable: cannot deny sessions %s with an "
                "ACL: %s",
                root,
                exc,
            )
            return False
    return True


def _is_open_dir(path: Path) -> bool:
    """Whether any account may rename entries in ``path``."""
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return stat.S_ISDIR(mode) and bool(mode & stat.S_IWOTH) and not mode & stat.S_ISVTX


def _is_shared_dir(path: Path) -> bool:
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return False
    return bool(mode & (stat.S_ISVTX | stat.S_IWOTH))


def _required_path_under(root: Path) -> Path | None:
    """A path every session needs that denying ``root`` would also deny."""
    for required in (Path(tempfile.gettempdir()), *_SESSION_REQUIRED_PATHS):
        if required == root or required.is_relative_to(root):
            return required
    return None


def _probe_dir(root: Path) -> Path:
    """The existing directory whose filesystem will hold ``root``."""
    candidate = root
    while not candidate.is_dir():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def _acquire_backend_lock() -> bool:
    """Take the process-backend lock file, in ``/run`` for root or the temp dir
    otherwise, for the life of the worker; return whether this worker holds it.

    Workers that see the same lock file share the mount root and session accounts,
    so only one of them serves sessions.
    """
    global _backend_lock_fd
    with _backend_lock_mutex:
        if _backend_lock_fd is not None:
            return True
        base = Path("/run") if os.geteuid() == 0 else Path(tempfile.gettempdir())
        try:
            fd = os.open(
                base / _BACKEND_LOCK_NAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError:
            logger.debug("Cannot open the process-backend lock", exc_info=True)
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                logger.debug("Cannot take the process-backend lock", exc_info=True)
            return False
        _backend_lock_fd = fd
        return True


@dataclass(slots=True)
class MountPlan:
    """The session's mount paths, each under the mount root."""

    inputs: list[tuple[Path, str]] = field(default_factory=list)
    output: Path | None = None


def _plan_mounts(request: SessionRequest) -> MountPlan:
    """Check the mount paths against each other before anything is created.

    A mount path nested in another would put the backend's own files inside a
    directory the session writes to.
    """
    plan = MountPlan(
        inputs=[
            (Path(resolved.mount_path), resolved.task_id)
            for resolved in request.resolved_inputs
        ]
    )
    if (output := request.cfg.output) is not None:
        plan.output = Path(
            normalize_mount_path(output.mount_path, field_name="sshOutput.mountPath")
        )
    paths = [path for path, _ in plan.inputs]
    if plan.output is not None:
        paths.append(plan.output)
    for index, path in enumerate(paths):
        if path == Path(SAFE_MOUNT_ROOT):
            raise ExecutionError(
                f"SSH mountPath {path.as_posix()} cannot be the mount root itself"
            )
        for other in paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ExecutionError(
                    f"SSH mountPaths {path.as_posix()} and {other.as_posix()} overlap"
                )
    return plan


def _link_mounts(
    plan: MountPlan, staged_inputs: Path | None, output_path: Path | None
) -> None:
    """Link each mount path to its target in the session's directory, under a mount
    root emptied first."""
    links = [
        (mount_path, staged_inputs / task_id)
        for mount_path, task_id in plan.inputs
        if staged_inputs is not None
    ]
    if plan.output is not None and output_path is not None:
        links.append((plan.output, output_path))
    if not links:
        return
    root = Path(SAFE_MOUNT_ROOT)
    try:
        _reset_mount_root(root, create=True)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot prepare {root.as_posix()} on this worker: {exc}", retryable=True
        ) from exc
    for mount_path, target in links:
        _link_mount_path(root, mount_path.as_posix(), target)


def _clear_mount_root() -> None:
    try:
        _reset_mount_root(Path(SAFE_MOUNT_ROOT), create=False)
    except OSError:
        logger.warning("Could not clear %s", SAFE_MOUNT_ROOT.as_posix(), exc_info=True)


def _reset_mount_root(root: Path, create: bool) -> None:
    """Empty ``root`` without following a link, leaving it root-owned ``0755``.

    ``root`` itself is kept rather than replaced when it is a directory, since it
    may be a mount point.
    """
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        if not create:
            return
        root.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(root, 0o755)
        info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(root)
        if not create:
            return
        os.mkdir(root, 0o755)
    _refuse_nested_mounts(root)
    fd = os.open(root, _DIR_FLAGS)
    try:
        with os.scandir(fd) as scanner:
            entries = list(scanner)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.name, dir_fd=fd)
            else:
                os.unlink(entry.name, dir_fd=fd)
        if os.geteuid() == 0:
            os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o755)  # nosec B103 - the session must traverse it
    finally:
        os.close(fd)


def _refuse_nested_mounts(root: Path) -> None:
    """Refuse to empty ``root`` when a filesystem is mounted anywhere below it.

    Everything under the mount root is the backend's own, so a mount there is an
    operator's, and emptying the root would delete what it holds. A bind mount from
    the same filesystem keeps the device number, so the kernel's mount table is what
    finds it; the device check covers a worker that cannot read that table.
    """
    for mount_point in _mount_points():
        if mount_point != root and mount_point.is_relative_to(root):
            raise OSError(
                errno.EBUSY, "a filesystem is mounted below the mount root", mount_point
            )
    device = os.lstat(root).st_dev
    for parent, dirs, _ in os.walk(root, followlinks=False):
        for name in dirs:
            path = os.path.join(parent, name)
            if os.lstat(path).st_dev != device:
                raise OSError(
                    errno.EBUSY, "a filesystem is mounted below the mount root", path
                )


def _mount_points() -> list[Path]:
    """Mount points in this process's mount namespace, or none when unreadable."""
    try:
        text = _MOUNTINFO.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    points: list[Path] = []
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) > 4:
            points.append(Path(_OCTAL_ESCAPE_RE.sub(_unescape_octal, fields[4])))
    return points


def _unescape_octal(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 8))


def _link_mount_path(root: Path, mount_path: str, target: Path) -> None:
    """Create ``mount_path`` as a link to ``target``, never following a link.

    Each missing component below ``root`` is created relative to its parent's
    descriptor, and an existing one is only entered if it is a real directory.
    """
    parts = PurePosixPath(mount_path).relative_to(PurePosixPath(root)).parts
    if not parts:
        raise ExecutionError(
            f"mountPath {mount_path} must name a path below {root.as_posix()} on "
            "this worker"
        )
    fd = os.open(root, _DIR_FLAGS)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                raise ExecutionError(
                    f"mountPath {mount_path} conflicts with another mountPath"
                ) from exc
            os.close(fd)
            fd = child
            os.fchmod(fd, 0o755)  # nosec B103 - the session must traverse it
        try:
            os.symlink(target, parts[-1], dir_fd=fd)
        except FileExistsError as exc:
            raise ExecutionError(
                f"mountPath {mount_path} conflicts with another mountPath"
            ) from exc
    finally:
        os.close(fd)


def _hand_over_tree(root: Path, account: SessionAccount) -> None:
    """Hand ``root``, a tree the backend created, to ``account`` without following a
    link in it."""
    os.lchown(root, account.uid, account.gid)
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            os.lchown(os.path.join(current, name), account.uid, account.gid)


class ProcessSession(SSHSession):
    """An SSH session running as an sshd process beside the worker."""

    def __init__(
        self,
        backend: ProcessSessionBackend,
        process: subprocess.Popen[bytes],
        port: int,
        session_dir: Path,
        log_path: Path,
        output_path: Path | None,
        finish_sentinel: Path,
        account: SessionAccount,
        signals: RunSignals,
    ) -> None:
        self._backend = backend
        self._process = process
        self._port = port
        self._session_dir = session_dir
        self._log_path = log_path
        self._output_path = output_path
        self._finish_sentinel = finish_sentinel
        self.account = account
        self._signals = signals
        self._cleaned = False

    def login_user(self) -> str:
        return self.account.name

    def wait_ready(self, timeout_sec: float) -> int | None:
        signals = self._signals
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if signals.raise_if_cancelled():
                return None
            exit_code = self._process.poll()
            if exit_code is not None:
                if signals.raise_if_cancelled():
                    return None
                raise ExecutionError(
                    f"sshd exited (code {exit_code}) before the SSH session became "
                    f"ready.{self._log_tail()}"
                )
            if is_ssh_ready("127.0.0.1", self._port):
                return self._port
            time.sleep(_READY_POLL_SEC)
        if signals.raise_if_cancelled():
            return None
        raise ExecutionError(
            f"Timed out waiting for SSH readiness on port {self._port}."
            f"{self._log_tail()}"
        )

    def poll(self) -> int | None:
        return self._process.poll()

    def finish_requested(self) -> bool:
        # Root must not follow a link the session placed here.
        return os.path.lexists(self._finish_sentinel)

    def established_connections(self) -> int | None:
        if (proc_net_tcp := read_local_proc_net_tcp()) is None:
            return None
        return count_established_connections(proc_net_tcp, self._port)

    def output_size_bytes(self) -> int | None:
        if (output_path := self._output_path) is None:
            return None
        return path_size_bytes(output_path)

    def collect_output(self, destination: Path, max_bytes: int | None) -> None:
        """Copy the session's directories and regular files into ``destination``,
        read by a child running as the session's account, so nothing is read with
        more access than the session has."""
        if (output_path := self._output_path) is None:
            return
        # The session has ended; with its processes gone the tree holds still.
        if not kill_processes(self.account.uid):
            raise ExecutionError(
                "Could not stop the SSH session's processes to collect its output"
            )
        self._signals.raise_if_cancelled()
        destination.mkdir(parents=True, exist_ok=True)
        archiver = _archive_as(self.account, output_path)
        assert (archive := archiver.stdout) is not None
        try:
            extract_output_archive(
                iter(lambda: archive.read(_COPY_CHUNK), b""),
                destination,
                max_bytes,
                self._signals.raise_if_cancelled,
            )
        except (OSError, tarfile.TarError) as exc:
            raise ExecutionError(f"Failed to collect SSH output: {exc}") from exc
        finally:
            _end_archiver(archiver)

    def stop(self, timeout_sec: float) -> None:
        # The session's own processes go first, while sshd's subreaper still lives
        # to reap what they leave; those forked while sshd stops go after it.
        kill_processes(self.account.uid)
        _terminate(self._process, timeout_sec)
        kill_processes(self.account.uid)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        clean = False
        try:
            clean = _discard_session(self._process, self.account, self._session_dir)
        finally:
            # A failure above must not strand the worker refusing every later
            # session; the reap before the next session is the backstop.
            self._backend._release(self, clean)

    def _log_tail(self, max_chars: int = 2000) -> str:
        text = _read_log(self._log_path)
        return f"\nsshd output:\n{text[-max_chars:]}" if text else ""


@dataclass(slots=True)
class SessionManifest:
    """The account and sshd a session allocated, recorded so a later worker can reap
    them if this one dies."""

    session_dir: Path
    account: str
    sshd_pid: int | None = None

    def write(self) -> None:
        path = self.session_dir / _MANIFEST_NAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"account": self.account, "sshd_pid": self.sshd_pid}),
            encoding="utf-8",
        )
        tmp.chmod(0o600)
        tmp.replace(path)

    @classmethod
    def read(cls, session_dir: Path) -> "SessionManifest | None":
        try:
            raw: dict[str, Any] = json.loads(
                (session_dir / _MANIFEST_NAME).read_text(encoding="utf-8")
            )
            return cls(
                session_dir=session_dir,
                account=str(raw["account"]),
                sshd_pid=raw.get("sshd_pid"),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None


def reap_session(session_dir: Path) -> bool:
    """Kill the sshd and retire the account that ``session_dir``'s manifest records,
    then remove the directory; return ``False``, keeping it, if the account stays.
    """
    manifest = SessionManifest.read(session_dir)
    if manifest is not None:
        config_path = (session_dir / "sshd_config").as_posix()
        if manifest.sshd_pid is not None and _is_our_sshd(
            manifest.sshd_pid, config_path
        ):
            _kill_tree(manifest.sshd_pid)
        try:
            uid = pwd.getpwnam(manifest.account).pw_uid
        except KeyError:
            uid = None
        if uid is not None and not retire_account(manifest.account, uid):
            return False
        logger.info("Reaped SSH session %s left by an earlier worker", session_dir.name)
    remove_tree(session_dir)
    return True


def _is_our_sshd(pid: int, config_path: str) -> bool:
    """Return whether ``pid`` runs the sshd started with ``config_path``, directly
    or under its subreaper, so a recycled pid is never signalled.

    sshd rewrites its process title, so the command line is read as one string.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"sshd" in cmdline and config_path.encode() in cmdline


def _kill_tree(pid: int) -> None:
    """SIGKILL ``pid`` and every process below it, since a subreaper's child outlives
    the subreaper's SIGKILL."""
    try:
        root = psutil.Process(pid)
        victims = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for proc in victims:
        try:
            proc.send_signal(signal.SIGKILL)
        except psutil.Error:
            continue
    psutil.wait_procs(victims, timeout=_TERMINATE_GRACE_SEC)


def _make_private_dir(path: Path, mode: int) -> None:
    """Create ``path`` root-owned, or check that the existing one is."""
    try:
        path.mkdir(mode=mode)
    except FileExistsError:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0:
            raise ExecutionError(
                f"{path.as_posix()} is not a root-owned directory"
            ) from None
    path.chmod(mode)


def _archive_as(account: SessionAccount, source: Path) -> subprocess.Popen[bytes]:
    """Start a child running as ``account`` that writes a tar of ``source`` to its
    stdout."""
    tar = find_tar()
    if tar is None:
        raise ExecutionError("tar is missing from this worker image")
    try:
        return subprocess.Popen(  # nosec B603 - argv list, no shell=True, the worker's own interpreter and an absolute path via shutil.which()
            exec_as(
                account.uid,
                account.gid,
                [tar, "--create", "--file=-", f"--directory={source.as_posix()}", "."],
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=_sanitized_spawn_env(),
            cwd="/",
            start_new_session=True,
        )
    except OSError as exc:
        raise ExecutionError(f"Failed to read the SSH output: {exc}") from exc


def _end_archiver(archiver: subprocess.Popen[bytes]) -> None:
    """Stop the archiving child, logging a non-zero exit, which ``tar`` gives when
    it leaves out a file it could not read."""
    assert archiver.stdout is not None
    archiver.stdout.close()
    if archiver.poll() is None:
        archiver.kill()
    code = archiver.wait()
    if code not in (0, -signal.SIGKILL):
        logger.warning(
            "Reading the SSH output exited with code %d; files the session could "
            "not read are left out",
            code,
        )


def _discard_session(
    process: subprocess.Popen[bytes] | None,
    account: SessionAccount | None,
    session_dir: Path,
) -> bool:
    """Stop the sshd, retire the account and remove the paths a session allocated,
    each step whether or not an earlier one failed; return whether the sshd and
    account are gone."""
    steps: list[Callable[[], Any]] = []
    if process is not None:
        steps.append(lambda: _terminate(process, _TERMINATE_GRACE_SEC))
    if account is not None:
        steps.append(account.release)
    clean = True
    for step in steps:
        try:
            step()
        except Exception:
            clean = False
            logger.warning("Failed to release part of an SSH session", exc_info=True)
    _clear_mount_root()
    remove_tree(session_dir)
    return clean


def _terminate(process: subprocess.Popen[bytes], timeout_sec: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_tree(process.pid)
        try:
            process.wait(timeout=_TERMINATE_GRACE_SEC)
        except subprocess.TimeoutExpired:
            logger.warning("sshd did not exit after SIGKILL")


def _start_sshd(
    sshd_path: str,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    account: SessionAccount,
    exported_env: list[str],
) -> tuple[subprocess.Popen[bytes], int, Path]:
    """Start sshd, retrying on another port when it loses the race to bind.

    ``_pick_free_port`` has to release the port before sshd claims it, so the
    kernel can hand it to something else in between.
    """
    log_path = session_dir / "sshd.log"
    config_path = session_dir / "sshd_config"
    detail = ""
    for _ in range(_PORT_ATTEMPTS):
        port = _pick_free_port()
        config_path.write_text(
            _render_sshd_config(
                port=port,
                session_dir=session_dir,
                host_key=host_key,
                authorized_keys=authorized_keys,
                login_user=account.name,
                exported_env=exported_env,
            ),
            encoding="utf-8",
        )
        config_path.chmod(0o600)
        process = _spawn_sshd(sshd_path, config_path, log_path)
        deadline = time.time() + _READY_PROBE_SEC
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if is_ssh_ready("127.0.0.1", port):
                return process, port, log_path
            time.sleep(0.05)
        if process.poll() is None:
            return process, port, log_path
        detail = _read_log(log_path)
        if not any(marker in detail.lower() for marker in _BIND_FAILURE_MARKERS):
            raise ExecutionError(f"sshd exited immediately.\nsshd output:\n{detail}")
        logger.info("sshd could not bind port %d; retrying on another port", port)
    raise ExecutionError(
        f"sshd could not bind a free port after {_PORT_ATTEMPTS} attempts."
        f"\nsshd output:\n{detail}"
    )


def _spawn_sshd(
    sshd_path: str, config_path: Path, log_path: Path
) -> subprocess.Popen[bytes]:
    log_handle = log_path.open("wb")
    log_path.chmod(0o600)
    try:
        return subprocess.Popen(  # nosec B603 - argv list, no shell=True, the worker's own interpreter and absolute paths via shutil.which()
            _launch_argv([sshd_path, "-D", "-e", "-f", config_path.as_posix()]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_sanitized_spawn_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise ExecutionError(f"Failed to start sshd: {exc}") from exc
    finally:
        log_handle.close()


def _launch_argv(argv: list[str]) -> list[str]:
    """Return the command line that runs ``argv`` niced, first in line for the OOM
    killer, and under a subreaper when ``tini`` is installed.

    A runaway session then cannot starve the worker's heartbeat, and what it
    orphans is reaped whatever runs as PID 1.
    """
    global _missing_tini_logged
    if tini := shutil.which(_TINI_BINARY):
        argv = [tini, "-s", "--", *argv]
    elif not _missing_tini_logged:
        _missing_tini_logged = True
        logger.warning(
            "tini is missing from this worker image; what an SSH session orphans is "
            "left to PID 1 to reap"
        )
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _LAUNCH_SCRIPT,
        str(_SESSION_NICENESS),
        str(_SESSION_OOM_SCORE_ADJ),
        *argv,
    ]


def _generate_host_key(keygen_path: str, host_key: Path) -> None:
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [keygen_path, "-q", "-t", "ed25519", "-N", "", "-f", host_key.as_posix()],
        capture_output=True,
        timeout=_KEYGEN_TIMEOUT_SEC,
        env=_sanitized_spawn_env(),
        check=False,
    )
    if result.returncode != 0 or not host_key.exists():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionError(f"Failed to generate SSH host key: {detail}")


def _install_finish_helper(session_dir: Path, sentinel: Path) -> Path:
    """Give the session the ``flowmesh-finish`` command the Docker image ships.

    ``0711`` is enough for a PATH lookup, which stats candidates rather than
    listing the directory.
    """
    bin_dir = session_dir / "bin"
    bin_dir.mkdir(mode=0o711)
    helper = bin_dir / "flowmesh-finish"
    helper.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        f'touch "{sentinel.as_posix()}"\n'
        'echo "FlowMesh finish requested; the SSH session will close shortly."\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    return bin_dir


def _sanitized_spawn_env() -> dict[str, str]:
    """Return an environment for helper processes that carries none of the
    worker's secrets, which sshd would otherwise pass to a session's shell."""
    env = {key: value for key in _SPAWN_ENV_KEYS if (value := os.environ.get(key))}
    env.setdefault("PATH", os.defpath)
    return env


def _render_sshd_config(
    port: int,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    login_user: str,
    exported_env: list[str],
) -> str:
    permit_env = ",".join(exported_env) if exported_env else "no"
    return "\n".join(
        (
            f"Port {port}",
            "ListenAddress 0.0.0.0",
            f"HostKey {host_key.as_posix()}",
            f"PidFile {(session_dir / 'sshd.pid').as_posix()}",
            f"AuthorizedKeysFile {authorized_keys.as_posix()}",
            f"AllowUsers {login_user}",
            "PasswordAuthentication no",
            "KbdInteractiveAuthentication no",
            "PubkeyAuthentication yes",
            "PermitRootLogin no",
            f"PermitUserEnvironment {permit_env}",
            "StrictModes no",
            "UsePAM no",
            "PrintMotd no",
            "AllowTcpForwarding no",
            "X11Forwarding no",
            "AllowAgentForwarding no",
            "GatewayPorts no",
            "Subsystem sftp internal-sftp",
            "",
        )
    )


def _render_authorized_keys(
    authorized_keys: list[str], environment: dict[str, str]
) -> tuple[str, list[str]]:
    """Render authorized_keys, carrying session env as per-key options.

    sshd does not pass its own environment into a login shell, so the values the
    session is supposed to see travel as ``environment=`` options on each key.
    Returns the rendered file and the names exported, which
    ``PermitUserEnvironment`` must list.
    """
    exported = [
        name
        for name, value in sorted(environment.items())
        if _is_safe_env_entry(name, value)
    ]
    options = ",".join(f'environment="{name}={environment[name]}"' for name in exported)
    lines = [
        f"{options} {key}" if options else key
        for raw_key in authorized_keys
        if (key := raw_key.strip())
    ]
    return ("\n".join(lines) + "\n" if lines else "", exported if lines else [])


def _is_safe_env_entry(name: str, value: str) -> bool:
    if not _ENV_NAME_RE.match(name):
        logger.warning("Dropping SSH session env var with unsupported name %r", name)
        return False
    if any(ch in value for ch in _ENV_VALUE_FORBIDDEN):
        logger.warning("Dropping SSH session env var %s: unsupported value", name)
        return False
    return True


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _read_log(log_path: Path, max_chars: int = 4000) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").strip()[
            -max_chars:
        ]
    except OSError:
        return ""
