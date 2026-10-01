"""Serves SSH sessions on a root worker with no Docker socket.

* **One interactive session per worker.** Sessions share the worker's filesystem
  and process namespace, so a second concurrent session is refused, as is a
  non-interactive task. Workers that share a root filesystem or
  ``/var/lib/flowmesh`` share the mount root, session accounts and session
  directories, so only one of them serves sessions.
* **Isolation by account.** Each session logs in as its own throwaway account,
  denied every root of the worker's state, and sshd and its helpers start from a
  scrubbed environment.
* **Links into the session's own directory.** The session's inputs and output live
  in a root-owned directory of its own, and its mount paths are links to them
  under a mount root emptied before and after every session.
* **The worker's size is the cap.** The ``SSH_MAX_CPU`` / ``MEMORY`` / ``PIDS``
  caps and the GPU subset need cgroup and device control over the worker itself,
  so a session runs niced and, where the kernel honours it, preferred by the OOM
  killer.
"""

import contextlib
import errno
import fcntl
import json
import logging
import os
import pwd
import re
import select
import shutil
import socket
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
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
    interpreter_argv,
    kill_processes,
    lock_account,
    process_identity_available,
    reap_stale_accounts,
    remove_tree,
    retire_account,
)

logger = logging.getLogger(__name__)

# On disk, beside the ACL ledger: staged inputs and output can be large.
SESSIONS_ROOT = acl.STATE_DIR / "ssh-sessions"
_MANIFEST_NAME = "manifest.json"
_AUTHORIZED_KEYS_NAME = "authorized_keys"
_SSHD_CANDIDATES = ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "sshd")
_KEYGEN_BINARY = "ssh-keygen"
_TAR_BINARY = "tar"
_TINI_BINARY = "tini"
_KEYGEN_TIMEOUT_SEC = 30.0
_TERMINATE_GRACE_SEC = 5.0
_KILL_POLL_SEC = 0.05
# The reader streams without pause, so this long without a byte means it is stuck.
_ARCHIVE_IDLE_TIMEOUT_SEC = 60.0
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
_LAUNCH_SCRIPT = (
    "import os, sys\n"
    "os.nice(int(sys.argv[1]))\n"
    "try:\n"
    "    with open('/proc/self/oom_score_adj', 'w') as fh:\n"
    "        fh.write(sys.argv[2])\n"
    "except OSError as exc:\n"
    "    print(f'Cannot raise the SSH session OOM score: {exc}', file=sys.stderr)\n"
    "os.execv(sys.argv[3], sys.argv[3:])\n"
)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_BACKEND_LOCK_NAME = "flowmesh-ssh-process.lock"
_RUN_DIR = Path("/run")
_MAX_LINK_HOPS = 40
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
_backend_lock_fds: list[int] | None = None
_backend_lock_mutex = threading.Lock()


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


def find_tini() -> str | None:
    return shutil.which(_TINI_BINARY)


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
        if find_tini() is None:
            logger.info(
                "Process SSH backend unavailable: tini is missing, so nothing would "
                "reap what a session orphans (install tini in the worker image)"
            )
            return False
        if not _acquire_backend_lock():
            logger.info(
                "Process SSH backend unavailable: another worker sharing this root "
                "filesystem or %s already serves process-mode sessions",
                acl.STATE_DIR.as_posix(),
            )
            return False
        if not _acl_ready(config):
            _release_backend_lock()
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

    def _default_session_host(self) -> str:
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
            session_dir.chmod(0o711)
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
            authorized_keys = session_dir / _AUTHORIZED_KEYS_NAME
            rendered, exported = _render_authorized_keys(
                cfg.authorized_keys, environment
            )
            # It carries the session's env; sshd reads it as the session user.
            _write_private(authorized_keys, rendered)
            account.grant_read(authorized_keys)
            process, port, log_path = _start_sshd(
                sshd_path,
                session_dir,
                host_key,
                authorized_keys,
                account,
                exported,
                self.session_bind_host(cfg.access_mode),
            )
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
    """Return the roots :func:`prepare_state_roots` makes ready; raise while a
    filesystem content store's root it had to leave out does not exist yet."""
    roots = prepare_state_roots(config)
    for root in _shared_roots(config):
        if not os.path.lexists(root) and not _inside_any(root, roots):
            raise ExecutionError(
                f"Refusing the SSH session: the shared content store {root} does "
                "not exist yet",
                retryable=True,
            )
    return roots


def prepare_state_roots(config: WorkerConfig) -> list[Path]:
    """Return the :func:`denied_roots` a session's deny entries land on, creating
    each missing one root-owned ``0700``; raise if one cannot be denied safely.

    A missing root inside another that exists is left out: a session cannot reach
    anything below a denied root. A filesystem content store is shared across
    nodes, so the content plane creates its root.
    """
    if problem := _state_problem(config):
        raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
    roots = denied_roots(config)
    existing = [root for root in roots if os.path.lexists(root)]
    shared = _shared_roots(config)
    ready: list[Path] = []
    for root in roots:
        if not os.path.lexists(root):
            if _inside_any(root, existing) or root in shared:
                continue
            try:
                _create_state_root(root)
            except OSError as exc:
                raise ExecutionError(
                    f"Cannot create worker state {root}: {exc}", retryable=True
                ) from exc
        ready.append(root)
    if problem := _state_problem(config):
        raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
    return ready


def _inside_any(path: Path, roots: Sequence[Path]) -> bool:
    return any(path != root and path.is_relative_to(root) for root in roots)


def denied_roots(config: WorkerConfig) -> list[Path]:
    """Return the worker state a session is denied, resolved to the paths
    ``setfacl`` acts on."""
    return list(
        dict.fromkeys(Path(os.path.realpath(path)) for path in _state_paths(config))
    )


def _shared_roots(config: WorkerConfig) -> list[Path]:
    if config.object_store.backend != BACKEND_FILESYSTEM:
        return []
    return [Path(os.path.realpath(config.object_store.filesystem_root))]


def _state_paths(config: WorkerConfig) -> list[Path]:
    """Return each path in ``DENIED_CONFIG_FIELDS``, and a filesystem content store's
    root, as the worker uses it, made absolute, with the heartbeat file replaced by
    its directory."""
    paths: list[Path] = []
    for field_name in DENIED_CONFIG_FIELDS:
        value = getattr(config, field_name)
        for configured in value if isinstance(value, tuple) else (value,):
            path = Path(configured).absolute()
            # Denying only the file would still let a session list its name,
            # which contains the worker token.
            paths.append(path.parent if field_name == "hb_file" else path)
    if config.object_store.backend == BACKEND_FILESYSTEM:
        paths.append(Path(config.object_store.filesystem_root).absolute())
    return paths


def _create_state_root(root: Path) -> None:
    """Create ``root`` as ``0700`` and each missing parent as ``0755``, never
    following a link."""
    parts = root.parts[1:]
    fd = os.open("/", _DIR_FLAGS)
    try:
        for index, part in enumerate(parts):
            try:
                os.mkdir(part, 0o700 if index == len(parts) - 1 else 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
    finally:
        os.close(fd)


def _state_problem(config: WorkerConfig) -> str | None:
    """Why this worker's state cannot be denied to a session safely, if it cannot."""
    roots = denied_roots(config)
    for path in _state_paths(config):
        if problem := _path_problem(path, roots):
            return problem
    for root in roots:
        if blocked := _required_path_under(root):
            return f"denying {root} would also deny {blocked}"
    return None


def _path_problem(path: Path, denied: Sequence[Path] = ()) -> str | None:
    """Why a session could replace ``path`` or a directory on the way to it, if
    it could.

    Every directory that resolving ``path`` looks a name up in is checked,
    those a link leads through included, except one at or below a root in
    ``denied``, which the session cannot search. A session can rename any entry
    of a directory it can write; in a sticky one only its own, so there the
    entry must be a directory the worker owns, not a link.
    """
    directory = Path("/")
    pending = list(path.parts[1:])
    hops = 0
    while pending:
        name = pending.pop(0)
        if name == "..":
            directory = directory.parent
            continue
        entry = directory / name
        try:
            dir_mode = os.stat(directory).st_mode
        except OSError as exc:
            return (
                f"cannot inspect {directory} on the way to worker state {path}: {exc}"
            )
        try:
            info: os.stat_result | None = os.lstat(entry)
        except FileNotFoundError:
            info = None
        except OSError as exc:
            return f"cannot inspect {entry} on the way to worker state {path}: {exc}"
        if dir_mode & stat.S_IWOTH and not any(
            directory.is_relative_to(root) for root in denied
        ):
            if not dir_mode & stat.S_ISVTX:
                return (
                    f"{directory}, on the way to worker state {path}, is "
                    "world-writable, so a session could replace what it holds"
                )
            if info is not None and (
                info.st_uid != os.geteuid() or stat.S_ISLNK(info.st_mode)
            ):
                return (
                    f"{entry}, on the way to worker state {path}, sits in a shared "
                    "directory but is not a directory this worker owns"
                )
        if info is None:
            return None
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            if hops > _MAX_LINK_HOPS:
                return f"too many links on the way to worker state {path}"
            target = PurePosixPath(os.readlink(entry))
            if target.is_absolute():
                directory = Path("/")
                pending[:0] = target.parts[1:]
            else:
                pending[:0] = target.parts
            continue
        directory = entry
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
        roots = prepare_state_roots(config)
    except ExecutionError as exc:
        logger.info("Process SSH backend unavailable: %s", exc)
        return False
    # The ledger and each session's authorized_keys, which the session reads by an
    # ACL entry, live there.
    for root in (*roots, acl.STATE_DIR):
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
    """Take the process-backend lock files, in ``/run`` and ``/var/lib/flowmesh``
    for root or the temp dir otherwise, for the life of the worker; return whether
    this worker holds them."""
    global _backend_lock_fds
    with _backend_lock_mutex:
        if _backend_lock_fds is not None:
            return True
        if os.geteuid() == 0:
            if not _root_only_dir(_RUN_DIR):
                logger.info(
                    "Process SSH backend unavailable: %s is not a root-owned "
                    "directory only root can write",
                    _RUN_DIR.as_posix(),
                )
                return False
            try:
                _make_private_dir(acl.STATE_DIR, 0o711)
            except (OSError, ExecutionError):
                logger.debug("Cannot prepare %s", acl.STATE_DIR, exc_info=True)
                return False
            bases: tuple[Path, ...] = (_RUN_DIR, acl.STATE_DIR)
        else:
            bases = (Path(tempfile.gettempdir()),)
        fds: list[int] = []
        for base in bases:
            if (fd := _take_lock(base / _BACKEND_LOCK_NAME)) is None:
                for held in fds:
                    os.close(held)
                return False
            fds.append(fd)
        _backend_lock_fds = fds
        return True


def _release_backend_lock() -> None:
    global _backend_lock_fds
    with _backend_lock_mutex:
        for fd in _backend_lock_fds or ():
            os.close(fd)
        _backend_lock_fds = None


def _root_only_dir(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == 0
        and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _take_lock(path: Path) -> int | None:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    except OSError:
        logger.debug("Cannot open the process-backend lock %s", path, exc_info=True)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno not in (errno.EAGAIN, errno.EACCES):
            logger.debug("Cannot take the process-backend lock %s", path, exc_info=True)
        return None
    return fd


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
        self._stop_lock = threading.Lock()
        self._collecting = False
        self._archiver: subprocess.Popen[bytes] | None = None

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
        with self._stop_lock:
            self._collecting = True
        try:
            self._collect(output_path, destination, max_bytes)
        finally:
            with self._stop_lock:
                self._collecting = False
                self._archiver = None

    def _collect(
        self, output_path: Path, destination: Path, max_bytes: int | None
    ) -> None:
        # The session has ended. Once nothing can log in and none of its processes
        # is left, the reader is the only one its account has, so the tree holds
        # still.
        self._end_logins()
        if not kill_processes(self.account.uid):
            raise ExecutionError(
                "Could not stop the SSH session's processes to collect its output"
            )
        self._signals.raise_if_cancelled()
        destination.mkdir(parents=True, exist_ok=True)
        archiver = _archive_as(self.account, output_path)
        with self._stop_lock:
            self._archiver = archiver
        finished = False
        try:
            self._signals.raise_if_cancelled()
            extract_output_archive(
                _read_archive(archiver),
                destination,
                max_bytes,
                self._signals.raise_if_cancelled,
            )
            finished = True
        except (OSError, tarfile.TarError) as exc:
            self._signals.raise_if_cancelled()
            raise ExecutionError(f"Failed to collect SSH output: {exc}") from exc
        finally:
            code = _end_archiver(archiver, finished)
        self._signals.raise_if_cancelled()
        if code is not None and code < 0:
            raise ExecutionError(
                f"Reading the SSH output was killed by signal {-code}, so the output "
                "collected may be incomplete"
            )

    def stop(self, timeout_sec: float) -> None:
        # While the output is collected, the only process left is the reader: a
        # stop lets it finish and a cancel kills it. Otherwise the session's own
        # processes go first, while sshd's subreaper still lives to reap what they
        # leave; those forked while sshd ends go after it.
        with self._stop_lock:
            if self._collecting:
                if self._signals.cancelled and self._archiver is not None:
                    self._archiver.kill()
                return
        self._kill_unless_collecting()
        _end_sshd(self._process)
        self._kill_unless_collecting()

    def _kill_unless_collecting(self) -> None:
        with self._stop_lock:
            if not self._collecting:
                kill_processes(self.account.uid)

    def _end_logins(self) -> None:
        """Bar the account from logging in, then end sshd and every connection it
        holds, so no process of the session starts from here on."""
        try:
            (self._session_dir / _AUTHORIZED_KEYS_NAME).unlink(missing_ok=True)
        except OSError as exc:
            raise ExecutionError(
                f"Could not bar logins to the SSH session to collect its output: {exc}"
            ) from exc
        lock_account(self.account.name)
        _end_sshd(self._process)

    def cleanup(self) -> None:
        with self._stop_lock:
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
    """The account a session allocated, recorded so a later worker can reap it if
    this one dies."""

    session_dir: Path
    account: str

    def write(self) -> None:
        path = self.session_dir / _MANIFEST_NAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"account": self.account}), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)

    @classmethod
    def read(cls, session_dir: Path) -> "SessionManifest | None":
        try:
            raw: dict[str, Any] = json.loads(
                (session_dir / _MANIFEST_NAME).read_text(encoding="utf-8")
            )
            return cls(session_dir=session_dir, account=str(raw["account"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None


def reap_session(session_dir: Path) -> bool:
    """Kill the session's sshd and retire the account ``session_dir``'s manifest
    records, then remove the directory; return ``False``, keeping it, if the account
    stays.

    The sshd is found by its config path, so one a worker started and died before
    it could record is found too.
    """
    config_path = (session_dir / "sshd_config").as_posix()
    for proc in psutil.process_iter():
        if _is_our_sshd(proc, config_path):
            _kill_tree(proc)
    manifest = SessionManifest.read(session_dir)
    if manifest is not None:
        try:
            uid = pwd.getpwnam(manifest.account).pw_uid
        except KeyError:
            uid = None
        if uid is not None and not retire_account(manifest.account, uid):
            return False
        logger.info("Reaped SSH session %s left by an earlier worker", session_dir.name)
    remove_tree(session_dir)
    return True


def _is_our_sshd(proc: psutil.Process, config_path: str) -> bool:
    """Return whether ``proc`` runs the sshd started with ``config_path``, directly
    or under its subreaper."""
    try:
        args = proc.cmdline()
    except psutil.Error:
        return False
    runs_sshd = any(
        os.path.basename(arg) == "sshd" or arg.startswith("sshd:") for arg in args
    )
    return runs_sshd and config_path in args


def _end_sshd(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL sshd, its subreaper and every connection process below them, so none
    outlives the session to log in to it later."""
    if process.poll() is not None:
        return
    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        return
    _kill_tree(root)
    try:
        process.wait(timeout=_TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        logger.warning("sshd did not exit after SIGKILL")


def _kill_tree(root: psutil.Process) -> None:
    """SIGKILL ``root`` and every process below it.

    ``root`` is stopped first so it starts nothing new, and what is below it is
    killed round after round, since a subreaper adopts what a killed process
    leaves; ``root`` goes last. A psutil process signals only the process it was
    made for, never a recycled pid.
    """
    with contextlib.suppress(psutil.Error):
        root.suspend()
    deadline = time.monotonic() + _TERMINATE_GRACE_SEC
    while time.monotonic() < deadline:
        try:
            victims = [p for p in root.children(recursive=True) if _is_live(p)]
        except psutil.Error:
            break
        if not victims:
            break
        for proc in victims:
            with contextlib.suppress(psutil.Error):
                proc.kill()
        time.sleep(_KILL_POLL_SEC)
    with contextlib.suppress(psutil.Error):
        root.kill()


def _is_live(proc: psutil.Process) -> bool:
    try:
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


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


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to a new file at ``path`` that only root can read."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


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


def _read_archive(archiver: subprocess.Popen[bytes]) -> Iterator[bytes]:
    """Yield what ``archiver`` writes until it closes its output, raising once it
    writes nothing for ``_ARCHIVE_IDLE_TIMEOUT_SEC``."""
    assert (stream := archiver.stdout) is not None
    fd = stream.fileno()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while True:
        if not poller.poll(_ARCHIVE_IDLE_TIMEOUT_SEC * 1000):
            raise ExecutionError(
                "Reading the SSH output made no progress for "
                f"{_ARCHIVE_IDLE_TIMEOUT_SEC:g}s"
            )
        if not (chunk := os.read(fd, _COPY_CHUNK)):
            return
        yield chunk


def _end_archiver(archiver: subprocess.Popen[bytes], finished: bool) -> int | None:
    """Stop the archiving child and return its exit code, or ``None`` when this had
    to kill it.

    A ``finished`` read lets it write out its last record and exit. A positive code
    is logged: ``tar`` gives one when it leaves out a file it could not read.
    """
    assert (stream := archiver.stdout) is not None
    code: int | None = None
    try:
        if finished:
            for _ in _read_archive(archiver):
                pass
            try:
                code = archiver.wait(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                pass
    finally:
        stream.close()
        if code is None:
            archiver.kill()
            archiver.wait()
    if code is not None and code > 0:
        logger.warning(
            "Reading the SSH output exited with code %d; files the session could "
            "not read are left out",
            code,
        )
    return code


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
        steps.append(lambda: _end_sshd(process))
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


def _start_sshd(
    sshd_path: str,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    account: SessionAccount,
    exported_env: list[str],
    bind_host: str,
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
                bind_host=bind_host,
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
    if (tini := find_tini()) is None:
        raise ExecutionError(
            "SSH session cannot start: tini is missing from this worker image"
        )
    log_handle = log_path.open("wb")
    log_path.chmod(0o600)
    try:
        return subprocess.Popen(  # nosec B603 - argv list, no shell=True, the worker's own interpreter and absolute paths via shutil.which()
            _launch_argv(tini, [sshd_path, "-D", "-e", "-f", config_path.as_posix()]),
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


def _launch_argv(tini: str, argv: list[str]) -> list[str]:
    """Return the command line that runs ``argv`` niced, preferred by the OOM killer
    where the kernel honours it, and under ``tini`` as a subreaper.

    A runaway session then cannot starve the worker's heartbeat, and what it
    orphans is reaped whatever runs as PID 1.
    """
    return interpreter_argv(
        _LAUNCH_SCRIPT,
        str(_SESSION_NICENESS),
        str(_SESSION_OOM_SCORE_ADJ),
        tini,
        "-s",
        "--",
        *argv,
    )


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
    bin_dir.chmod(0o711)
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
    bind_host: str,
) -> str:
    permit_env = ",".join(exported_env) if exported_env else "no"
    return "\n".join(
        (
            f"Port {port}",
            f"ListenAddress {bind_host}",
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
