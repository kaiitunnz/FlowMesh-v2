"""Process session backend: sshd as a process beside the worker.

For root workers with no Docker socket, which cannot create a sibling container to
put a session in. Having no container around the session shapes the backend:

* **One interactive session per worker.** Sessions sharing a worker would share its
  filesystem and process namespace, so a second concurrent session is refused, and
  a non-interactive task, which needs a container runtime to run its image, is
  refused too.
* **Isolation by account.** Each session logs in as its own throwaway account,
  denied every root of the worker's state (see ``session_identity``), and sshd and
  its helpers start from a scrubbed environment.
* **Real paths, created fresh.** The session's mount paths are created under a
  mount root the backend recreates for every session, one component at a time and
  never through a symlink, and only paths the backend created are handed to the
  session.
* **No worker-side resource cap.** ``SSH_MAX_*`` and the GPU subset need cgroup and
  device control the worker does not have over itself.
"""

import json
import logging
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.schemas.worker import SSHBackendName
from shared.tasks.worker_message import WorkerHardware
from shared.utils import parse_float_env
from worker.config import WorkerConfig

from ...base_executor import ExecutionError, RunSignals
from ..base import (
    SessionInterrupted,
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    is_ssh_ready,
    read_local_proc_net_tcp,
    resolve_tailnet_address,
)
from ..config import (
    SAFE_MOUNT_ROOT,
    STOP_TIMEOUT_SEC,
    normalize_mount_path,
    raise_if_exceeded,
)
from ..inputs import stage_inputs_locally
from ..session_identity import (
    SessionAccount,
    account_name_for,
    delete_account,
    kill_processes,
    lift_denials,
    process_identity_available,
    reap_stale_accounts,
    supports_denials,
)

logger = logging.getLogger(__name__)

SESSIONS_ROOT = Path("/run/flowmesh/ssh-sessions")
_MANIFEST_NAME = "manifest.json"
_SSHD_CANDIDATES = ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "sshd")
_KEYGEN_BINARY = "ssh-keygen"
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


class ProcessSessionBackend(SSHSessionBackend):
    name = SSHBackendName.PROCESS
    supports_noninteractive = False

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        super().__init__(config, hardware)
        self._lock = threading.Lock()
        self._active: ProcessSession | None = None
        # A worker that died with a session up left it behind.
        self._reap_stale()

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        if not process_identity_available():
            logger.info(
                "Process SSH backend unavailable: this worker cannot give a session "
                "an account of its own (it is not root, or lacks useradd/userdel)"
            )
            return False
        if find_sshd() is None or find_ssh_keygen() is None:
            logger.info(
                "Process SSH backend unavailable: sshd or ssh-keygen is missing "
                "(install openssh-server in the worker image)"
            )
            return False
        if not supports_denials(config.state_roots):
            logger.info(
                "Process SSH backend unavailable: this worker's state cannot be "
                "denied to a session (setfacl is missing or ACLs are unsupported)"
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
            self._reap_stale()

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
            session = self._create_session(request)
            self._active = session
        return session

    def teardown(self, worker_name: str) -> None:
        with self._lock:
            session = self._active
        if session is None:
            return
        session.stop(parse_float_env("SSH_STOP_TIMEOUT_SEC", STOP_TIMEOUT_SEC))
        session.cleanup()

    def _release(self, session: "ProcessSession") -> None:
        with self._lock:
            if self._active is session:
                self._active = None

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
        if cfg.image:
            logger.info(
                "Ignoring SSH spec image %s: process-mode sessions run in the "
                "worker's own root filesystem",
                cfg.image,
            )

        session_dir = SESSIONS_ROOT / request.session_id
        manifest = SessionManifest(
            session_dir=session_dir,
            account=account_name_for(request.session_id),
            roots=[root.as_posix() for root in self._config.state_roots],
        )
        account: SessionAccount | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            _make_private_dir(SESSIONS_ROOT.parent, 0o711)
            _make_private_dir(SESSIONS_ROOT, 0o711)
            session_dir.mkdir(mode=0o711)
            manifest.write()
            account = SessionAccount.create(
                manifest.account, session_dir / "home", self._config.state_roots
            )
            manifest.uid = account.uid
            manifest.write()
            if cfg.user != account.name:
                logger.info(
                    "Ignoring SSH spec user %s: this session logs in as its own "
                    "account %s",
                    cfg.user,
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
            output_path = _materialize_mounts(plan, staged_inputs, account)

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
            _discard_session(process, account, session_dir)
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

    def _reap_stale(self) -> None:
        """Remove what a session left behind when its worker died: its sshd, its
        account and denials, and its paths."""
        if os.getuid() != 0:
            return
        roots = self._config.state_roots
        if SESSIONS_ROOT.is_dir() and not SESSIONS_ROOT.is_symlink():
            for session_dir in SESSIONS_ROOT.iterdir():
                reap_session(session_dir)
        _reset_mount_root()
        reap_stale_accounts(roots)


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


def _materialize_mounts(
    plan: MountPlan, staged_inputs: Path | None, account: SessionAccount
) -> Path | None:
    """Create the mount paths under a fresh mount root; returns the output path."""
    _reset_mount_root()
    for mount_path, task_id in plan.inputs:
        assert staged_inputs is not None
        parent = _make_dirs_nofollow(mount_path.parent)
        os.symlink(staged_inputs / task_id, mount_path.name, dir_fd=parent)
        os.close(parent)
    if plan.output is None:
        return None
    parent = _make_dirs_nofollow(plan.output.parent)
    try:
        os.mkdir(plan.output.name, mode=0o700, dir_fd=parent)
        os.chown(
            plan.output.name,
            account.uid,
            account.gid,
            dir_fd=parent,
            follow_symlinks=False,
        )
    finally:
        os.close(parent)
    return plan.output


def _reset_mount_root() -> None:
    """Recreate the mount root empty and root-owned, so nothing a session left in it
    survives into the next."""
    root = Path(SAFE_MOUNT_ROOT)
    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    parent = _make_dirs_nofollow(Path(root.parent))
    try:
        os.mkdir(root.name, mode=0o755, dir_fd=parent)
    finally:
        os.close(parent)


def _make_dirs_nofollow(path: Path) -> int:
    """Open ``path`` as a directory fd, creating each missing component as a
    root-owned directory and refusing any component that is a symlink."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=fd)
            except FileExistsError:
                pass
            try:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
            except OSError as exc:
                raise ExecutionError(
                    f"Cannot create SSH mount path {path.as_posix()}: {part} is not "
                    "a directory"
                ) from exc
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _hand_over_tree(root: Path, account: SessionAccount) -> None:
    """Give the session ownership of a tree the backend created, never following a
    symlink in it."""
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

    def login_user(self) -> str:
        return self.account.name

    def wait_ready(self, timeout_sec: float = 30.0) -> int | None:
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
        return self._finish_sentinel.exists()

    def established_connections(self) -> int | None:
        if (proc_net_tcp := read_local_proc_net_tcp()) is None:
            return None
        return count_established_connections(proc_net_tcp, self._port)

    def output_size_bytes(self) -> int | None:
        if (output_path := self._output_path) is None:
            return None
        try:
            return _tree_size(output_path)
        except OSError as exc:
            logger.debug("SSH output size check failed: %s", exc)
            return None

    def collect_output(self, destination: Path, max_bytes: int | None) -> None:
        """Copy the session's output into ``destination``: regular files and
        directories only, read without following a symlink."""
        if (output_path := self._output_path) is None:
            return
        # The session has ended; with its processes gone the tree holds still.
        kill_processes(self.account.uid)
        self._signals.raise_if_cancelled()
        destination.mkdir(parents=True, exist_ok=True)
        _copy_tree_nofollow(
            output_path, destination, max_bytes, self._signals.raise_if_cancelled
        )

    def stop(self, timeout_sec: float) -> None:
        _terminate(self._process, timeout_sec)
        kill_processes(self.account.uid)

    def cleanup(self) -> None:
        try:
            _discard_session(self._process, self.account, self._session_dir)
        finally:
            # A failure above must not strand the worker refusing every later
            # session; the reap before the next session is the backstop.
            self._backend._release(self)

    def _log_tail(self, max_chars: int = 2000) -> str:
        text = _read_log(self._log_path)
        return f"\nsshd output:\n{text[-max_chars:]}" if text else ""


@dataclass(slots=True)
class SessionManifest:
    """What a session allocated outside its worker's memory, so a worker that died
    with the session up can reap it."""

    session_dir: Path
    account: str
    roots: list[str]
    uid: int | None = None
    sshd_pid: int | None = None

    def write(self) -> None:
        path = self.session_dir / _MANIFEST_NAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "account": self.account,
                    "uid": self.uid,
                    "roots": self.roots,
                    "sshd_pid": self.sshd_pid,
                }
            ),
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
                roots=[str(root) for root in raw.get("roots") or []],
                uid=raw.get("uid"),
                sshd_pid=raw.get("sshd_pid"),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None


def reap_session(session_dir: Path) -> None:
    """Undo one session a dead worker left behind, as its manifest records it."""
    manifest = SessionManifest.read(session_dir)
    if manifest is not None:
        config_path = (session_dir / "sshd_config").as_posix()
        if manifest.sshd_pid is not None and _is_our_sshd(
            manifest.sshd_pid, config_path
        ):
            try:
                os.kill(manifest.sshd_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if manifest.uid is not None:
            kill_processes(manifest.uid)
            lift_denials(manifest.uid, [Path(root) for root in manifest.roots])
        delete_account(manifest.account)
        logger.info("Reaped SSH session %s left by an earlier worker", session_dir.name)
    shutil.rmtree(session_dir, ignore_errors=True)


def _is_our_sshd(pid: int, config_path: str) -> bool:
    """Whether ``pid`` is the sshd started with ``config_path``, so a recycled pid
    is never signalled.

    sshd rewrites its process title, so the check reads the command line as one
    string.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"sshd" in cmdline and config_path.encode() in cmdline


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


def _tree_size(root: Path) -> int:
    """Total size of the regular files under ``root``, without following links."""
    total = 0
    for current, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            info = os.lstat(os.path.join(current, name))
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _copy_tree_nofollow(
    source: Path,
    destination: Path,
    max_bytes: int | None,
    check: Callable[[], Any],
) -> None:
    total = 0
    for current, dirs, files in os.walk(source, followlinks=False):
        relative = Path(current).relative_to(source)
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in dirs:
            if os.path.islink(os.path.join(current, name)):
                logger.warning("Skipping symlink %s in SSH output", name)
        for name in files:
            check()
            path = os.path.join(current, name)
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError:
                logger.warning("Skipping %s in SSH output: not a regular file", name)
                continue
            with os.fdopen(fd, "rb") as src:
                if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                    logger.warning(
                        "Skipping %s in SSH output: not a regular file", name
                    )
                    continue
                total += os.fstat(src.fileno()).st_size
                if max_bytes is not None:
                    raise_if_exceeded(total, max_bytes)
                with (target_dir / name).open("wb") as dst:
                    while chunk := src.read(_COPY_CHUNK):
                        check()
                        dst.write(chunk)


def _discard_session(
    process: subprocess.Popen[bytes] | None,
    account: SessionAccount | None,
    session_dir: Path,
) -> None:
    """Undo whatever a session allocated, each step whether or not the one before it
    succeeded."""
    steps: list[Callable[[], Any]] = []
    if process is not None:
        steps.append(lambda: _terminate(process, _TERMINATE_GRACE_SEC))
    if account is not None:
        steps.append(account.release)
    steps.append(_reset_mount_root)
    steps.append(lambda: shutil.rmtree(session_dir, ignore_errors=True))
    for step in steps:
        try:
            step()
        except Exception:
            logger.warning("Failed to release part of an SSH session", exc_info=True)


def _terminate(process: subprocess.Popen[bytes], timeout_sec: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
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
        return subprocess.Popen(  # nosec B603 - argv list, no shell=True, absolute path via find_sshd()
            [sshd_path, "-D", "-e", "-f", config_path.as_posix()],
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
    """Environment for helper processes, carrying none of the worker's secrets.

    sshd would otherwise inherit the worker's environment, and a session's own
    shell is sshd's child.
    """
    env = {key: value for key in _SPAWN_ENV_KEYS if (value := os.environ.get(key))}
    env.setdefault("PATH", os.defpath)
    return env


def _render_sshd_config(
    port: int,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    login_user: str,
    exported_env: list[str] | None = None,
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
