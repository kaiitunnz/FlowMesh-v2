"""Session backend seam for the SSH executor.

An SSH session is a sandbox running ``sshd`` plus the transport details needed
to reach it. A worker with a Docker socket puts the session in a sibling
container; a root worker without one runs it as a process under an account of its
own. The executor above it owns the task lifecycle and TTL and idle reaping.
"""

import errno
import io
import ipaddress
import logging
import os
import re
import shutil
import socket
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import psutil

from shared.schemas.worker import SSHBackendName
from shared.tasks.worker_message import WorkerHardware
from worker.config import WorkerConfig

from ..base_executor import ExecutionError, RunSignals
from .config import (
    FINISH_SENTINEL_PATH,
    ResolvedSSHInput,
    SSHConfig,
    raise_if_exceeded,
)

logger = logging.getLogger(__name__)

LOOPBACK_BIND_HOST = "127.0.0.1"
ANY_BIND_HOST = "0.0.0.0"  # nosec B104 - a direct session must be dialable
LOOPBACK_SCOPE = "loopback"
NETWORK_SCOPE = "network"

# Tailscale hands every node an address out of the CGNAT range, which is how a
# rented box advertises an address a client can dial.
TAILNET_NETWORK = ipaddress.ip_network("100.64.0.0/10")

_TCP_STATE_ESTABLISHED = "01"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VALUE_FORBIDDEN = ('"', "\\", "\n", "\r")
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# A directory removed or swapped for a link mid-walk.
_GONE_ERRNOS = frozenset({errno.ENOENT, errno.ELOOP, errno.ENOTDIR})
_MAX_TREE_DEPTH = 64


class SessionInterrupted(Exception):
    """A cancel or stop reached an SSH task before its session started."""


class SessionGone(Exception):
    """A running session vanished without exiting."""


@dataclass(slots=True)
class SessionRequest:
    """Everything a backend needs to bring one session up."""

    task_id: str
    session_id: str
    worker_name: str
    cfg: SSHConfig
    out_dir: Path
    resolved_inputs: list[ResolvedSSHInput]
    signals: RunSignals


class SSHSession(ABC):
    """A session that has been started by a backend."""

    @abstractmethod
    def wait_ready(self, timeout_sec: float) -> int | None:
        """Block until sshd accepts connections; return its reachable port.

        Returns ``None`` for a session stopped first, and raises
        ``TaskCancelledError`` for one cancelled first.
        """

    @abstractmethod
    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while the session is still up.

        Raises ``SessionGone`` when the session disappeared without exiting.
        """

    @abstractmethod
    def finish_requested(self) -> bool:
        """Whether the session asked to finish via the in-session helper."""

    @abstractmethod
    def established_connections(self) -> int | None:
        """Count of established SSH connections, or ``None`` when unobservable.

        ``None`` means the idle reaper has no evidence either way and must not
        reap.
        """

    @abstractmethod
    def output_size_bytes(self) -> int | None:
        """Current size of the session's output directory, or ``None`` when it has
        none or it cannot be measured."""

    @abstractmethod
    def collect_output(self, destination: Path, max_bytes: int | None) -> None:
        """Copy the session's output directory into ``destination``, failing once
        it passes ``max_bytes``."""

    @abstractmethod
    def login_user(self) -> str:
        """Username this session accepts, as reported to the client."""

    @abstractmethod
    def stop(self, timeout_sec: float) -> None:
        """Ask the session to terminate, escalating after ``timeout_sec``."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release everything the session allocated."""

    def drain_logs(self) -> None:
        """Forward session output to the worker log until the stream closes."""
        return None

    def save_logs(self, out_dir: Path) -> None:
        """Persist session output under ``out_dir`` as a fallback capture."""
        return None


class SSHSessionBackend(ABC):
    """Creates and reaps SSH sessions for one worker."""

    name: ClassVar[SSHBackendName]
    supports_noninteractive: ClassVar[bool] = True
    """Whether the backend can run a user-supplied image non-interactively."""

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        self._config = config
        self._hardware = hardware

    @classmethod
    @abstractmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        """Whether this backend can create sessions on this worker."""

    @abstractmethod
    def prepare(self) -> None:
        """Initialize whatever the backend needs before the first session."""

    def reap_stale(self) -> None:
        """Remove what sessions left behind when an earlier worker died."""
        return None

    @abstractmethod
    def start_session(self, request: SessionRequest) -> SSHSession:
        """Create and start a session.

        Raises ``SessionInterrupted`` when a cancel or stop reaches the task before
        the session starts, having released whatever it allocated.
        """

    @abstractmethod
    def teardown(self, worker_name: str) -> None:
        """Reap any sessions ``worker_name`` still owns."""

    def session_bind_host(self, access_mode: str) -> str:
        """Return the address the session's sshd listens on.

        Only a ``direct`` session is dialled from outside the worker; a relayed one is
        reached over loopback by the worker's own relay lane.
        """
        return ANY_BIND_HOST if access_mode == "direct" else LOOPBACK_BIND_HOST

    def session_host(self) -> str:
        """Return the host name a client reaches a network-bound session at."""
        if override := self._config.ssh_direct_host:
            return override
        return self._default_session_host()

    def session_address(self, access_mode: str) -> str:
        """Return the address a client uses to reach the session.

        Derived from the bind so the two cannot disagree: a session bound to loopback
        is reachable only from the worker's own host, whatever name the worker
        otherwise answers to.
        """
        if self.session_scope(access_mode) == NETWORK_SCOPE:
            return self.session_host()
        return LOOPBACK_BIND_HOST

    def session_scope(self, access_mode: str) -> str:
        """Return which addresses the session accepts connections on.

        ``loopback`` reaches it only from the worker's own host; ``network`` says it
        is bound beyond loopback, not that any given client can route to the worker.
        """
        if self.session_bind_host(access_mode) == ANY_BIND_HOST:
            return NETWORK_SCOPE
        return LOOPBACK_SCOPE

    def _default_session_host(self) -> str:
        return socket.getfqdn()

    def _build_environment(
        self,
        user: str,
        authorized_keys: list[str],
        extra_env: dict[str, Any],
        staged_input_specs: list[tuple[str, str]],
        create_dirs: list[str],
        bootstrap_entrypoint: bool,
        gpu_device_ids: list[str] | None = None,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        if bootstrap_entrypoint:
            env["SSH_USER"] = user
            # sshd starts a login with a clean environment, so the task's env rides
            # each key as an option, under the names the entrypoint permits.
            rendered, exported = render_authorized_keys(
                authorized_keys, {str(k): str(v) for k, v in extra_env.items()}
            )
            if rendered:
                env["AUTHORIZED_KEYS"] = rendered.rstrip("\n")
            if exported:
                env["FLOWMESH_PERMIT_ENV"] = ",".join(exported)
            env["SSH_UID"] = str(os.getuid())
            env["SSH_GID"] = str(os.getgid())
        if gpu_device_ids and (visible := self._cuda_visible_devices(gpu_device_ids)):
            env["CUDA_VISIBLE_DEVICES"] = visible
        if staged_input_specs:
            env["FLOWMESH_STAGED_INPUT_SPECS"] = "\n".join(
                f"{mount_path}\t{target_path}"
                for mount_path, target_path in staged_input_specs
            )
        if create_dirs:
            env["FLOWMESH_CREATE_DIRS"] = "\n".join(create_dirs)
        env["FLOWMESH_FINISH_SENTINEL"] = FINISH_SENTINEL_PATH
        for k, v in extra_env.items():
            env[str(k)] = str(v)
        return env

    def _cuda_visible_devices(self, gpu_device_ids: list[str]) -> str | None:
        return ",".join(gpu_device_ids)


def is_ssh_ready(host: str, port: int) -> bool:
    """Whether something on ``host:port`` answers with an SSH banner."""
    try:
        with socket.create_connection((host, port), timeout=1.0) as sock:
            sock.settimeout(1.0)
            banner = sock.recv(64)
            return banner.startswith(b"SSH-")
    except OSError:
        return False


class TreeTooDeepError(OSError):
    """A tree is nested deeper than :func:`iter_tree` descends."""


def iter_tree(root: Path) -> Iterator[tuple[PurePosixPath, os.DirEntry[str], int]]:
    """Walk ``root`` without following a link anywhere, at any time.

    Yields ``(relative path, entry, parent directory fd)`` for every entry, parents
    before children; the fd stays open until the walk resumes. Each directory is
    opened relative to its parent with ``O_NOFOLLOW``, so a component swapped for a
    link mid-walk is skipped rather than followed, which keeps the walk safe over a
    tree someone else is still writing. A missing or linked ``root`` yields nothing;
    a directory it cannot open, or a tree deeper than the walk will go, raises
    rather than being skipped.
    """
    try:
        fd = os.open(root, _DIR_OPEN_FLAGS)
    except OSError as exc:
        if exc.errno in _GONE_ERRNOS:
            return
        raise
    try:
        yield from _iter_dir(fd, PurePosixPath(), 0)
    finally:
        os.close(fd)


def _iter_dir(
    dir_fd: int, relative: PurePosixPath, depth: int
) -> Iterator[tuple[PurePosixPath, os.DirEntry[str], int]]:
    if depth >= _MAX_TREE_DEPTH:
        raise TreeTooDeepError(
            f"{relative.as_posix()} is nested deeper than {_MAX_TREE_DEPTH} levels"
        )
    with os.scandir(dir_fd) as scanner:
        entries = list(scanner)
    for entry in entries:
        entry_path = relative / entry.name
        yield entry_path, entry, dir_fd
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            child_fd = os.open(entry.name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in _GONE_ERRNOS:
                continue
            raise OSError(exc.errno, exc.strerror, entry_path.as_posix()) from exc
        try:
            yield from _iter_dir(child_fd, entry_path, depth + 1)
        finally:
            os.close(child_fd)


def path_size_bytes(path: Path) -> int:
    """Total size of the regular files under ``path``, never following a link.

    Raises rather than under-report: a size that skipped part of the tree would let
    the output outgrow its limit unnoticed.
    """
    total = 0
    try:
        for _, entry, _ in iter_tree(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    except OSError as exc:
        raise ExecutionError(f"Cannot measure session output: {exc}") from exc
    return total


class _ChunkReader(io.RawIOBase):
    """A readable stream over an iterator of byte chunks that calls ``check`` on
    each read."""

    def __init__(self, chunks: Iterable[bytes], check: Callable[[], Any]) -> None:
        self._chunks = iter(chunks)
        self._check = check
        self._pending = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._pending:
            try:
                self._pending = memoryview(next(self._chunks))
            except StopIteration:
                return 0
        # Closing a Docker stream before its first read leaves its connection open.
        self._check()
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


def extract_output_archive(
    chunks: Iterable[bytes],
    destination: Path,
    max_bytes: int | None,
    check: Callable[[], Any],
    source_name: str | None = None,
) -> None:
    """Extract a streamed tar of a session's output into ``destination``: its
    directories and regular files only, failing once the files pass ``max_bytes``,
    before any more of them is read.

    ``source_name`` strips the archive's own top-level directory, as Docker's
    archive of a container path carries it.
    """
    total = 0
    with tarfile.open(fileobj=_ChunkReader(chunks, check), mode="r|") as archive:
        for member in archive:
            # A streamed archive otherwise keeps every member it has read.
            archive.members = []  # type: ignore[attr-defined]
            check()
            relative = _relative_archive_path(member.name, source_name)
            if relative is None:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            total += member.size
            if max_bytes is not None:
                raise_if_exceeded(total, max_bytes)
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with target.open("wb") as fh:
                shutil.copyfileobj(extracted, fh)


def _relative_archive_path(member_name: str, source_name: str | None) -> Path | None:
    parts = [
        part for part in PurePosixPath(member_name).parts if part not in ("", ".", "/")
    ]
    if source_name is not None and source_name in parts:
        parts = parts[parts.index(source_name) + 1 :]
    if not parts or any(part == ".." for part in parts):
        return None
    return Path(*parts)


def count_established_connections(proc_net_tcp: str, port: int) -> int:
    """Count established TCP connections to ``port`` in ``/proc/net/tcp`` text.

    Accepts the concatenation of ``/proc/net/tcp`` and ``/proc/net/tcp6``; both
    encode the local address as ``<hex address>:<hex port>``.
    """
    total = 0
    for line in proc_net_tcp.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].endswith(":"):
            continue
        local_address = fields[1]
        if ":" not in local_address:
            continue
        try:
            local_port = int(local_address.rsplit(":", 1)[1], 16)
        except ValueError:
            continue
        if local_port == port and fields[3] == _TCP_STATE_ESTABLISHED:
            total += 1
    return total


def read_local_proc_net_tcp() -> str | None:
    """Read this network namespace's TCP tables, or ``None`` when unreadable."""
    chunks: list[str] = []
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            chunks.append(Path(name).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks) if chunks else None


def resolve_tailnet_address() -> str | None:
    """Return this host's tailnet address, or ``None`` when it has none."""
    try:
        interfaces = psutil.net_if_addrs()
    except OSError:
        return None
    for addresses in interfaces.values():
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            try:
                parsed = ipaddress.ip_address(address.address)
            except ValueError:
                continue
            if parsed in TAILNET_NETWORK:
                return str(parsed)
    return None


def render_authorized_keys(
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
        if is_safe_env_entry(name, value)
    ]
    options = ",".join(f'environment="{name}={environment[name]}"' for name in exported)
    lines = [
        f"{options} {key}" if options else key
        for raw_key in authorized_keys
        if (key := raw_key.strip())
    ]
    return ("\n".join(lines) + "\n" if lines else "", exported if lines else [])


def is_safe_env_entry(name: str, value: str) -> bool:
    """Return whether an env entry can ride an ``environment=`` key option."""
    if not _ENV_NAME_RE.match(name):
        logger.warning("Dropping SSH session env var with unsupported name %r", name)
        return False
    if any(ch in value for ch in _ENV_VALUE_FORBIDDEN):
        logger.warning("Dropping SSH session env var %s: unsupported value", name)
        return False
    return True
