"""Session backend seam for the SSH executor.

An SSH session is a sandbox running ``sshd`` plus the transport details needed
to reach it. The task lifecycle, TTL reaping, ``emit_update`` and the
``accessMode`` enum sit above this seam, in the executor.
"""

import logging
import os
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from shared.schemas.worker import SSHBackendName
from shared.tasks.worker_message import WorkerHardware
from worker.config import WorkerConfig

from ..base_executor import RunSignals
from .config import FINISH_SENTINEL_PATH, ResolvedSSHInput, SSHConfig

logger = logging.getLogger(__name__)

LOOPBACK_RELAY_HOST = "127.0.0.1"


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
    def output_size_bytes(self) -> int | None:
        """Current size of the session's output directory.

        ``None`` when the session has no output directory or its size cannot be
        measured right now.
        """

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

    @abstractmethod
    def start_session(self, request: SessionRequest) -> SSHSession:
        """Create and start a session.

        Raises ``SessionInterrupted`` when a cancel or stop reaches the task before
        the session starts, having released whatever it allocated.
        """

    @abstractmethod
    def teardown(self, worker_name: str) -> None:
        """Reap any sessions ``worker_name`` still owns."""

    def relay_host(self) -> str:
        """Address at which this worker's session ports are reachable.

        The supervisor that dials the relay uplink is the consumer: it opens a
        TCP connection to this address, so it must be routable *from the
        supervisor*, not from the worker.
        """
        return LOOPBACK_RELAY_HOST

    def session_host(self) -> str:
        """Host name reported to the user as the session's location."""
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
            if authorized_keys:
                env["AUTHORIZED_KEYS"] = "\n".join(authorized_keys)
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


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total
