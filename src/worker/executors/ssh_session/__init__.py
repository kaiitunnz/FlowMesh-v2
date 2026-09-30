"""SSH session backends and the configuration they share."""

from worker.config import WorkerConfig

from .backends.docker import DockerSessionBackend
from .base import (
    LOOPBACK_RELAY_HOST,
    SessionGone,
    SessionInterrupted,
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
)
from .config import (
    ResolvedSSHInput,
    SSHConfig,
    SSHOutputConfig,
    normalize_mount_path,
    reserve_mount_path,
)


def select_backend_cls(config: WorkerConfig) -> type[SSHSessionBackend] | None:
    """Resolve the session backend this worker should use, if any."""
    return DockerSessionBackend if DockerSessionBackend.is_available(config) else None


__all__ = [
    "LOOPBACK_RELAY_HOST",
    "DockerSessionBackend",
    "ResolvedSSHInput",
    "SSHConfig",
    "SSHOutputConfig",
    "SSHSession",
    "SSHSessionBackend",
    "SessionGone",
    "SessionInterrupted",
    "SessionRequest",
    "normalize_mount_path",
    "reserve_mount_path",
    "select_backend_cls",
]
