"""SSH session backends and the configuration they share."""

from shared.schemas.worker import SSHBackendName
from worker.config import WorkerConfig

from .backends.docker import DockerSessionBackend
from .backends.process import ProcessSessionBackend
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

BACKENDS: dict[SSHBackendName, type[SSHSessionBackend]] = {
    DockerSessionBackend.name: DockerSessionBackend,
    ProcessSessionBackend.name: ProcessSessionBackend,
}


def select_backend_cls(config: WorkerConfig) -> type[SSHSessionBackend] | None:
    """Resolve the session backend this worker should use, if any."""
    requested = config.ssh_session_backend
    if requested is SSHBackendName.OFF:
        return None
    candidates = (
        (DockerSessionBackend, ProcessSessionBackend)
        if requested is SSHBackendName.AUTO
        else (BACKENDS[requested],)
    )
    return next((cls for cls in candidates if cls.is_available(config)), None)


__all__ = [
    "BACKENDS",
    "LOOPBACK_RELAY_HOST",
    "DockerSessionBackend",
    "ProcessSessionBackend",
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
