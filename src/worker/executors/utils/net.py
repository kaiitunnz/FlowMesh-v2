import socket

from ..base_executor import ExecutionError


def resolve_bind_port(requested: int | None, bind_host: str) -> int:
    """Resolve the port a server binds on ``bind_host``.

    When ``requested`` is ``None`` a free ephemeral port is selected; hardcoding
    a default (e.g. 8000) collides with co-located services on a host-networked
    node. When ``requested`` is set but unavailable, raise so the caller reports
    a clear error instead of a raw bind failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, requested or 0))
        except OSError as exc:
            raise ExecutionError(
                f"serve port {requested} is unavailable on the worker "
                f"({bind_host}): {exc}. Choose a different spec.port, or omit it "
                "to auto-select a free port."
            ) from exc
        return probe.getsockname()[1]
