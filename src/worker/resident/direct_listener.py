"""The replica worker's claim-gated resident target-leg listener.

A trusted deployment lets the root carry a resident invocation's target leg straight
into this listener. Each frame it reads goes to the replica
sidecar, which answers over the same connection, so the sidecar's claim gate is the sole
authority over which traffic reaches the engine. It fronts the sidecar; the resident
engine listener stays loopback-only behind it.
"""

import logging
import socket
from collections.abc import Awaitable, Callable

from shared.network.mtls import MutualTlsMaterial
from shared.network.mtls_listener import (
    MAX_CONNECTIONS,
    ConnectionHandler,
    MutualTlsFrameListener,
)
from shared.network.relay_frame import RelayFrame
from shared.resident.transport import ResidentFrameSink

# Routes one frame into the replica sidecar, answering over the connection's own sink.
FrameDelivery = Callable[[RelayFrame, ResidentFrameSink], Awaitable[None]]


class _SidecarConnection(ConnectionHandler):
    """Hands one connection's frames to the replica sidecar."""

    def __init__(self, deliver: FrameDelivery, sink: ResidentFrameSink) -> None:
        self._deliver = deliver
        self._sink = sink

    async def on_frame(self, frame: RelayFrame) -> None:
        await self._deliver(frame, self._sink)

    def close(self) -> None:
        return None


class ResidentDirectListener:
    """Serves the worker's target-leg connections on the resident lane loop."""

    def __init__(
        self,
        *,
        sock: socket.socket,
        material: MutualTlsMaterial,
        deliver: FrameDelivery,
        max_connections: int = MAX_CONNECTIONS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sock = sock
        self._listener = MutualTlsFrameListener(
            material=material,
            handler=lambda sink: _SidecarConnection(deliver, sink),
            max_connections=max_connections,
            logger=logger or logging.getLogger("resident-direct-listener"),
        )

    @property
    def port(self) -> int:
        return self._listener.port

    async def start(self) -> None:
        await self._listener.start_on_socket(self._sock)

    async def stop(self) -> None:
        await self._listener.stop()


__all__ = ["ResidentDirectListener"]
