"""The node's purpose-scoped resident target-leg listener.

A trusted deployment lets the root open a resident invocation's target leg straight to
the node hosting the selected replica. Each frame it reads goes to the node's local
sidecar uplink, and the worker's frames for that session answer over the same
connection.

Routing is by the frame's control-minted relay session, which the node resolves against
its own durable routing record: the dialer names only that session, never a host, port,
or engine endpoint. The frames stay opaque here — the replica worker's claim gate is the
only authority over what reaches an engine.
"""

import logging

from shared.network.frame_stream import FrameSink
from shared.network.mtls import MutualTlsMaterial
from shared.network.mtls_listener import ConnectionHandler, MutualTlsFrameListener
from shared.network.relay_frame import RelayFrame

from ...resident.worker_bridge import ResidentWorkerBridge


class _UplinkConnection(ConnectionHandler):
    """Binds each session this connection carries to the node's local sidecar uplink."""

    def __init__(self, bridge: ResidentWorkerBridge, sink: FrameSink) -> None:
        self._bridge = bridge
        self._sink = sink
        self._sessions: set[str] = set()

    async def on_frame(self, frame: RelayFrame) -> None:
        if frame.session_id not in self._sessions:
            self._sessions.add(frame.session_id)
            self._bridge.bind_target_leg(frame.session_id, self._sink)
        await self._bridge.on_frame(frame)

    def close(self) -> None:
        for session_id in self._sessions:
            self._bridge.release_target_leg(session_id)


class NodeTargetLegListener:
    """Serves the node's target-leg connections into its local sidecar uplink."""

    def __init__(
        self,
        *,
        endpoint: str,
        material: MutualTlsMaterial,
        bridge: ResidentWorkerBridge,
        logger: logging.Logger | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._listener = MutualTlsFrameListener(
            material=material,
            handler=lambda sink: _UplinkConnection(bridge, sink),
            logger=logger or logging.getLogger("node-target-leg-listener"),
        )

    @property
    def port(self) -> int:
        return self._listener.port

    async def start(self) -> None:
        await self._listener.start_on_endpoint(self._endpoint)

    async def stop(self) -> None:
        await self._listener.stop()


__all__ = ["NodeTargetLegListener"]
