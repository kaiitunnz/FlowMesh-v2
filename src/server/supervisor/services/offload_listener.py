"""The node's purpose-scoped listener for a directly dialed offload.

A trusted deployment lets an admitted invocation's origin open a connection straight to
the node hosting the selected replica. Each frame it reads goes to the node's local
sidecar uplink, and the worker's frames for that session answer over the same
connection, so neither direction enters the rendezvous.

Routing is by the frame's control-minted relay session, which the node resolves against
its own durable routing record: the dialer names only that session, never a host, port,
or engine endpoint. The frames stay opaque here — the replica worker's claim gate is the
only authority over what reaches an engine. The verified origin identity is bound to
each session the connection carries, so the node hands its local sidecar the same
origin control resolved.
"""

import logging

from shared.network.frame_stream import FrameSink
from shared.network.mtls import MutualTlsMaterial
from shared.network.mtls_listener import ConnectionHandler, MutualTlsFrameListener
from shared.network.relay_frame import RelayFrame

from ...resident.worker_bridge import ResidentWorkerBridge


class _UplinkConnection(ConnectionHandler):
    """Binds each session this connection carries to the node's local sidecar uplink."""

    def __init__(
        self, bridge: ResidentWorkerBridge, sink: FrameSink, origin: frozenset[str]
    ) -> None:
        self._bridge = bridge
        self._sink = sink
        self._origin = origin
        self._sessions: set[str] = set()

    async def on_frame(self, frame: RelayFrame) -> None:
        if frame.session_id not in self._sessions:
            self._sessions.add(frame.session_id)
            self._bridge.bind_offload(frame.session_id, self._sink)
        await self._bridge.on_frame(frame)

    def close(self) -> None:
        for session_id in self._sessions:
            self._bridge.release_offload(session_id)


def _is_registered_origin(identities: frozenset[str]) -> bool:
    """Whether the verified dialer carries an identity from the deployment's CA.

    The CA issues one only to a registered worker or node, so holding a verified
    identity is what admits the connection here; which invocation it may carry is the
    replica claim gate's decision, on the fenced handoff the frames deliver.
    """
    return bool(identities)


class NodeOffloadListener:
    """Serves the node's dialed offload connections into its local sidecar uplink."""

    def __init__(
        self,
        *,
        endpoint: str,
        material: MutualTlsMaterial | None,
        bridge: ResidentWorkerBridge,
        logger: logging.Logger | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._listener = MutualTlsFrameListener(
            material=material,
            handler=lambda sink, origin: _UplinkConnection(bridge, sink, origin),
            admits=_is_registered_origin,
            logger=logger or logging.getLogger("node-offload-listener"),
        )

    @property
    def port(self) -> int:
        return self._listener.port

    async def start(self) -> None:
        await self._listener.start_on_endpoint(self._endpoint)

    async def stop(self) -> None:
        await self._listener.stop()


__all__ = ["NodeOffloadListener"]
