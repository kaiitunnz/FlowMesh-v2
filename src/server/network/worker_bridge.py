"""The node-local bridge: opaque frames between the relay and a co-located worker.

On each node the reverse-relay attachment hands every down frame here, and it forwards
the frame — opaquely — to the co-located worker the session names: a frame travelling
toward the origin goes to the worker that opened the exchange, one travelling toward the
target goes to the worker serving it, chosen by the frame's direction. A worker's
produced frames publish to the up stream for the root to bridge onward. The bridge never
decodes a wire body, cursor, or window: the worker endpoints own the protocol. One
instance serves one namespace, so a node carrying both resident invocations and content
transfers runs one bridge per keyspace.
"""

import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from shared.network.frame_stream import FrameSink
from shared.network.relay_frame import (
    RESIDENT_FRAME_KIND,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
)

from .reverse_relay import (
    BinaryRedis,
    RelayKeyspace,
    RelaySessionStore,
    RelayStreamStore,
)

# Enqueues a dispatch payload straight to a co-located worker; True if it is local.
LocalEnqueue = Callable[[str, dict[str, Any]], Awaitable[bool]]

# Far more sessions than a node's dialed connections end while their worker could still
# be producing frames for one.
_DROPPED_MEMORY = 4096


class RelayWorkerBridge:
    """Forwards one namespace's relay frames between the relay and local workers."""

    def __init__(
        self,
        redis: BinaryRedis,
        node_id: str,
        enqueue_local: LocalEnqueue,
        *,
        keyspace: RelayKeyspace,
        frame_kind: str = RESIDENT_FRAME_KIND,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(redis, keyspace)
        self._sessions = RelaySessionStore(redis, keyspace)
        self._node_id = node_id
        self._enqueue_local = enqueue_local
        self._frame_kind = frame_kind
        self._peers: dict[str, FrameSink] = {}
        self._dropped: OrderedDict[str, None] = OrderedDict()
        self._logger = logger or logging.getLogger("relay-worker-bridge")

    async def on_frame(self, frame: RelayFrame) -> None:
        """Forward one down frame to the local worker its session and direction name."""
        await self.forward(frame, await self.session(frame.session_id))

    async def forward(self, frame: RelayFrame, record: Mapping[str, str]) -> None:
        """Forward one frame by its session's routing record, already loaded.

        A session's record never changes, so a connection carrying many of its frames
        loads it once.
        """
        if not record:
            return
        if frame.direction is RelayDirection.TARGET_TO_ORIGIN:
            worker_id = record.get("origin_worker")
        else:
            worker_id = record.get("target_worker")
        if not worker_id:
            return
        await self._enqueue_local(
            worker_id,
            {
                "kind": "mediated_op",
                "frame_kind": self._frame_kind,
                "payload": frame.to_wire(),
            },
        )

    def rebind(self, node_id: str) -> None:
        """Publish worker frames on the up stream of the node's new id."""
        self._node_id = node_id

    @property
    def node_id(self) -> str:
        return self._node_id

    async def session(self, session_id: str) -> dict[str, str]:
        """Load one session's control-written routing record; empty when it has none."""
        return await self._sessions.load(session_id)

    def bind_peer(self, session_id: str, sink: FrameSink) -> None:
        """Answer one session's worker frames over the connection its origin dialed.

        The binding is taken from the first admitted dialer to name the session and held
        until that connection releases it: the forward direction is
        claim-gated at the replica, while this reverse direction rests on the session id
        being unguessable and the dialer holding a deployment identity. Naming the
        expected origin to a target would need control to carry it.
        """
        self._peers.setdefault(session_id, sink)

    def release_peer(self, session_id: str, sink: FrameSink) -> None:
        """Drop a session's binding, if ``sink`` is the connection holding it."""
        if self._peers.get(session_id) is sink:
            del self._peers[session_id]

    async def drop_peer(
        self, session_id: str, sink: FrameSink, record: Mapping[str, str]
    ) -> None:
        """End a session whose dialed connection is gone, at the worker serving it.

        The worker is cancelled, and anything it still produces for the session is
        discarded here: its origin read it only over that connection.
        """
        if self._peers.get(session_id) is not sink:
            return
        del self._peers[session_id]
        self._dropped[session_id] = None
        while len(self._dropped) > _DROPPED_MEMORY:
            self._dropped.popitem(last=False)
        await self.forward(
            RelayFrame(
                kind=RelayFrameKind.CANCEL,
                session_id=session_id,
                correlation_id="",
                operation_id="",
                direction=RelayDirection.ORIGIN_TO_TARGET,
            ),
            record,
        )

    async def publish_up(self, frame: RelayFrame) -> None:
        """Return a worker's produced frame to the origin that is waiting for it.

        A session an origin dialed answers over that same connection, so its frames
        never enter the rendezvous; every other session publishes to this node's up
        stream for the root to bridge onward.
        """
        if (sink := self._peers.get(frame.session_id)) is not None:
            await sink.send(frame)
            return
        if frame.session_id in self._dropped:
            return
        await self._streams.publish_up(self._node_id, frame)
