"""The node-local resident bridge: opaque frames between the relay and a worker.

On each node the reverse-relay attachment hands every down frame here, and it forwards
the frame — opaquely — to the co-located worker the session names: a response toward an
invocation this node originated goes to the origin worker, a request toward a replica
this node hosts goes to the replica worker, chosen by the frame's direction. A worker's
produced frames publish to the up stream for the root to bridge onward, or return over
the target-leg connection the root opened for that session. The bridge never decodes a
resident wire body, cursor, or window: the worker endpoints own the protocol.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from shared.network.relay_frame import RelayDirection, RelayFrame
from shared.resident.transport import ResidentFrameSink

from ..network.reverse_relay import BinaryRedis, RelaySessionStore, RelayStreamStore

# Enqueues a dispatch payload straight to a co-located worker; True if it is local.
LocalEnqueue = Callable[[str, dict[str, Any]], Awaitable[bool]]


class ResidentWorkerBridge:
    """Forwards resident relay frames between the reverse-relay and local workers."""

    def __init__(
        self,
        redis: BinaryRedis,
        node_id: str,
        enqueue_local: LocalEnqueue,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(redis)
        self._sessions = RelaySessionStore(redis)
        self._node_id = node_id
        self._enqueue_local = enqueue_local
        self._target_legs: dict[str, ResidentFrameSink] = {}
        self._logger = logger or logging.getLogger("resident-worker-bridge")

    def bind_target_leg(self, session_id: str, sink: ResidentFrameSink) -> None:
        """Return one session's worker frames over the root's target-leg connection."""
        self._target_legs[session_id] = sink

    def release_target_leg(self, session_id: str) -> None:
        self._target_legs.pop(session_id, None)

    async def on_frame(self, frame: RelayFrame) -> None:
        """Forward one down frame to the local worker its session and direction name."""
        record = await self._sessions.load(frame.session_id)
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
                "frame_kind": "resident_frame",
                "payload": frame.to_wire(),
            },
        )

    async def publish_up(self, frame: RelayFrame) -> None:
        """Carry a worker's produced frame back toward the root.

        A session the root reached over a target-leg connection answers on that
        connection; every other session publishes to this node's up stream for the root
        rendezvous to bridge onward.
        """
        sink = self._target_legs.get(frame.session_id)
        if sink is not None:
            await sink.send(frame)
            return
        await self._streams.publish_up(self._node_id, frame)
