"""The root rendezvous bridge for the reverse-rendezvous relay.

Origins and targets both attach outward to the root by writing their per-node ``:up``
stream and reading their per-node ``:down`` stream. The bridge is the only party that
moves a frame between two nodes: it reads a node's up stream from a durable cursor and,
for each frame, forwards it to the peer node's down stream chosen by the session's
routing record. It reads only the routing and flow-control fields — session, direction,
sequence, acknowledgement, window — and forwards the fence and payload opaquely. It
holds no admission, credit, or engine authority.

Draining is fair: within a read batch, priority control frames go first, then data
frames are round-robined across sessions so one busy session cannot starve another on
the shared per-node stream. The durable cursor advances in stream order, so a restarted
bridge re-reads from the last recorded position and any re-forwarded frame is
deduplicated by its receiver.
"""

import logging
from collections import OrderedDict, deque

from .reverse_relay import (
    BinaryRedis,
    RelayDirection,
    RelaySessionStore,
    RelayStreamStore,
    StreamEntry,
)


class RootCursorStore:
    """The bridge's durable read position per attached node up stream."""

    _KEY = "rr:root:up_cursor"

    def __init__(self, redis: BinaryRedis) -> None:
        self._redis = redis

    async def get(self, node_id: str) -> str:
        raw = await self._redis.hgetall(self._KEY)
        value = raw.get(node_id.encode())
        return value.decode() if value else "0"

    async def set(self, node_id: str, entry_id: str) -> None:
        await self._redis.hset(self._KEY, mapping={node_id: entry_id})


class RootRendezvousBridge:
    """Bridges opaque relay frames between attached nodes by their session routing."""

    def __init__(
        self,
        streams: RelayStreamStore,
        sessions: RelaySessionStore,
        cursors: RootCursorStore,
        *,
        batch: int = 64,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = streams
        self._sessions = sessions
        self._cursors = cursors
        self._batch = batch
        self._logger = logger or logging.getLogger("network-rendezvous")

    async def pump_node(self, node_id: str) -> int:
        """Forward one bounded batch from a node's up stream; return the count read."""
        after = await self._cursors.get(node_id)
        entries = await self._streams.read_up(
            node_id, after, count=self._batch, block_ms=0
        )
        if not entries:
            return 0
        for entry in self._fair_order(entries):
            await self._forward(entry)
        await self._cursors.set(node_id, entries[-1].entry_id)
        return len(entries)

    @staticmethod
    def _fair_order(entries: list[StreamEntry]) -> list[StreamEntry]:
        control = [e for e in entries if e.frame.is_control]
        buckets: OrderedDict[str, deque[StreamEntry]] = OrderedDict()
        for entry in entries:
            if entry.frame.is_control:
                continue
            buckets.setdefault(entry.frame.session_id, deque()).append(entry)
        rotated: list[StreamEntry] = []
        while buckets:
            for session_id in list(buckets):
                queue = buckets[session_id]
                rotated.append(queue.popleft())
                if not queue:
                    del buckets[session_id]
        return control + rotated

    async def _forward(self, entry: StreamEntry) -> None:
        frame = entry.frame
        record = await self._sessions.load(frame.session_id)
        if not record:
            self._logger.warning("relay frame for unknown session %s", frame.session_id)
            return
        if frame.direction is RelayDirection.ORIGIN_TO_TARGET:
            destination = record.get("target_node")
        else:
            destination = record.get("origin_node")
        if destination:
            await self._streams.publish_down(destination, frame)


__all__ = ["RootCursorStore", "RootRendezvousBridge"]
