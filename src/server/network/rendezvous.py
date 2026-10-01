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
the shared per-node stream. The durable cursor advances after each batch, so a bridge
that restarts mid-batch (or errors before recording the cursor) re-reads and may
re-forward a frame; the receiving endpoint drops the re-forward by its per-direction
sequence, so a data frame lands once within a receiver's lifetime. That dedup is an
in-memory high-water mark, so a receiver restart can re-deliver a re-forwarded frame;
grants are idempotent and chunks reassemble, but the guarantee is not exactly-once
across a receiver restart. A forwarded prefix is trimmed at or below the recorded
cursor, bounding the per-node stream.
"""

import logging
from collections import OrderedDict, deque

from .reverse_relay import (
    BinaryRedis,
    RelayDirection,
    RelayKeyspace,
    RelaySessionStore,
    RelayStreamStore,
    StreamEntry,
)


class RootCursorStore:
    """The bridge's durable read position per attached node up stream."""

    def __init__(self, redis: BinaryRedis, keyspace: RelayKeyspace) -> None:
        self._redis = redis
        self._key = keyspace.root_cursor

    async def get(self, node_id: str) -> str:
        return (await self.get_many([node_id]))[node_id]

    async def get_many(self, node_ids: list[str]) -> dict[str, str]:
        raw = await self._redis.hgetall(self._key)
        return {
            node_id: value.decode() if (value := raw.get(node_id.encode())) else "0"
            for node_id in node_ids
        }

    async def set(self, node_id: str, entry_id: str) -> None:
        await self._redis.hset(self._key, mapping={node_id: entry_id})


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
        entries, last_id = await self._streams.read_up(
            node_id, after, count=self._batch, block_ms=None
        )
        return await self._drain(node_id, entries, last_id)

    async def pump_ready(self, node_ids: list[str], block_ms: int) -> int:
        """Wait up to ``block_ms`` for any node's up stream, then forward one bounded
        batch from each stream that has frames; return the count read.

        One read covers every node, so a frame on any stream wakes the bridge at once
        and an idle node never holds the others up.
        """
        cursors = await self._cursors.get_many(node_ids)
        batches = await self._streams.read_up_many(
            cursors, count=self._batch, block_ms=block_ms
        )
        read = 0
        for node_id, (entries, last_id) in batches.items():
            read += await self._drain(node_id, entries, last_id)
        return read

    async def _drain(
        self, node_id: str, entries: list[StreamEntry], last_id: str | None
    ) -> int:
        if last_id is None:
            return 0
        for entry in self._fair_order(entries):
            await self._forward(node_id, entry)
        await self._cursors.set(node_id, last_id)
        # Trim the forwarded prefix of this node's up stream at or below the recorded
        # cursor so it stays bounded; unforwarded frames past the cursor are never cut.
        await self._streams.trim_up_to(
            node_id, RelayDirection.ORIGIN_TO_TARGET, last_id
        )
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

    async def _forward(self, source_node: str, entry: StreamEntry) -> None:
        """Forward one frame to its peer node, only from the end that may send it.

        A frame travels origin-to-target only off the record's origin node stream and
        target-to-origin only off its target node stream, so a frame another node
        writes naming a live session is dropped rather than injected into it.
        """
        frame = entry.frame
        record = await self._sessions.load(frame.session_id)
        if not record:
            self._logger.warning("relay frame for unknown session %s", frame.session_id)
            return
        if frame.direction is RelayDirection.ORIGIN_TO_TARGET:
            sender, destination = record.get("origin_node"), record.get("target_node")
        else:
            sender, destination = record.get("target_node"), record.get("origin_node")
        if sender != source_node:
            self._logger.warning(
                "dropping relay frame for session %s read from node %s, not its sender",
                frame.session_id,
                source_node,
            )
            return
        if destination:
            await self._streams.publish_down(destination, frame)


__all__ = ["RootCursorStore", "RootRendezvousBridge"]
