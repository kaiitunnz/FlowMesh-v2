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
from collections.abc import Callable

from shared.network.frame_stream import FrameSink

from .reverse_relay import (
    RESIDENT_RELAY_KEYSPACE,
    BinaryRedis,
    RelayDirection,
    RelayFrame,
    RelayKeyspace,
    RelaySessionStore,
    RelayStreamStore,
    StreamEntry,
)


class RootCursorStore:
    """The bridge's durable read position per attached node up stream."""

    def __init__(
        self, redis: BinaryRedis, keyspace: RelayKeyspace = RESIDENT_RELAY_KEYSPACE
    ) -> None:
        self._redis = redis
        self._key = keyspace.root_cursor

    async def get(self, node_id: str) -> str:
        raw = await self._redis.hgetall(self._key)
        value = raw.get(node_id.encode())
        return value.decode() if value else "0"

    async def set(self, node_id: str, entry_id: str) -> None:
        await self._redis.hset(self._key, mapping={node_id: entry_id})


# Selects the offloaded sink for one session from its routing record, or ``None`` when
# the session's frames toward the target belong on the target node's down stream.
OffloadSelector = Callable[[str, dict[str, str]], FrameSink | None]

# Counts one bridged frame and its payload bytes on a named leg and transport.
LegMeter = Callable[[str, str, int], None]

# The legs the bridge itself carries: an origin's frames up to the root, and a target's
# frames back to it over the relay.
SOURCE_TO_ROOT_LEG = "source_to_root"
TARGET_LEG = "target"
_RELAY = "control_relay"

# How many released sessions to remember. A lost session re-drives under a fresh session
# id, so this only has to outlive the frames already in flight for the old one.
_RELEASED_MEMORY = 1024


class RelayTargetSink:
    """Publishes a frame down to the target node its session's record names."""

    def __init__(self, streams: RelayStreamStore, sessions: RelaySessionStore) -> None:
        self._streams = streams
        self._sessions = sessions

    async def send(self, frame: RelayFrame) -> None:
        record = await self._sessions.load(frame.session_id)
        if destination := record.get("target_node"):
            await self._streams.publish_down(destination, frame)


class RelayOriginSink:
    """Publishes a frame down to the origin node its session's record names."""

    def __init__(self, streams: RelayStreamStore, sessions: RelaySessionStore) -> None:
        self._streams = streams
        self._sessions = sessions

    async def send(self, frame: RelayFrame) -> None:
        record = await self._sessions.load(frame.session_id)
        if destination := record.get("origin_node"):
            await self._streams.publish_down(destination, frame)


class RootRendezvousBridge:
    """Bridges opaque relay frames between attached nodes by their session routing.

    A session the wiring selects an offloaded sink for has its frames toward the target
    carried over that sink; the target's frames come back through the same sink's own
    delivery and publish down to the origin node, so only the target leg moves. The
    bridge selects and forwards, reading a frame's routing and flow-control fields as it
    always does.
    """

    def __init__(
        self,
        streams: RelayStreamStore,
        sessions: RelaySessionStore,
        cursors: RootCursorStore,
        *,
        offload_for: OffloadSelector | None = None,
        meter: LegMeter | None = None,
        batch: int = 64,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = streams
        self._sessions = sessions
        self._cursors = cursors
        self._offload_for = offload_for
        self._meter = meter
        self._sinks: dict[str, FrameSink] = {}
        self._released: OrderedDict[str, None] = OrderedDict()
        self._batch = batch
        self._logger = logger or logging.getLogger("network-rendezvous")

    async def pump_node(self, node_id: str) -> int:
        """Forward one bounded batch from a node's up stream; return the count read.

        The read is non-blocking: this driver multiplexes every node's up stream on one
        loop, so a blocking read on an idle node would wedge the whole cluster's bridge.
        """
        after = await self._cursors.get(node_id)
        entries, last_id = await self._streams.read_up(
            node_id, after, count=self._batch, block_ms=None
        )
        if last_id is None:
            return 0
        for entry in self._fair_order(entries):
            await self._forward(entry)
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

    async def _forward(self, entry: StreamEntry) -> None:
        frame = entry.frame
        record = await self._sessions.load(frame.session_id)
        if not record:
            self._logger.warning("relay frame for unknown session %s", frame.session_id)
            return
        if frame.direction is RelayDirection.TARGET_TO_ORIGIN:
            # A target's frame that reached the root over the relay: this session's
            # target leg was not offloaded.
            self._meter_leg(TARGET_LEG, len(frame.payload))
            if destination := record.get("origin_node"):
                await self._streams.publish_down(destination, frame)
            return
        self._meter_leg(SOURCE_TO_ROOT_LEG, len(frame.payload))
        sink = self._target_sink(frame.session_id, record)
        if sink is None:
            self._meter_leg(TARGET_LEG, len(frame.payload))
            if destination := record.get("target_node"):
                await self._streams.publish_down(destination, frame)
            return
        try:
            await sink.send(frame)
        except OSError as exc:
            # The offloaded sink ended mid-session, so this delivery is ambiguous: drop
            # the sink and leave the outcome to the endpoints that own it.
            self._logger.warning(
                "offloaded target leg lost for session %s: %s", frame.session_id, exc
            )
            self.release(frame.session_id)

    def _meter_leg(self, leg: str, payload_bytes: int) -> None:
        if self._meter is not None:
            self._meter(leg, _RELAY, payload_bytes)

    def _target_sink(self, session_id: str, record: dict[str, str]) -> FrameSink | None:
        """The offloaded sink for this session, or ``None`` for the down stream."""
        if self._offload_for is None or session_id in self._released:
            return None
        if (sink := self._sinks.get(session_id)) is not None:
            return sink
        sink = self._offload_for(session_id, record)
        if sink is not None:
            self._sinks[session_id] = sink
        return sink

    def release(self, session_id: str) -> None:
        """Drop one session's offloaded sink, on its terminal or its reap.

        A released session is remembered briefly so a late frame for it takes the target
        node's down stream rather than re-opening the leg it just lost.
        """
        self._sinks.pop(session_id, None)
        self._released[session_id] = None
        while len(self._released) > _RELEASED_MEMORY:
            self._released.popitem(last=False)


__all__ = [
    "SOURCE_TO_ROOT_LEG",
    "TARGET_LEG",
    "RelayOriginSink",
    "RelayTargetSink",
    "RootCursorStore",
    "RootRendezvousBridge",
]
