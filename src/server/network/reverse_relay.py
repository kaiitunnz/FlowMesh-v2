"""Reverse-rendezvous relay substrate over per-node outbound Redis streams.

The universal ``control_relay`` transport carries a resident invocation without either
end accepting an inbound connection: an origin and a target each attach outward to the
root rendezvous, which bridges opaque framed data between their per-node streams. This
module is the transport primitives — frame codec, per-node stream cursor reads, the
durable per-session record, and the ownership lease that hands a leg's single receiver
over on restart. It is deliberately free of any resident, claim, or admission concept:
a frame carries an opaque payload (the resident fence and body ride inside it) and is
keyed only by the relay session, invocation, and idempotency identifiers used for
routing and dedupe.

Durability follows a cursor lease, not a consumer group: each leg has one logical
receiver that reads its node stream from a durable stored cursor, and a restart reclaims
an owner-fenced lease and resumes from that cursor. Unacknowledged frames are never
trimmed — a stream is trimmed only at or below the cumulative-acknowledged id.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ..clients.redis import (
    resident_relay_down_key,
    resident_relay_session_key,
    resident_relay_up_key,
)

# A crashed origin can never trim its own session record; a generous TTL bounds the leak
# without expiring an active one — a single invocation's session lives far under it.
_SESSION_TTL_MS = 1_800_000

_logger = logging.getLogger("reverse-relay")


class RelayFrameKind(StrEnum):
    # Bulk DATA, plus the WINDOW grant and CANCEL that ride the priority control lane,
    # bypassing the byte window so a full data window cannot deadlock cancellation.
    DATA = "data"
    WINDOW = "window"
    CANCEL = "cancel"


class RelayDirection(StrEnum):
    """A frame's flow direction within a session."""

    ORIGIN_TO_TARGET = "o2t"
    TARGET_TO_ORIGIN = "t2o"


_CONTROL_KINDS = frozenset({RelayFrameKind.WINDOW, RelayFrameKind.CANCEL})


@dataclass(frozen=True)
class RelayFrame:
    """One relay frame. ``payload`` is opaque bytes the root never reads (the resident
    fence and body ride inside it); the rest are routing and flow-control metadata it
    may read. ``seq`` orders a direction's data for receiver dedup; ``ack`` carries a
    grant's cumulative byte credit."""

    kind: RelayFrameKind
    session_id: str
    invocation_id: str
    idm: str
    direction: RelayDirection
    seq: int = 0
    ack: int = 0
    payload: bytes = b""

    @property
    def is_control(self) -> bool:
        return self.kind in _CONTROL_KINDS

    def to_fields(self) -> dict[bytes, bytes]:
        fields: dict[bytes, bytes] = {
            b"k": self.kind.value.encode(),
            b"s": self.session_id.encode(),
            b"i": self.invocation_id.encode(),
            b"m": self.idm.encode(),
            b"d": self.direction.value.encode(),
            b"q": str(self.seq).encode(),
            b"a": str(self.ack).encode(),
        }
        if self.payload:
            fields[b"y"] = self.payload
        return fields

    @staticmethod
    def from_fields(fields: dict[bytes, bytes]) -> "RelayFrame":
        return RelayFrame(
            kind=RelayFrameKind(fields[b"k"].decode()),
            session_id=fields[b"s"].decode(),
            invocation_id=fields[b"i"].decode(),
            idm=fields[b"m"].decode(),
            direction=RelayDirection(fields[b"d"].decode()),
            seq=int(fields[b"q"]),
            ack=int(fields[b"a"]),
            payload=fields.get(b"y", b""),
        )


class BinaryRedis(Protocol):
    """The binary-safe async Redis surface the substrate uses (no decoded responses)."""

    async def xadd(self, name: str, fields: dict[bytes, bytes]) -> bytes: ...
    async def xread(
        self, streams: dict[str, str], count: int, block: int | None
    ) -> list[Any]: ...
    async def xtrim(self, name: str, minid: str, approximate: bool) -> int: ...
    async def hset(self, name: str, mapping: dict[str, str]) -> int: ...
    async def hgetall(self, name: str) -> dict[bytes, bytes]: ...
    async def pexpire(self, name: str, ms: int) -> int: ...
    async def set(self, name: str, value: str, nx: bool, px: int) -> bool | None: ...
    async def get(self, name: str) -> bytes | None: ...
    async def delete(self, name: str) -> int: ...
    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> Any: ...


@dataclass
class StreamEntry:
    """One read stream entry: its Redis id and decoded frame."""

    entry_id: str
    frame: RelayFrame


class RelayStreamStore:
    """Cursor reads and acked-bounded trims over the per-node up/down streams."""

    def __init__(self, redis: BinaryRedis) -> None:
        self._redis = redis

    async def publish_up(self, node_id: str, frame: RelayFrame) -> str:
        key = resident_relay_up_key(node_id)
        return (await self._redis.xadd(key, frame.to_fields())).decode()

    async def publish_down(self, node_id: str, frame: RelayFrame) -> str:
        key = resident_relay_down_key(node_id)
        return (await self._redis.xadd(key, frame.to_fields())).decode()

    async def read_up(
        self, node_id: str, after_id: str, count: int, block_ms: int | None
    ) -> tuple[list[StreamEntry], str | None]:
        key = resident_relay_up_key(node_id)
        return await self._read(key, after_id, count, block_ms)

    async def read_down(
        self, node_id: str, after_id: str, count: int, block_ms: int | None
    ) -> tuple[list[StreamEntry], str | None]:
        return await self._read(
            resident_relay_down_key(node_id), after_id, count, block_ms
        )

    async def _read(
        self, key: str, after_id: str, count: int, block_ms: int | None
    ) -> tuple[list[StreamEntry], str | None]:
        """Read a batch and return its decodable entries plus the last raw id seen.

        Each entry is decoded inside the read so an undecodable frame — an unknown kind
        or direction, a non-int field, a rolling-upgrade mix — is skipped rather than
        raising out and stalling the stream. The last raw id is returned even when its
        frame was skipped, so the caller advances the cursor past a poison frame.
        """
        result = await self._redis.xread({key: after_id}, count=count, block=block_ms)
        entries: list[StreamEntry] = []
        last_id: str | None = None
        for _stream, items in result or []:
            for entry_id, fields in items:
                eid = (
                    entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                )
                last_id = eid
                try:
                    entries.append(StreamEntry(eid, RelayFrame.from_fields(fields)))
                except (KeyError, ValueError):
                    _logger.warning("skipping undecodable relay frame %s", eid)
        return entries, last_id

    async def trim_up_to(
        self, node_id: str, direction: RelayDirection, min_id: str
    ) -> None:
        """Trim at or below ``min_id`` (the recorded cursor, the last-forwarded id) —
        never above it, so a frame past the cursor is never discarded."""
        key = (
            resident_relay_up_key(node_id)
            if direction is RelayDirection.ORIGIN_TO_TARGET
            else resident_relay_down_key(node_id)
        )
        await self._redis.xtrim(key, minid=min_id, approximate=False)


class RelaySessionStore:
    """The durable per-session routing record: origin/target nodes and sidecar route."""

    def __init__(self, redis: BinaryRedis) -> None:
        self._redis = redis

    async def load(self, session_id: str) -> dict[str, str]:
        raw = await self._redis.hgetall(resident_relay_session_key(session_id))
        return {k.decode(): v.decode() for k, v in raw.items()}

    async def update(self, session_id: str, **fields: str | int) -> None:
        mapping = {k: str(v) for k, v in fields.items()}
        key = resident_relay_session_key(session_id)
        await self._redis.hset(key, mapping=mapping)
        # Bound the record against an origin crash that can never reap it; the delivery
        # path deletes it well within the TTL on every terminal.
        await self._redis.pexpire(key, _SESSION_TTL_MS)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(resident_relay_session_key(session_id))


# Owner-fenced compare-and-act: refresh or drop the lease only while this owner still
# holds it, atomically, so a lapsed owner that wakes after a successor took over cannot
# extend or delete the successor's lease.
_REFRESH_IF_OWNER = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)
_RELEASE_IF_OWNER = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)


class RelayLease:
    """A leg's single-receiver ownership lease with owner-fenced handover.

    Acquisition is atomic (``SET NX PX``); a lapsed lease lets a successor acquire, and
    refresh and release compare-and-act under the owner atomically, so a stalled prior
    owner that wakes after handover cannot extend or delete the successor's lease. The
    lease is transport-recovery ownership only and never touches admission credit.
    """

    def __init__(self, redis: BinaryRedis, ttl_ms: int = 15000) -> None:
        self._redis = redis
        self._ttl = ttl_ms

    @staticmethod
    def _key(session_id: str, leg: str) -> str:
        return f"{resident_relay_session_key(session_id)}:lease:{leg}"

    async def acquire(self, session_id: str, leg: str, owner: str) -> bool:
        got = await self._redis.set(
            self._key(session_id, leg), owner, nx=True, px=self._ttl
        )
        return bool(got)

    async def owns(self, session_id: str, leg: str, owner: str) -> bool:
        held = await self._redis.get(self._key(session_id, leg))
        return held is not None and held.decode() == owner

    async def refresh(self, session_id: str, leg: str, owner: str) -> bool:
        got = await self._redis.eval(
            _REFRESH_IF_OWNER, 1, self._key(session_id, leg), owner, str(self._ttl)
        )
        return bool(got)

    async def release(self, session_id: str, leg: str, owner: str) -> None:
        await self._redis.eval(_RELEASE_IF_OWNER, 1, self._key(session_id, leg), owner)


@dataclass
class WindowState:
    # A direction's byte window: the granted in-flight budget, the sent-but-unacked
    # bytes, and the cumulative bytes the receiver has confirmed. A cumulative ack is
    # idempotent — the receiver reports its running drained total and the sender frees
    # the delta since the last ack.
    granted: int
    used: int = 0
    acked: int = 0

    def can_send(self, size: int) -> bool:
        # A frame fits within the remaining window, or — when nothing is yet in flight —
        # is admitted alone so a payload larger than the whole window still makes
        # progress instead of deadlocking. Resident completions are chunked under the
        # window, so alone-admit only ever carries a single oversized control frame.
        return self.used == 0 or self.used + size <= self.granted

    def on_ack(self, cumulative: int) -> None:
        if cumulative > self.acked:
            self.used = max(0, self.used - (cumulative - self.acked))
            self.acked = cumulative


class DirectionWindow:
    """Sender-side flow control for one direction's data frames.

    A send reserves its byte size against the receiver's granted window, blocking until
    earlier bytes are acknowledged when the window is full, so a slow receiver holds no
    more than its advertised window in flight. A grant advances the cumulative ack and
    wakes a blocked sender. It carries only relay-window byte credit and never a service
    claim's admission credit.
    """

    def __init__(self, granted: int) -> None:
        self._state = WindowState(granted=granted)
        self._cond = asyncio.Condition()

    async def reserve(self, size: int) -> None:
        async with self._cond:
            while not self._state.can_send(size):
                await self._cond.wait()
            self._state.used += size

    async def grant(self, cumulative: int) -> None:
        async with self._cond:
            self._state.on_ack(cumulative)
            self._cond.notify_all()

    @property
    def in_flight(self) -> int:
        return self._state.used


__all__ = [
    "BinaryRedis",
    "DirectionWindow",
    "RelayDirection",
    "RelayFrame",
    "RelayFrameKind",
    "RelayLease",
    "RelaySessionStore",
    "RelayStreamStore",
    "StreamEntry",
    "WindowState",
]
