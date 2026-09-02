"""Reverse-rendezvous relay substrate over per-node outbound Redis streams.

The universal ``control_relay`` transport carries a resident invocation without either
end accepting an inbound connection: an origin and a target each attach outward to the
root rendezvous, which bridges opaque framed data between their per-node streams. This
module is the transport primitives — frame codec, per-node stream cursor reads, the
durable per-session record, and the ownership lease that hands a leg's single receiver
over on restart. It is deliberately free of any resident, claim, or admission concept:
frames carry an opaque fence and an opaque payload, and are keyed only by the relay
session, invocation, and idempotency identifiers used for routing and dedupe.

Durability follows a cursor lease, not a consumer group: each leg has one logical
receiver that reads its node stream from a durable stored cursor, and a restart reclaims
an owner-fenced lease and resumes from that cursor. Unacknowledged frames are never
trimmed — a stream is trimmed only at or below the cumulative-acknowledged id.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ..clients.redis import (
    resident_relay_down_key,
    resident_relay_session_key,
    resident_relay_up_key,
)


class RelayFrameKind(StrEnum):
    """The frame kinds. Every kind but ``DATA`` rides the priority control lane: it
    bypasses the byte window so a full data window cannot deadlock cancellation."""

    OPEN = "open"
    DATA = "data"
    ACK = "ack"
    WINDOW = "window"
    CANCEL = "cancel"
    TERMINAL = "terminal"
    ERROR = "error"


class RelayDirection(StrEnum):
    """A frame's flow direction within a session."""

    ORIGIN_TO_TARGET = "o2t"
    TARGET_TO_ORIGIN = "t2o"


_CONTROL_KINDS = frozenset(
    {
        RelayFrameKind.OPEN,
        RelayFrameKind.ACK,
        RelayFrameKind.WINDOW,
        RelayFrameKind.CANCEL,
        RelayFrameKind.TERMINAL,
        RelayFrameKind.ERROR,
    }
)


@dataclass(frozen=True)
class RelayFrame:
    """One relay frame. ``fence`` and ``payload`` are opaque bytes the root never reads;
    the remaining fields are the routing and flow-control metadata it may read."""

    kind: RelayFrameKind
    session_id: str
    invocation_id: str
    idm: str
    direction: RelayDirection
    seq: int = 0
    ack: int = 0
    window: int = 0
    payload: bytes = b""
    fence: bytes = b""

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
            b"w": str(self.window).encode(),
        }
        if self.payload:
            fields[b"y"] = self.payload
        if self.fence:
            fields[b"f"] = self.fence
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
            window=int(fields[b"w"]),
            payload=fields.get(b"y", b""),
            fence=fields.get(b"f", b""),
        )


class BinaryRedis(Protocol):
    """The binary-safe async Redis surface the substrate uses (no decoded responses)."""

    async def xadd(self, name: str, fields: dict[bytes, bytes]) -> bytes: ...
    async def xread(
        self, streams: dict[str, str], count: int, block: int
    ) -> list[Any]: ...
    async def xtrim(self, name: str, minid: str, approximate: bool) -> int: ...
    async def hset(self, name: str, mapping: dict[str, str]) -> int: ...
    async def hgetall(self, name: str) -> dict[bytes, bytes]: ...
    async def set(self, name: str, value: str, nx: bool, px: int) -> bool | None: ...
    async def get(self, name: str) -> bytes | None: ...
    async def delete(self, name: str) -> int: ...


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
        self, node_id: str, after_id: str, count: int, block_ms: int
    ) -> list[StreamEntry]:
        key = resident_relay_up_key(node_id)
        return await self._read(key, after_id, count, block_ms)

    async def read_down(
        self, node_id: str, after_id: str, count: int, block_ms: int
    ) -> list[StreamEntry]:
        return await self._read(
            resident_relay_down_key(node_id), after_id, count, block_ms
        )

    async def _read(
        self, key: str, after_id: str, count: int, block_ms: int
    ) -> list[StreamEntry]:
        result = await self._redis.xread({key: after_id}, count=count, block=block_ms)
        entries: list[StreamEntry] = []
        for _stream, items in result or []:
            for entry_id, fields in items:
                eid = (
                    entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                )
                entries.append(StreamEntry(eid, RelayFrame.from_fields(fields)))
        return entries

    async def trim_up_to(
        self, node_id: str, direction: RelayDirection, min_id: str
    ) -> None:
        """Trim strictly at or below ``min_id`` (the cumulative-acked id) — never above,
        so an unacknowledged frame is never discarded."""
        key = (
            resident_relay_up_key(node_id)
            if direction is RelayDirection.ORIGIN_TO_TARGET
            else resident_relay_down_key(node_id)
        )
        await self._redis.xtrim(key, minid=min_id, approximate=False)


class RelaySessionStore:
    """The durable per-session record: the cursor/lease home and window state.

    It references the fence and identities opaquely and is never a source of truth for
    admission credit — control state remains that sole authority.
    """

    def __init__(self, redis: BinaryRedis) -> None:
        self._redis = redis

    async def load(self, session_id: str) -> dict[str, str]:
        raw = await self._redis.hgetall(resident_relay_session_key(session_id))
        return {k.decode(): v.decode() for k, v in raw.items()}

    async def update(self, session_id: str, **fields: str | int) -> None:
        mapping = {k: str(v) for k, v in fields.items()}
        await self._redis.hset(resident_relay_session_key(session_id), mapping=mapping)


class RelayLease:
    """A leg's single-receiver ownership lease with owner-fenced handover.

    Acquisition is atomic (``SET NX PX``); a lapsed lease lets a successor acquire, and
    the prior owner is fenced because every action first checks it still owns the lease.
    The lease is transport-recovery ownership only and never touches admission credit.
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
        if not await self.owns(session_id, leg, owner):
            return False
        await self._redis.set(self._key(session_id, leg), owner, nx=False, px=self._ttl)
        return True

    async def release(self, session_id: str, leg: str, owner: str) -> None:
        if await self.owns(session_id, leg, owner):
            await self._redis.delete(self._key(session_id, leg))


# The reverse-attachment liveness clock, injected so tests are deterministic.
Clock = Callable[[], float]


def default_clock() -> float:
    return time.monotonic()


@dataclass
class WindowState:
    """A direction's byte window: the granted in-flight budget, the bytes sent but not
    yet acknowledged, and the cumulative bytes the receiver has confirmed consuming.

    A cumulative ack is idempotent: a receiver reports the running total it has drained,
    and the sender frees the delta since the last ack. Control frames bypass this budget
    entirely and are never counted.
    """

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

    @property
    def granted(self) -> int:
        return self._state.granted


def make_stores(
    redis: BinaryRedis, *, lease_ttl_ms: int = 15000
) -> tuple[RelayStreamStore, RelaySessionStore, RelayLease]:
    """Bundle the three substrate stores over one binary-safe Redis connection."""
    return (
        RelayStreamStore(redis),
        RelaySessionStore(redis),
        RelayLease(redis, ttl_ms=lease_ttl_ms),
    )


__all__ = [
    "BinaryRedis",
    "Clock",
    "DirectionWindow",
    "RelayDirection",
    "RelayFrame",
    "RelayFrameKind",
    "RelayLease",
    "RelaySessionStore",
    "RelayStreamStore",
    "StreamEntry",
    "WindowState",
    "default_clock",
    "make_stores",
]
