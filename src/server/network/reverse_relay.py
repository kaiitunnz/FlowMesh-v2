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

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from shared.network.relay_frame import (
    DirectionWindow,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    WindowState,
)

from ..clients.redis import (
    RESIDENT_RELAY_ROOT_CURSOR_KEY,
    resident_relay_down_cursor_key,
    resident_relay_down_key,
    resident_relay_session_key,
    resident_relay_up_key,
)

# A crashed origin can never trim its own session record; a generous TTL bounds the leak
# without expiring an active one — a single invocation's session lives far under it.
_SESSION_TTL_MS = 1_800_000

_logger = logging.getLogger("reverse-relay")


@dataclass(frozen=True)
class RelayKeyspace:
    """The redis key builders that place one reverse-relay namespace end to end.

    A keyspace isolates a namespace's per-node up/down streams, per-session routing
    record and its lease, the root bridge's per-node read cursor, and each node
    attachment's own down-stream cursor. The parameterization lets another namespace run
    a disjoint ``RelayKeyspace`` beside the resident ``rr:*`` streams without sharing a
    stream, record, lease, or cursor.
    """

    up: Callable[[str], str]
    down: Callable[[str], str]
    session: Callable[[str], str]
    root_cursor: str
    down_cursor: Callable[[str], str]


RESIDENT_RELAY_KEYSPACE = RelayKeyspace(
    up=resident_relay_up_key,
    down=resident_relay_down_key,
    session=resident_relay_session_key,
    root_cursor=RESIDENT_RELAY_ROOT_CURSOR_KEY,
    down_cursor=resident_relay_down_cursor_key,
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

    def __init__(
        self, redis: BinaryRedis, keyspace: RelayKeyspace = RESIDENT_RELAY_KEYSPACE
    ) -> None:
        self._redis = redis
        self._ks = keyspace

    async def publish_up(self, node_id: str, frame: RelayFrame) -> str:
        key = self._ks.up(node_id)
        return (await self._redis.xadd(key, frame.to_fields())).decode()

    async def publish_down(self, node_id: str, frame: RelayFrame) -> str:
        key = self._ks.down(node_id)
        return (await self._redis.xadd(key, frame.to_fields())).decode()

    async def read_up(
        self, node_id: str, after_id: str, count: int, block_ms: int | None
    ) -> tuple[list[StreamEntry], str | None]:
        return await self._read(self._ks.up(node_id), after_id, count, block_ms)

    async def read_down(
        self, node_id: str, after_id: str, count: int, block_ms: int | None
    ) -> tuple[list[StreamEntry], str | None]:
        return await self._read(self._ks.down(node_id), after_id, count, block_ms)

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
            self._ks.up(node_id)
            if direction is RelayDirection.ORIGIN_TO_TARGET
            else self._ks.down(node_id)
        )
        await self._redis.xtrim(key, minid=min_id, approximate=False)


class RelaySessionStore:
    """The durable per-session routing record: origin/target nodes and sidecar route."""

    def __init__(
        self, redis: BinaryRedis, keyspace: RelayKeyspace = RESIDENT_RELAY_KEYSPACE
    ) -> None:
        self._redis = redis
        self._ks = keyspace

    async def load(self, session_id: str) -> dict[str, str]:
        raw = await self._redis.hgetall(self._ks.session(session_id))
        return {k.decode(): v.decode() for k, v in raw.items()}

    async def update(self, session_id: str, **fields: str | int) -> None:
        mapping = {k: str(v) for k, v in fields.items()}
        key = self._ks.session(session_id)
        await self._redis.hset(key, mapping=mapping)
        # Bound the record against an origin crash that can never reap it; the delivery
        # path deletes it well within the TTL on every terminal.
        await self._redis.pexpire(key, _SESSION_TTL_MS)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._ks.session(session_id))


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

    def __init__(
        self,
        redis: BinaryRedis,
        ttl_ms: int = 15000,
        keyspace: RelayKeyspace = RESIDENT_RELAY_KEYSPACE,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_ms
        self._ks = keyspace

    def _key(self, session_id: str, leg: str) -> str:
        return f"{self._ks.session(session_id)}:lease:{leg}"

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


__all__ = [
    "RESIDENT_RELAY_KEYSPACE",
    "BinaryRedis",
    "DirectionWindow",
    "RelayDirection",
    "RelayFrame",
    "RelayFrameKind",
    "RelayKeyspace",
    "RelayLease",
    "RelaySessionStore",
    "RelayStreamStore",
    "StreamEntry",
    "WindowState",
]
