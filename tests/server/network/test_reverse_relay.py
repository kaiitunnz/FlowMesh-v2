"""The reverse-relay substrate: frame codec, cursor reads, acked-bounded trim, and the
owner-fenced cursor lease that hands a leg's single receiver over on restart.

A deterministic in-memory binary-Redis stub models exactly the surface the substrate
uses (stream append / cursor read / MINID trim, hash CRUD, and SET NX PX with a
test-driven clock) so the recovery path is proven without a live Redis or a fake.
"""

import asyncio
from typing import Any

from server.network.reverse_relay import (
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelayLease,
    RelaySessionStore,
    RelayStreamStore,
)


class _FakeBinaryRedis:
    """Models the binary-safe stream/hash/string surface with a controllable clock."""

    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self._seq = 0
        self._hashes: dict[str, dict[bytes, bytes]] = {}
        self._strings: dict[str, tuple[str, float]] = {}  # value, expires_at
        self.now = 0.0

    async def xadd(self, name: str, fields: dict[bytes, bytes]) -> bytes:
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self._streams.setdefault(name, []).append((entry_id, dict(fields)))
        return entry_id.encode()

    async def xread(self, streams: dict[str, str], count: int, block: int) -> list[Any]:
        out: list[Any] = []
        for key, after in streams.items():
            after_n = _idnum(after)
            items = [
                (eid.encode(), fields)
                for eid, fields in self._streams.get(key, [])
                if _idnum(eid) > after_n
            ][:count]
            if items:
                out.append((key.encode(), items))
        return out

    async def xtrim(self, name: str, minid: str, approximate: bool) -> int:
        keep_from = _idnum(minid)
        before = self._streams.get(name, [])
        self._streams[name] = [e for e in before if _idnum(e[0]) >= keep_from]
        return len(before) - len(self._streams[name])

    async def hset(self, name: str, mapping: dict[str, str]) -> int:
        h = self._hashes.setdefault(name, {})
        for k, v in mapping.items():
            h[k.encode()] = v.encode()
        return len(mapping)

    async def hgetall(self, name: str) -> dict[bytes, bytes]:
        return dict(self._hashes.get(name, {}))

    async def set(self, name: str, value: str, nx: bool, px: int) -> bool | None:
        live = self._live(name)
        if nx and live is not None:
            return None
        self._strings[name] = (value, self.now + px / 1000.0)
        return True

    async def get(self, name: str) -> bytes | None:
        live = self._live(name)
        return live.encode() if live is not None else None

    async def delete(self, name: str) -> int:
        return 1 if self._strings.pop(name, None) is not None else 0

    def _live(self, name: str) -> str | None:
        entry = self._strings.get(name)
        if entry is None:
            return None
        value, expires_at = entry
        if self.now >= expires_at:
            self._strings.pop(name, None)
            return None
        return value


def _idnum(entry_id: str) -> int:
    return int(entry_id.split("-")[0]) if entry_id not in ("0", "") else 0


def _frame(kind: RelayFrameKind, seq: int = 0, payload: bytes = b"") -> RelayFrame:
    return RelayFrame(
        kind=kind,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.TARGET_TO_ORIGIN,
        seq=seq,
        payload=payload,
    )


def test_frame_codec_round_trips_raw_payload_and_opaque_fence() -> None:
    frame = RelayFrame(
        kind=RelayFrameKind.OPEN,
        session_id="rly-9",
        invocation_id="inv-9",
        idm="idm-9",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=3,
        ack=2,
        window=4096,
        payload=b"\x00\x01\x02not-utf8\xff",
        fence=b'{"handoff":"opaque"}',
    )
    restored = RelayFrame.from_fields(frame.to_fields())
    assert restored == frame
    assert restored.payload == b"\x00\x01\x02not-utf8\xff"  # binary-safe, not base64
    assert restored.fence == b'{"handoff":"opaque"}'  # opaque to the substrate


def test_cursor_read_returns_only_entries_after_the_stored_id() -> None:
    async def run() -> None:
        redis = _FakeBinaryRedis()
        streams = RelayStreamStore(redis)
        first = await streams.publish_down("nde-t", _frame(RelayFrameKind.DATA, seq=1))
        await streams.publish_down("nde-t", _frame(RelayFrameKind.DATA, seq=2))
        # From cursor "0" both entries read; from `first` only the second.
        all_entries = await streams.read_down("nde-t", "0", count=10, block_ms=0)
        assert [e.frame.seq for e in all_entries] == [1, 2]
        after_first = await streams.read_down("nde-t", first, count=10, block_ms=0)
        assert [e.frame.seq for e in after_first] == [2]

    asyncio.run(run())


def test_trim_never_discards_an_unacknowledged_frame() -> None:
    async def run() -> None:
        redis = _FakeBinaryRedis()
        streams = RelayStreamStore(redis)
        acked = await streams.publish_down("nde-t", _frame(RelayFrameKind.DATA, seq=1))
        await streams.publish_down("nde-t", _frame(RelayFrameKind.DATA, seq=2))
        # Trim below the unacked id: MINID keeps seq=2 (the substrate never trims above
        # the cumulative-acked id, so an unacknowledged frame always survives).
        await streams.trim_up_to("nde-t", RelayDirection.TARGET_TO_ORIGIN, acked)
        surviving = await streams.read_down("nde-t", "0", count=10, block_ms=0)
        assert [e.frame.seq for e in surviving] == [1, 2]
        # A trim strictly above the acked id would drop seq=2; the substrate never does
        # that — trim_up_to is called only with the cumulative-acked id.

    asyncio.run(run())


def test_lease_hands_over_atomically_and_fences_the_lapsed_owner() -> None:
    async def run() -> None:
        redis = _FakeBinaryRedis()
        lease = RelayLease(redis, ttl_ms=1000)
        # A wins the leg; a competitor B cannot acquire while A's lease is live.
        assert await lease.acquire("rly-1", "t2o", "consumer-A")
        assert not await lease.acquire("rly-1", "t2o", "consumer-B")
        assert await lease.owns("rly-1", "t2o", "consumer-A")
        # A's lease lapses; B reclaims it and A is fenced out of the leg.
        redis.now += 2.0
        assert await lease.acquire("rly-1", "t2o", "consumer-B")
        assert await lease.owns("rly-1", "t2o", "consumer-B")
        assert not await lease.owns("rly-1", "t2o", "consumer-A")
        assert not await lease.refresh("rly-1", "t2o", "consumer-A")
        # A releasing does not steal the leg back from B.
        await lease.release("rly-1", "t2o", "consumer-A")
        assert await lease.owns("rly-1", "t2o", "consumer-B")

    asyncio.run(run())


def test_session_record_round_trips_the_durable_cursor() -> None:
    async def run() -> None:
        redis = _FakeBinaryRedis()
        sessions = RelaySessionStore(redis)
        await sessions.update("rly-1", t2o_last_acked=7, t2o_cursor="12-0")
        record = await sessions.load("rly-1")
        assert record["t2o_last_acked"] == "7"
        assert record["t2o_cursor"] == "12-0"

    asyncio.run(run())


def test_control_frames_ride_the_priority_lane() -> None:
    for kind in (
        RelayFrameKind.OPEN,
        RelayFrameKind.ACK,
        RelayFrameKind.WINDOW,
        RelayFrameKind.CANCEL,
        RelayFrameKind.TERMINAL,
        RelayFrameKind.ERROR,
    ):
        assert _frame(kind).is_control
    assert not _frame(RelayFrameKind.DATA).is_control
