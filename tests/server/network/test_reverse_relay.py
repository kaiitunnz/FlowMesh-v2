"""The reverse-relay substrate: frame codec, cursor reads, acked-bounded trim, and the
owner-fenced cursor lease that hands a leg's single receiver over on restart.

The shared in-memory binary-Redis stub models exactly the surface the substrate uses, so
the recovery path is proven without a live Redis or a fake dependency.
"""

import asyncio

from server.network.reverse_relay import (
    DirectionWindow,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelayLease,
    RelaySessionStore,
    RelayStreamStore,
    WindowState,
)

from ._relay_fakes import FakeBinaryRedis, relay_frame


def test_frame_codec_round_trips_a_raw_binary_payload() -> None:
    frame = RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-9",
        invocation_id="inv-9",
        idm="idm-9",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=3,
        ack=2,
        payload=b"\x00\x01\x02not-utf8\xff",
    )
    restored = RelayFrame.from_fields(frame.to_fields())
    assert restored == frame
    assert restored.payload == b"\x00\x01\x02not-utf8\xff"  # binary-safe, not base64


def test_cursor_read_returns_only_entries_after_the_stored_id() -> None:
    async def run() -> None:
        streams = RelayStreamStore(FakeBinaryRedis())
        first = await streams.publish_down(
            "nde-t", relay_frame(RelayFrameKind.DATA, seq=1)
        )
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=2))
        # From cursor "0" both entries read; from `first` only the second.
        all_entries = await streams.read_down("nde-t", "0", count=10, block_ms=0)
        assert [e.frame.seq for e in all_entries] == [1, 2]
        after_first = await streams.read_down("nde-t", first, count=10, block_ms=0)
        assert [e.frame.seq for e in after_first] == [2]

    asyncio.run(run())


def test_trim_never_discards_an_unacknowledged_frame() -> None:
    async def run() -> None:
        streams = RelayStreamStore(FakeBinaryRedis())
        acked = await streams.publish_down(
            "nde-t", relay_frame(RelayFrameKind.DATA, seq=1)
        )
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=2))
        # Trim at the cumulative-acked id: MINID keeps seq=2 (the substrate never trims
        # above the acked id, so an unacknowledged frame always survives).
        await streams.trim_up_to("nde-t", RelayDirection.TARGET_TO_ORIGIN, acked)
        surviving = await streams.read_down("nde-t", "0", count=10, block_ms=0)
        assert [e.frame.seq for e in surviving] == [1, 2]

    asyncio.run(run())


def test_lease_hands_over_atomically_and_fences_the_lapsed_owner() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
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
        sessions = RelaySessionStore(FakeBinaryRedis())
        await sessions.update("rly-1", t2o_last_acked=7, t2o_cursor="12-0")
        record = await sessions.load("rly-1")
        assert record["t2o_last_acked"] == "7"
        assert record["t2o_cursor"] == "12-0"

    asyncio.run(run())


def test_control_frames_ride_the_priority_lane() -> None:
    for kind in (RelayFrameKind.WINDOW, RelayFrameKind.CANCEL):
        assert relay_frame(kind).is_control
    assert not relay_frame(RelayFrameKind.DATA).is_control


def test_window_grant_advances_only_relay_window_credit() -> None:
    # A cumulative ack frees the delta since the last ack and is idempotent. It touches
    # only the direction's own byte accounting; the substrate cannot name, and so never
    # releases, a service-claim admission credit.
    window = WindowState(granted=100, used=80)
    window.on_ack(30)
    assert (window.used, window.acked) == (50, 30)
    window.on_ack(30)  # a repeated cumulative ack releases nothing further
    assert (window.used, window.acked) == (50, 30)
    window.on_ack(80)
    assert (window.used, window.acked) == (0, 80)


def test_direction_window_blocks_the_sender_until_a_grant_arrives() -> None:
    async def run() -> None:
        window = DirectionWindow(granted=100)
        await window.reserve(60)
        assert window.in_flight == 60
        # A second 60-byte send exceeds the 100-byte window: the sender blocks, holding
        # no more than its granted window in flight, until the receiver grants.
        blocked = asyncio.ensure_future(window.reserve(60))
        await asyncio.sleep(0.01)
        assert not blocked.done()
        assert window.in_flight == 60
        await window.grant(60)
        await asyncio.wait_for(blocked, timeout=1.0)
        assert window.in_flight == 60

    asyncio.run(run())
