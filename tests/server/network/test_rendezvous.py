"""The root rendezvous bridge routes opaque frames between attached nodes by session,
drains fairly (priority control first, then round-robin across sessions), and resumes
from a durable cursor.
"""

import asyncio

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import (
    RelayDirection,
    RelayFrameKind,
    RelaySessionStore,
    RelayStreamStore,
)

from ._relay_fakes import FakeBinaryRedis, relay_frame


async def _bridge(
    redis: FakeBinaryRedis,
) -> tuple[RootRendezvousBridge, RelayStreamStore, RelaySessionStore]:
    streams = RelayStreamStore(redis)
    sessions = RelaySessionStore(redis)
    bridge = RootRendezvousBridge(streams, sessions, RootCursorStore(redis))
    return bridge, streams, sessions


def test_forwards_each_direction_to_the_peer_node_opaquely() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        bridge, streams, sessions = await _bridge(redis)
        await sessions.update("rly-1", origin_node="nde-o", target_node="nde-t")

        # origin->target rides the origin's up stream and lands on the target's down.
        await streams.publish_up(
            "nde-o",
            relay_frame(
                RelayFrameKind.DATA,
                direction=RelayDirection.ORIGIN_TO_TARGET,
                seq=1,
                payload=b"\x00req\xff",
            ),
        )
        assert await bridge.pump_node("nde-o") == 1
        down_t, _ = await streams.read_down("nde-t", "0", count=10, block_ms=None)
        assert [e.frame.payload for e in down_t] == [b"\x00req\xff"]  # opaque, intact

        # target->origin rides the target's up stream and lands on the origin's down.
        await streams.publish_up(
            "nde-t",
            relay_frame(
                RelayFrameKind.DATA,
                direction=RelayDirection.TARGET_TO_ORIGIN,
                seq=1,
                payload=b"resp",
            ),
        )
        assert await bridge.pump_node("nde-t") == 1
        down_o, _ = await streams.read_down("nde-o", "0", count=10, block_ms=None)
        assert [e.frame.payload for e in down_o] == [b"resp"]

    asyncio.run(run())


def test_priority_control_frames_drain_before_data() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        bridge, streams, sessions = await _bridge(redis)
        await sessions.update("rly-1", origin_node="nde-o", target_node="nde-t")
        o2t = RelayDirection.ORIGIN_TO_TARGET
        # Two data frames sit ahead of a cancel; the cancel forwards first.
        await streams.publish_up(
            "nde-o", relay_frame(RelayFrameKind.DATA, direction=o2t, seq=1)
        )
        await streams.publish_up(
            "nde-o", relay_frame(RelayFrameKind.DATA, direction=o2t, seq=2)
        )
        await streams.publish_up(
            "nde-o", relay_frame(RelayFrameKind.CANCEL, direction=o2t)
        )
        await bridge.pump_node("nde-o")
        down, _ = await streams.read_down("nde-t", "0", 10, None)
        kinds = [e.frame.kind for e in down]
        assert kinds[0] is RelayFrameKind.CANCEL  # priority lane, ahead of the backlog

    asyncio.run(run())


def test_round_robin_interleaves_busy_and_quiet_sessions() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        bridge, streams, sessions = await _bridge(redis)
        await sessions.update("rly-A", origin_node="nde-o", target_node="nde-t")
        await sessions.update("rly-B", origin_node="nde-o", target_node="nde-t")
        o2t = RelayDirection.ORIGIN_TO_TARGET
        # A dumps three frames, then B sends one; fair drain must not bury B.
        for seq in (1, 2, 3):
            await streams.publish_up(
                "nde-o",
                relay_frame(
                    RelayFrameKind.DATA, session_id="rly-A", direction=o2t, seq=seq
                ),
            )
        await streams.publish_up(
            "nde-o",
            relay_frame(RelayFrameKind.DATA, session_id="rly-B", direction=o2t, seq=1),
        )
        await bridge.pump_node("nde-o")
        down, _ = await streams.read_down("nde-t", "0", 10, None)
        order = [e.frame.session_id for e in down]
        # First round takes one from each session before A's backlog drains.
        assert order[:2] == ["rly-A", "rly-B"]

    asyncio.run(run())


def test_bridge_reads_non_blocking_so_an_idle_node_does_not_wedge() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        bridge, streams, sessions = await _bridge(redis)
        await sessions.update("rly-1", origin_node="nde-b", target_node="nde-t")
        o2t = RelayDirection.ORIGIN_TO_TARGET
        # nde-a is idle; nde-b has a frame. A blocking read on the idle node would wedge
        # this serial multi-node driver, so nde-b would never be bridged.
        await streams.publish_up(
            "nde-b", relay_frame(RelayFrameKind.DATA, direction=o2t, seq=1)
        )
        assert await bridge.pump_node("nde-a") == 0  # idle: returns at once
        assert await bridge.pump_node("nde-b") == 1  # still served
        assert redis.blocks == [None, None]  # every bridge read is non-blocking

    asyncio.run(run())


def test_durable_cursor_resumes_and_does_not_reforward() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        bridge, streams, sessions = await _bridge(redis)
        await sessions.update("rly-1", origin_node="nde-o", target_node="nde-t")
        o2t = RelayDirection.ORIGIN_TO_TARGET
        await streams.publish_up(
            "nde-o", relay_frame(RelayFrameKind.DATA, direction=o2t, seq=1)
        )
        assert await bridge.pump_node("nde-o") == 1
        # A second pump with nothing new forwards nothing (cursor advanced durably).
        assert await bridge.pump_node("nde-o") == 0
        down, _ = await streams.read_down("nde-t", "0", 10, None)
        assert len(down) == 1

    asyncio.run(run())
