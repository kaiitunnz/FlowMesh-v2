"""The rendezvous bridge carries a session's target leg over an offloaded sink."""

import asyncio

from server.network.rendezvous import (
    SOURCE_TO_ROOT_LEG,
    TARGET_LEG,
    RootCursorStore,
    RootRendezvousBridge,
)
from server.network.reverse_relay import RelaySessionStore, RelayStreamStore
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind

from ._relay_fakes import FakeBinaryRedis


def _frame(direction: RelayDirection, payload: bytes = b"opaque") -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=direction,
        seq=1,
        payload=payload,
    )


class _OffloadSink:
    def __init__(self) -> None:
        self.frames: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.frames.append(frame)


class _Failing:
    async def send(self, frame: RelayFrame) -> None:
        raise ConnectionResetError("leg gone")


async def _bridge(
    redis: FakeBinaryRedis, sink, *, target_leg: str
) -> tuple[RootRendezvousBridge, list[tuple[str, str, int]]]:
    await RelaySessionStore(redis).update(
        "rly-1",
        origin_node="nde-origin",
        target_node="nde-target",
        target_leg_transport=target_leg,
        target_leg_endpoint="10.0.0.4:41000",
    )
    legs: list[tuple[str, str, int]] = []
    bridge = RootRendezvousBridge(
        RelayStreamStore(redis),
        RelaySessionStore(redis),
        RootCursorStore(redis),
        offload_for=lambda session_id, record: sink,
        meter=lambda leg, transport, size: legs.append((leg, transport, size)),
    )
    return bridge, legs


def test_an_offloaded_session_leaves_the_target_down_stream_unused() -> None:
    async def scenario() -> tuple[_OffloadSink, dict, list[tuple[str, str, int]]]:
        redis = FakeBinaryRedis()
        sink = _OffloadSink()
        bridge, legs = await _bridge(redis, sink, target_leg="worker_direct")
        await RelayStreamStore(redis).publish_up(
            "nde-origin", _frame(RelayDirection.ORIGIN_TO_TARGET)
        )
        await bridge.pump_node("nde-origin")
        return sink, redis.streams, legs

    sink, streams, legs = asyncio.run(scenario())
    assert [f.payload for f in sink.frames] == [b"opaque"]
    assert "rr:node:nde-target:down" not in streams
    # The origin still carried its own leg to the root over the relay.
    assert (SOURCE_TO_ROOT_LEG, "control_relay", len(b"opaque")) in legs
    assert not [leg for leg in legs if leg[0] == TARGET_LEG]


def test_a_target_reply_still_reaches_the_origin_node_down_stream() -> None:
    async def scenario() -> dict:
        redis = FakeBinaryRedis()
        bridge, _legs = await _bridge(redis, _OffloadSink(), target_leg="worker_direct")
        await RelayStreamStore(redis).publish_up(
            "nde-target", _frame(RelayDirection.TARGET_TO_ORIGIN)
        )
        await bridge.pump_node("nde-target")
        return redis.streams

    assert "rr:node:nde-origin:down" in asyncio.run(scenario())


def test_a_session_without_an_offload_uses_the_target_down_stream() -> None:
    async def scenario() -> tuple[dict, list[tuple[str, str, int]]]:
        redis = FakeBinaryRedis()
        await RelaySessionStore(redis).update(
            "rly-1", origin_node="nde-origin", target_node="nde-target"
        )
        legs: list[tuple[str, str, int]] = []
        bridge = RootRendezvousBridge(
            RelayStreamStore(redis),
            RelaySessionStore(redis),
            RootCursorStore(redis),
            meter=lambda leg, transport, size: legs.append((leg, transport, size)),
        )
        await RelayStreamStore(redis).publish_up(
            "nde-origin", _frame(RelayDirection.ORIGIN_TO_TARGET)
        )
        await bridge.pump_node("nde-origin")
        return redis.streams, legs

    streams, legs = asyncio.run(scenario())
    assert "rr:node:nde-target:down" in streams
    assert {leg for leg, _, _ in legs} == {SOURCE_TO_ROOT_LEG, TARGET_LEG}


def test_a_lost_offloaded_leg_drops_the_sink_without_re_carrying_the_frame() -> None:
    async def scenario() -> dict:
        redis = FakeBinaryRedis()
        bridge, _legs = await _bridge(redis, _Failing(), target_leg="node_relay")
        await RelayStreamStore(redis).publish_up(
            "nde-origin", _frame(RelayDirection.ORIGIN_TO_TARGET)
        )
        await bridge.pump_node("nde-origin")
        return redis.streams

    # The frame is neither re-sent over the relay nor silently duplicated.
    assert "rr:node:nde-target:down" not in asyncio.run(scenario())
