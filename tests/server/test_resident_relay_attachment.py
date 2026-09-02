"""The per-node resident relay attachment: it dispatches down frames to local delivery,
resumes from a durable cursor, publishes responses to its up stream, and lets only the
lease owner consume so a restart hands the leg over rather than double-consuming.
"""

import asyncio

from server.network.reverse_relay import (
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelayStreamStore,
)
from server.supervisor.services.resident_relay_attachment import (
    LocalDelivery,
    ResidentRelayAttachment,
)
from tests.server.network._relay_fakes import FakeBinaryRedis, relay_frame


class _RecordingDelivery(LocalDelivery):
    def __init__(self) -> None:
        self.frames: list[str] = []

    async def on_frame(self, frame: RelayFrame) -> None:
        self.frames.append(f"{frame.session_id}:{frame.seq}")


def _attachment(
    redis: FakeBinaryRedis, owner: str
) -> tuple[ResidentRelayAttachment, _RecordingDelivery]:
    delivery = _RecordingDelivery()
    return (
        ResidentRelayAttachment(redis, "nde-t", delivery, owner=owner),
        delivery,
    )


def test_dispatches_down_frames_and_resumes_from_the_cursor() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        streams = RelayStreamStore(redis)
        attachment, delivery = _attachment(redis, "owner-1")
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=1))
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=2))
        assert await attachment.pump_once() == 2
        assert delivery.frames == ["rly-1:1", "rly-1:2"]
        # The durable cursor advanced: a second pump dispatches nothing new.
        assert await attachment.pump_once() == 0
        assert delivery.frames == ["rly-1:1", "rly-1:2"]

    asyncio.run(run())


def test_send_up_publishes_a_response_to_the_up_stream() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        streams = RelayStreamStore(redis)
        attachment, _ = _attachment(redis, "owner-1")
        await attachment.send_up(
            relay_frame(RelayFrameKind.DATA, direction=RelayDirection.TARGET_TO_ORIGIN)
        )
        up = await streams.read_up("nde-t", "0", count=10, block_ms=0)
        assert [e.frame.kind for e in up] == [RelayFrameKind.DATA]

    asyncio.run(run())


def test_only_the_lease_owner_consumes_the_down_stream() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        streams = RelayStreamStore(redis)
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=1))
        first, first_delivery = _attachment(redis, "owner-1")
        second, second_delivery = _attachment(redis, "owner-2")
        assert await first.pump_once() == 1  # first acquires the lease and consumes
        # A competitor is fenced (-1) while the owner holds the lease.
        assert await second.pump_once() == -1
        assert second_delivery.frames == []
        # After the owner releases, the competitor reclaims and can consume.
        await first.stop()
        await streams.publish_down("nde-t", relay_frame(RelayFrameKind.DATA, seq=2))
        assert await second.pump_once() == 1
        assert second_delivery.frames == ["rly-1:2"]

    asyncio.run(run())
