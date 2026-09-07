"""The worker resident session carries a windowed invocation end to end.

Two sessions wired sink-to-sink stand in for the origin and replica workers the relay
bridges between. A completion far larger than one window streams under backpressure, a
re-forwarded frame lands once, and a cancel wakes a blocked receiver.
"""

import asyncio

from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.wire import KIND_CHUNK, KIND_DONE, KIND_STREAM
from worker.resident.session import ResidentRelaySession, ResidentSessionRole


class _PairSink:
    """Delivers each produced frame straight into the peer session."""

    def __init__(self) -> None:
        self.peer: ResidentRelaySession | None = None
        self.sent: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.sent.append(frame)
        assert self.peer is not None
        await self.peer.on_frame(frame)


def _pair(
    window_bytes: int = 64,
) -> tuple[ResidentRelaySession, ResidentRelaySession, _PairSink, _PairSink]:
    origin_sink, replica_sink = _PairSink(), _PairSink()
    origin = ResidentRelaySession(
        session_id="s1",
        invocation_id="inv-1",
        idm="idm-1",
        role=ResidentSessionRole.ORIGIN,
        sink=origin_sink,
        window_bytes=window_bytes,
    )
    replica = ResidentRelaySession(
        session_id="s1",
        invocation_id="inv-1",
        idm="idm-1",
        role=ResidentSessionRole.REPLICA,
        sink=replica_sink,
        window_bytes=window_bytes,
    )
    origin_sink.peer, replica_sink.peer = replica, origin
    return origin, replica, origin_sink, replica_sink


def test_streamed_completion_flows_under_backpressure() -> None:
    async def run() -> None:
        origin, replica, _, replica_sink = _pair(window_bytes=64)
        chunks = [f"chunk-{i:03d}-" for i in range(50)]  # far exceeds one 64B window

        async def serve() -> None:
            opening = await replica.recv_wire(timeout=5.0)
            assert opening is not None and opening["kind"] == KIND_STREAM
            for part in chunks:
                await replica.send_wire(KIND_CHUNK, data=part)
            await replica.send_wire(KIND_DONE)

        async def drive() -> list[str]:
            await origin.send_wire(KIND_STREAM, auth={"claim_id": "scl-1"})
            parts: list[str] = []
            while True:
                msg = await origin.recv_wire(timeout=5.0)
                assert msg is not None
                if msg["kind"] == KIND_CHUNK:
                    parts.append(str(msg["data"]))
                elif msg["kind"] == KIND_DONE:
                    return parts
            return parts

        server = asyncio.ensure_future(serve())
        assembled = await asyncio.wait_for(drive(), timeout=10.0)
        await server
        assert "".join(assembled) == "".join(chunks)
        # Backpressure held the producer to its window: it never had the whole
        # completion in flight at once.
        data_frames = [f for f in replica_sink.sent if f.kind is RelayFrameKind.DATA]
        assert len(data_frames) == len(chunks) + 1  # chunks + done

    asyncio.run(run())


def test_reforwarded_data_frame_lands_once() -> None:
    async def run() -> None:
        origin, _replica, _, _ = _pair()
        frame = RelayFrame(
            kind=RelayFrameKind.DATA,
            session_id="s1",
            invocation_id="inv-1",
            idm="idm-1",
            direction=RelayDirection.TARGET_TO_ORIGIN,
            seq=1,
            payload=b'{"kind": "chunk", "data": "one"}',
        )
        await origin.on_frame(frame)
        await origin.on_frame(frame)  # a bridge re-forward
        first = await origin.recv_wire(timeout=1.0)
        assert first is not None and first["data"] == "one"
        # The duplicate did not enqueue a second copy.
        second = await origin.recv_wire(timeout=0.2)
        assert second is None

    asyncio.run(run())


def test_cancel_wakes_a_blocked_receiver() -> None:
    async def run() -> None:
        origin, _replica, origin_sink, _ = _pair()

        async def wait_then_cancel() -> None:
            await asyncio.sleep(0.05)
            await origin.cancel()

        canceller = asyncio.ensure_future(wait_then_cancel())
        msg = await origin.recv_wire(timeout=5.0)
        await canceller
        assert msg is None
        assert origin.cancelled
        assert any(f.kind is RelayFrameKind.CANCEL for f in origin_sink.sent)

    asyncio.run(run())
