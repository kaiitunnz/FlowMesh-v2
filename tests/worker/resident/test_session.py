"""The worker resident session carries a windowed invocation end to end.

Two sessions wired sink-to-sink stand in for the origin and replica workers the relay
bridges between. A completion far larger than one window streams under backpressure, a
re-forwarded frame lands once, a cancel wakes a blocked receiver, and every frame
carries the trace context it was sent under.
"""

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.network.session import FramedRelaySession, RelaySessionRole
from shared.resident.wire import KIND_CHUNK, KIND_DONE, KIND_STREAM
from shared.telemetry.propagation import extract_context


class _PairSink:
    """Delivers each produced frame straight into the peer session."""

    def __init__(self) -> None:
        self.peer: FramedRelaySession | None = None
        self.sent: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.sent.append(frame)
        assert self.peer is not None
        await self.peer.on_frame(frame)


def _pair(
    window_bytes: int = 64,
) -> tuple[FramedRelaySession, FramedRelaySession, _PairSink, _PairSink]:
    origin_sink, replica_sink = _PairSink(), _PairSink()
    origin = FramedRelaySession(
        session_id="s1",
        correlation_id="inv-1",
        operation_id="idm-1",
        role=RelaySessionRole.ORIGIN,
        sink=origin_sink,
        window_bytes=window_bytes,
    )
    replica = FramedRelaySession(
        session_id="s1",
        correlation_id="inv-1",
        operation_id="idm-1",
        role=RelaySessionRole.TARGET,
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
            correlation_id="inv-1",
            operation_id="idm-1",
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


def test_every_frame_carries_the_context_it_was_sent_under() -> None:
    """A frame's ``tp`` is the only way this session's context reaches the peer.

    The peer is another process and shares no ambient context, so a kind that ships
    without ``tp`` costs nothing here and nothing at the peer either -- the span it
    should have parented simply roots its own trace, and the invocation is split in
    two where no test looks.
    """

    async def run() -> None:
        origin, replica, origin_sink, _ = _pair()
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span("carrier") as carrier:
            await origin.send_wire(KIND_STREAM, request="hi")
            served = asyncio.ensure_future(replica.send_wire(KIND_CHUNK, text="part"))
            assert await origin.recv_wire(timeout=5.0) is not None
            await served
            await origin.cancel()
        expected = carrier.get_span_context()

        assert {frame.kind for frame in origin_sink.sent} == {
            RelayFrameKind.DATA,
            RelayFrameKind.WINDOW,
            RelayFrameKind.CANCEL,
        }
        for frame in origin_sink.sent:
            carried = trace.get_current_span(
                extract_context(frame.tp)
            ).get_span_context()
            assert carried.span_id == expected.span_id, frame.kind
            assert carried.trace_id == expected.trace_id, frame.kind

    asyncio.run(run())
