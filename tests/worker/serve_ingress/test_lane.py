"""The forward-ingress lane routes engine frames to the client's channel.

The lane delivers the engine's head and body frames to the request's channel; the
response is marked committed only once the listener writes the head to the client socket
(the channel's commit hook), never here on mere receipt, so a loss before the write may
still recover while a loss after it closes the client.
"""

from shared.resident.carriage import ControlRelayCarriage
from shared.resident.reports import ResidentStreamChunk, ResidentStreamHead
from shared.resident.transport import ResidentFrameSink
from worker.serve_ingress.channel import ServeIngressChannel
from worker.serve_ingress.lane import ServeIngressLane


class _NullSink(ResidentFrameSink):
    async def send(self, frame) -> None:  # pragma: no cover - never driven here
        pass


def _lane() -> ServeIngressLane:
    return ServeIngressLane(
        carriage=ControlRelayCarriage(_NullSink()),
        report_ack=lambda _ack: None,
        report_outcome=lambda _outcome: None,
    )


def _head(invocation_id: str = "inv-1") -> ResidentStreamHead:
    return ResidentStreamHead(
        invocation_id=invocation_id,
        session_id="rly-1",
        status=200,
        headers=(("content-type", "application/json"),),
    )


def test_the_head_and_chunk_route_to_the_registered_channel() -> None:
    lane = _lane()
    channel = ServeIngressChannel()
    lane._channels["inv-1"] = channel
    lane.on_stream_head(_head())
    lane.on_stream_chunk(
        ResidentStreamChunk(invocation_id="inv-1", session_id="rly-1", payload=b"x")
    )
    head = channel.drain(0.05)
    chunk = channel.drain(0.05)
    assert head is not None and head.kind == "head" and head.status == 200
    assert chunk is not None and chunk.kind == "chunk" and chunk.payload == b"x"


def test_the_lane_does_not_commit_on_head_receipt() -> None:
    # The head reaching the lane does not commit the response; only the listener writing
    # it to the client socket does, via the channel commit hook.
    lane = _lane()
    channel = ServeIngressChannel()
    committed: list = []
    channel.on_committed(lambda status, headers: committed.append((status, headers)))
    lane._channels["inv-1"] = channel
    lane.on_stream_head(_head())
    assert committed == []
    channel.commit_head(200, (("content-type", "application/json"),))
    assert committed == [(200, (("content-type", "application/json"),))]
    # Committed fires exactly once even if the listener re-signals.
    channel.commit_head(200, ())
    assert len(committed) == 1
