"""The forward-ingress lane reports its response committed once, on the head.

The ingress tees the engine's frames to its own client, so control never sees them.
When the head reaches the lane the response is committed to that client; the lane sends
that up exactly once, ahead of the outcome on the same ordered stream, so a post-commit
loss fails the response rather than re-driving the engine over bytes the client holds.
"""

from shared.resident.carriage import ControlRelayCarriage
from shared.resident.reports import ResidentStreamHead
from shared.resident.transport import ResidentFrameSink
from worker.serve_ingress.lane import ServeIngressLane


class _NullSink(ResidentFrameSink):
    async def send(self, frame) -> None:  # pragma: no cover - never driven here
        pass


def _lane(committed: list) -> ServeIngressLane:
    return ServeIngressLane(
        carriage=ControlRelayCarriage(_NullSink()),
        report_ack=lambda _ack: None,
        report_outcome=lambda _outcome: None,
        report_committed=lambda inv, status, headers: committed.append(
            (inv, status, headers)
        ),
    )


def _head(invocation_id: str = "inv-1") -> ResidentStreamHead:
    return ResidentStreamHead(
        invocation_id=invocation_id,
        session_id="rly-1",
        status=200,
        headers=(("content-type", "application/json"),),
    )


def test_the_head_reports_the_response_committed() -> None:
    committed: list = []
    _lane(committed).on_stream_head(_head())
    assert committed == [("inv-1", 200, (("content-type", "application/json"),))]


def test_a_duplicate_head_reports_committed_only_once() -> None:
    committed: list = []
    lane = _lane(committed)
    lane.on_stream_head(_head())
    lane.on_stream_head(_head())
    assert len(committed) == 1
