"""One resident invocation's trace context reaches the replica over the relay.

The two ends run in different processes, so the replica's ``flowmesh.engine_request``
lands under the origin's transport span only if every relay frame carries the context it
was sent under and each span opens from the carried value rather than an ambient one.
Both halves break silently: the spans are still emitted and the protocol still
completes, so the invocation simply splits into unrelated traces that no reader can
join back up.
"""

import asyncio
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan

from shared.network.relay_frame import RelayFrame
from shared.schemas.network import Transport
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.semconv import SPAN_ENGINE_REQUEST, transport_span_name
from tests.worker.otel_support import recorded_worker_spans, worker_telemetry

from .test_origin_replica_loop import _Harness

_TRACE_ID = "0102030405060708090a0b0c0d0e0f10"
_PARENT_SPAN_ID = "00f1e2d3c4b5a697"
_TRACEPARENT = f"00-{_TRACE_ID}-{_PARENT_SPAN_ID}-01"


def _invoke(
    level: TelemetryLevel,
) -> tuple[Sequence[ReadableSpan], list[RelayFrame]]:
    frames: list[RelayFrame] = []
    with recorded_worker_spans(worker_telemetry(level)) as exporter:

        async def scenario() -> None:
            harness = _Harness(observe=frames.append)
            harness.begin(traceparent=_TRACEPARENT)
            await asyncio.wait_for(harness.done.wait(), timeout=10.0)
            await harness.sidecar.aclose()

        asyncio.run(scenario())
        return exporter.get_finished_spans(), frames


def test_the_engine_request_lands_under_the_origins_transport_span() -> None:
    spans, _ = _invoke(TelemetryLevel.FINE)
    by_name = {span.name: span for span in spans}
    transport = by_name[transport_span_name(Transport.CONTROL_RELAY)]
    engine = by_name[SPAN_ENGINE_REQUEST]

    assert transport.get_span_context().trace_id == int(_TRACE_ID, 16)
    assert transport.parent is not None
    assert transport.parent.span_id == int(_PARENT_SPAN_ID, 16)

    assert engine.get_span_context().trace_id == int(_TRACE_ID, 16)
    assert engine.parent is not None
    assert engine.parent.span_id == transport.get_span_context().span_id


def test_no_relay_frame_carries_trace_context_when_telemetry_is_off() -> None:
    """Off costs zero bytes on the wire: the field is absent, not an explicit null."""
    spans, frames = _invoke(TelemetryLevel.OFF)
    assert not spans
    assert frames
    assert all(frame.tp is None for frame in frames)
