"""An exception's text never reaches a span, and never leaves the process.

An instrumentation site chooses its attributes one key at a time, so a payload can only
reach a span through a channel no site names: OTel's defaults copy an exception's
message into a span event and its status description. Nothing about the resulting leak
is visible at the raise site, and the exceptions that carry the most payload -- a model
response that failed to parse, a validation error echoing its input -- are raised by
code that has no idea a span is open around it.

All three halves are asserted here: the span itself is clean, an exporter that receives
a dirty span from anywhere else emits a clean one, and the tracer this process actually
builds is wired through that exporter.
"""

from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Tracer as SDKTracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.provider import (
    PayloadFreeSpanExporter,
    build_tracer,
    payload_free_span,
)
from shared.telemetry.semconv import PHYSICAL_ERROR_TYPE

_NONCE = "SECRET-NONCE-42"


class _Boom(ValueError):
    pass


def _tracer_and_spans() -> tuple[Any, InMemorySpanExporter]:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _rendered(span: ReadableSpan) -> str:
    """Everything a reader could pull off the span, as one string."""
    parts = [span.name, span.status.description or ""]
    parts += [f"{k}={v}" for k, v in (span.attributes or {}).items()]
    for event in span.events:
        parts.append(event.name)
        parts += [f"{k}={v}" for k, v in (event.attributes or {}).items()]
    return " ".join(parts)


def test_the_default_span_would_have_leaked_the_message() -> None:
    """The control: this is what the fixed sites did before, and why it matters."""
    tracer, exporter = _tracer_and_spans()
    with pytest.raises(_Boom):
        with tracer.start_as_current_span("leaky"):
            raise _Boom(f"model said: {_NONCE}")

    (span,) = exporter.get_finished_spans()
    assert _NONCE in _rendered(span)


def test_a_payload_free_span_records_the_type_and_not_the_message() -> None:
    tracer, exporter = _tracer_and_spans()
    with pytest.raises(_Boom):
        with payload_free_span(tracer, "guarded"):
            raise _Boom(f"model said: {_NONCE}")

    (span,) = exporter.get_finished_spans()
    assert _NONCE not in _rendered(span)
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert (span.attributes or {})[PHYSICAL_ERROR_TYPE] == "_Boom"


def test_the_exception_still_propagates_unchanged() -> None:
    """Telemetry never changes what the instrumented block does."""
    tracer, _ = _tracer_and_spans()
    original = _Boom(f"model said: {_NONCE}")
    with pytest.raises(_Boom) as caught:
        with payload_free_span(tracer, "guarded"):
            raise original
    assert caught.value is original
    assert _NONCE in str(caught.value)


class _Capturing(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_the_exporter_cleans_a_span_it_did_not_open() -> None:
    """The sites are one layer; a span recorded anywhere else still leaves clean.

    The worker's tracer also serves spans this change does not own, so the guarantee
    cannot rest on every current and future site remembering to ask for it.
    """
    tracer, exporter = _tracer_and_spans()
    with pytest.raises(_Boom):
        with tracer.start_as_current_span("leaky"):
            raise _Boom(f"model said: {_NONCE}")
    (dirty,) = exporter.get_finished_spans()
    assert _NONCE in _rendered(dirty)

    downstream = _Capturing()
    assert PayloadFreeSpanExporter(downstream).export([dirty]) is (
        SpanExportResult.SUCCESS
    )

    (clean,) = downstream.spans
    assert _NONCE not in _rendered(clean)
    assert clean.events == ()
    assert clean.status.status_code is StatusCode.ERROR
    assert clean.name == dirty.name
    assert clean.get_span_context().span_id == dirty.get_span_context().span_id
    assert clean.attributes == dirty.attributes


def test_the_process_tracer_is_wired_through_the_payload_free_exporter() -> None:
    """The unit above holds only while ``build_tracer`` keeps the wrapper in the chain.

    Dropping it changes nothing observable in this process -- every span still exports,
    still carries its attributes, and still reports its error -- so only what reaches
    the exporter can tell the two apart.
    """
    captured = _Capturing()
    config = TelemetryConfig(
        level=TelemetryLevel.FINE,
        traces_enabled=True,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint="http://collector.invalid:4317",
    )
    with patch(
        "shared.telemetry.provider.OTLPSpanExporter", lambda **_kwargs: captured
    ):
        tracer = build_tracer(config, {"service.name": "flowmesh-test"})

    assert isinstance(tracer, SDKTracer)
    try:
        with pytest.raises(_Boom):
            with tracer.start_as_current_span("unowned"):
                raise _Boom(f"model said: {_NONCE}")
    finally:
        tracer.span_processor.shutdown()

    (span,) = captured.spans
    assert _NONCE not in _rendered(span)
    assert span.events == ()
    assert not span.status.description
