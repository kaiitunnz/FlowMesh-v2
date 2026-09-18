"""The worker exports a span it did not open with the exception text stripped.

``payload_free_span`` covers the sites the worker owns, but its tracer also serves
spans opened plainly -- by an executor, a library, or a site added later -- and OTel's
defaults copy an exception's message into those. ``PayloadFreeSpanExporter`` removes it
as the span leaves the process, which holds only while the worker's own provider is
wired through it: dropped, every span still exports and nothing in-process differs.
"""

from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from shared.telemetry.config import TelemetryLevel
from tests.worker.otel_support import fresh_worker_provider, worker_telemetry
from worker.telemetry import otel

_NONCE = "SECRET-NONCE-42"


class _Boom(ValueError):
    pass


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


def _exported_text(span: ReadableSpan) -> str:
    """The free-text a reader could pull off an exported span, as one string."""
    parts = [span.status.description or ""]
    for event in span.events:
        parts.append(event.name)
        parts += [f"{key}={value}" for key, value in (event.attributes or {}).items()]
    return " ".join(parts)


def test_the_worker_provider_is_wired_through_the_payload_free_exporter() -> None:
    captured = _Capturing()
    config = worker_telemetry(
        TelemetryLevel.FINE, otlp_endpoint="http://collector.invalid:4317"
    )
    with patch("worker.telemetry.otel.OTLPSpanExporter", lambda **_kwargs: captured):
        with fresh_worker_provider(config) as provider:
            tracer = otel.get_tracer()
            with pytest.raises(_Boom):
                with tracer.start_as_current_span("unowned"):
                    raise _Boom(f"model said: {_NONCE}")
            provider.shutdown()

    (span,) = captured.spans
    assert _NONCE not in _exported_text(span)
    assert span.events == ()
    assert not span.status.description
