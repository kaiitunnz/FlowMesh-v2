"""Shared helpers for exercising a real ``ControlPlaneTracer`` in tests."""

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.control import ControlPlaneTracer


def recording_control_tracer(
    level: TelemetryLevel = TelemetryLevel.FULL,
) -> tuple[ControlPlaneTracer, InMemorySpanExporter]:
    """A real, enabled ``ControlPlaneTracer`` whose spans land in an in-memory list.

    Uses ``SimpleSpanProcessor`` (synchronous export) so a span is visible in the
    exporter as soon as its ``with`` block exits — no flush needed.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    config = TelemetryConfig(
        level=level,
        traces_enabled=True,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )
    return ControlPlaneTracer(tracer, config), exporter


def recording_tracer(
    level: TelemetryLevel = TelemetryLevel.FULL,
) -> tuple[Tracer, InMemorySpanExporter, TelemetryConfig]:
    """A real, enabled tracer whose spans land in an in-memory list, plus its config.

    Uses ``SimpleSpanProcessor`` (synchronous export) so a span is visible in the
    exporter as soon as it ends — no flush needed.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    config = TelemetryConfig(
        level=level,
        traces_enabled=True,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )
    return tracer, exporter, config


def spans_by_stage(exporter: InMemorySpanExporter, stage: str) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.name == f"flowmesh.control.{stage}"
    ]
