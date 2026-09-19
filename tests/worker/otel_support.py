"""The worker's own tracer provider, built from one test's telemetry config.

``_ensure_tracer_provider`` builds once per process and hands the result to OTel's
set-once global, so a test that needs the provider *its* config produces has to reset
both and put them back afterwards.
"""

import contextlib
from collections.abc import Iterator
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from worker.telemetry import otel


def worker_telemetry(
    level: TelemetryLevel, *, otlp_endpoint: str | None = None
) -> TelemetryConfig:
    """A traces-on worker config at ``level``, sampling everything."""
    return TelemetryConfig(
        level=level,
        traces_enabled=True,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=otlp_endpoint,
    )


@contextlib.contextmanager
def fresh_worker_provider(config: TelemetryConfig) -> Iterator[TracerProvider]:
    """The provider the worker wires under ``config``, discarded on exit.

    Every test that builds or rebuilds the worker provider goes through here. Whichever
    one builds it first otherwise pins the level and the exporter set for the rest of
    the session, and resetting the module flag alone leaves that provider behind as the
    next test's, so a result would depend on what ran before it.
    """
    with (
        patch.object(otel, "_telemetry_config", config),
        patch.object(otel, "_PROVIDER_INITIALIZED", False),
        patch.object(trace, "_TRACER_PROVIDER", None),
        patch.object(trace, "_TRACER_PROVIDER_SET_ONCE", Once()),
    ):
        otel.get_tracer()
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        yield provider


@contextlib.contextmanager
def recorded_worker_spans(config: TelemetryConfig) -> Iterator[InMemorySpanExporter]:
    """Every span the worker's tracer completes under ``config``."""
    exporter = InMemorySpanExporter()
    with fresh_worker_provider(config) as provider:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        yield exporter
