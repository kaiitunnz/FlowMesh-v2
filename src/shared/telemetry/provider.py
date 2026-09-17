"""Tracer/meter factory: builds the SDK provider when telemetry is on, and a null
twin that allocates nothing per call when it is off.

Each process calls ``build_tracer`` / ``build_meter`` once, from its own
``TelemetryConfig``, and threads the result explicitly rather than relying on OTel's
global provider.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext

from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Meter, NoOpMeter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import INVALID_SPAN, Span, Tracer

from .config import TelemetryConfig, TelemetryLevel

_TRACER_NAME = "flowmesh"
_METER_NAME = "flowmesh"
_DEFAULT_OTLP_TIMEOUT_SEC = 10.0


class _NullTracer(Tracer):
    """No-op tracer whose span context manager is a single reused instance.

    OTel's own ``NoOpTracer`` decorates ``start_as_current_span`` as a generator, which
    allocates a fresh context-manager object on every call even though its body does
    nothing. Returning the same ``nullcontext`` instance every time is what makes a
    disabled call site cost one attribute read and no allocation.
    """

    _span_context: AbstractContextManager[Span] = nullcontext(INVALID_SPAN)

    def start_span(self, *args: object, **kwargs: object) -> Span:
        return INVALID_SPAN

    def start_as_current_span(  # type: ignore[override]
        self, *args: object, **kwargs: object
    ) -> AbstractContextManager[Span]:
        return self._span_context


_NULL_TRACER = _NullTracer()


def _traces_active(config: TelemetryConfig) -> bool:
    return config.traces_enabled and config.emits(TelemetryLevel.COARSE)


def _metrics_active(config: TelemetryConfig) -> bool:
    return config.metrics_enabled and config.emits(TelemetryLevel.COARSE)


def build_tracer(
    config: TelemetryConfig,
    resource_attributes: Mapping[str, str],
    *,
    otlp_timeout_sec: float = _DEFAULT_OTLP_TIMEOUT_SEC,
) -> Tracer:
    """Build the process tracer, or the zero-allocation null twin when disabled.

    Disabled means ``traces_enabled`` is false or the level is below ``coarse``; either
    way, no ``TracerProvider`` is constructed and no OTLP exporter thread starts.
    """
    if not _traces_active(config):
        return _NULL_TRACER
    provider = TracerProvider(resource=Resource.create(dict(resource_attributes)))
    if config.otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=config.otlp_endpoint, timeout=otlp_timeout_sec
                )
            )
        )
    return provider.get_tracer(_TRACER_NAME)


def build_meter(
    config: TelemetryConfig,
    resource_attributes: Mapping[str, str],
    *,
    otlp_timeout_sec: float = _DEFAULT_OTLP_TIMEOUT_SEC,
) -> Meter:
    """Build the process meter, or the no-op twin when disabled.

    Disabled means ``metrics_enabled`` is false or the level is below ``coarse``;
    either way, no ``MeterProvider`` is constructed and no OTLP exporter thread starts.
    """
    if not _metrics_active(config):
        return NoOpMeter(_METER_NAME)
    readers: list[MetricReader] = []
    if config.otlp_endpoint:
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=config.otlp_endpoint, timeout=otlp_timeout_sec
                )
            )
        )
    provider = MeterProvider(
        resource=Resource.create(dict(resource_attributes)), metric_readers=readers
    )
    return provider.get_meter(_METER_NAME)
