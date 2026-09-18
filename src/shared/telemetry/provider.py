"""Tracer/meter factory: builds the SDK provider when telemetry is on, and a null
twin that allocates nothing per call when it is off.

Each process calls ``build_tracer`` / ``build_meter`` once, from its own
``TelemetryConfig``, and threads the result explicitly rather than relying on OTel's
global provider.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Meter, NoOpMeter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import INVALID_SPAN, Span, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import Attributes

from .config import TelemetryConfig, TelemetryLevel
from .ids import trace_sampled
from .semconv import PHYSICAL_ERROR_TYPE

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


def _without_free_text(span: ReadableSpan) -> ReadableSpan:
    if not span.events and not span.status.description:
        return span
    return ReadableSpan(
        name=span.name,
        context=span.get_span_context(),
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes,
        events=(),
        links=span.links,
        kind=span.kind,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class PayloadFreeSpanExporter(SpanExporter):
    """Exports through ``exporter`` with the two untyped span fields removed.

    A span's attributes are chosen one key at a time by the site that sets them, but
    an exception crossing an instrumented block writes its message and stacktrace into
    a span event and a status description without any site naming them. Both are
    dropped here, as the span leaves the process, so that what a raise site happens to
    put in an exception message cannot decide whether a payload reaches the store.
    """

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([_without_free_text(span) for span in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


@contextmanager
def payload_free_span(
    tracer: Tracer,
    name: str,
    *,
    context: Context | None = None,
    attributes: Mapping[str, str] | None = None,
) -> Iterator[Span]:
    """Open a span that records a failure's type and never its text.

    A failing span is marked ERROR and carries the exception's class name, which the
    code fixes, rather than the message and stacktrace OTel records by default.
    """
    with tracer.start_as_current_span(
        name,
        context=context,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_attribute(PHYSICAL_ERROR_TYPE, type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise


class _WorkflowRatioSampler(Sampler):
    """Samples a live span by the same decision the synthesized spans use.

    The SDK's own ratio sampler reads the low 64 bits of the trace id, which a
    workflow-derived id does not distribute uniformly. Sharing one predicate is also
    what keeps a live span and a span synthesized from the ledger on the same side of
    the decision, so a traced workflow is traced whole.
    """

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        *args: object,
        attributes: Attributes = None,
        **kwargs: object,
    ) -> SamplingResult:
        sampled = trace_sampled(self._ratio, trace_id)
        decision = Decision.RECORD_AND_SAMPLE if sampled else Decision.DROP
        return SamplingResult(decision, attributes if sampled else None)

    def get_description(self) -> str:
        return f"FlowMeshWorkflowRatio{{{self._ratio}}}"


def _traces_active(config: TelemetryConfig) -> bool:
    return config.traces_enabled and config.emits(TelemetryLevel.COARSE)


def _metrics_active(config: TelemetryConfig) -> bool:
    return config.metrics_enabled and config.emits(TelemetryLevel.COARSE)


def build_tracer(
    config: TelemetryConfig,
    resource_attributes: Mapping[str, str],
) -> Tracer:
    """Build the process tracer, or the zero-allocation null twin when disabled.

    Disabled means ``traces_enabled`` is false or the level is below ``coarse``; either
    way, no ``TracerProvider`` is constructed and no OTLP exporter thread starts.
    """
    if not _traces_active(config):
        return _NULL_TRACER
    provider = TracerProvider(
        resource=Resource.create(dict(resource_attributes)),
        sampler=ParentBased(root=_WorkflowRatioSampler(config.sample_ratio)),
    )
    if config.otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                PayloadFreeSpanExporter(
                    OTLPSpanExporter(
                        endpoint=config.otlp_endpoint, timeout=config.otlp_timeout_sec
                    )
                )
            )
        )
    return provider.get_tracer(_TRACER_NAME)


def build_meter(
    config: TelemetryConfig,
    resource_attributes: Mapping[str, str],
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
                    endpoint=config.otlp_endpoint, timeout=config.otlp_timeout_sec
                )
            )
        )
    provider = MeterProvider(
        resource=Resource.create(dict(resource_attributes)), metric_readers=readers
    )
    return provider.get_meter(_METER_NAME)
