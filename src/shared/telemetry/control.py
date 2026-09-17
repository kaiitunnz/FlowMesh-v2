"""The control-plane instrumentation seam: one span per orchestration stage.

``ControlPlaneTracer`` opens a ``flowmesh.control.<stage>`` span at each of the
control-plane boundaries (compilation, engine construction, dispatch, resident
admission, and the periodic ledger snapshot). Every span sets its parent explicitly
from a derived trace id and a derived span id rather than relying on ambient context:
most of these boundaries run off the request thread, where an ambient-parented span
would silently root a random trace of its own instead of nesting under its workflow.
"""

import hashlib
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext

from opentelemetry.context import Context
from opentelemetry.trace import (
    INVALID_SPAN,
    NonRecordingSpan,
    Span,
    SpanContext,
    TraceFlags,
    Tracer,
    get_current_span,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .config import TelemetryConfig, TelemetryLevel
from .ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from .provider import build_tracer
from .semconv import (
    PHYSICAL_STAGE,
    PHYSICAL_WINDOW,
    ControlPlaneStage,
    ControlPlaneWindow,
    control_span_name,
)

__all__ = [
    "ControlPlaneTracer",
    "NULL_CONTROL_TRACER",
    "format_traceparent",
    "serve_trace_id_int",
]

_NULL_SPAN: AbstractContextManager[Span] = nullcontext(INVALID_SPAN)
_PROPAGATOR = TraceContextTextMapPropagator()


def serve_trace_id_int(serve_task_id: str, request_id: str) -> int:
    """Stable 128-bit trace id for a gated serve request's own trace root.

    A serve request is driven by an external principal and owns no ``workflow_id``,
    so it must not borrow the workflow bijection (contract §1.1) — it roots a trace
    keyed by its own identity instead.
    """
    digest = hashlib.blake2b(
        f"serve:{serve_task_id}:{request_id}".encode(), digest_size=16
    ).digest()
    return int.from_bytes(digest, "big") or 1


def _span_context(trace_id: int, span_id: int) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _explicit_parent(trace_id: int, span_id: int) -> Context:
    return set_span_in_context(NonRecordingSpan(_span_context(trace_id, span_id)))


def format_traceparent(trace_id: int, span_id: int) -> str:
    """The W3C ``traceparent`` value naming a not-yet-exported explicit parent span.

    Used both to open a control span at an explicit parent and to stamp the same
    derivation onto a carrier (a permit, a resident handoff) that a worker parents its
    own span on — one mechanism, two call shapes.
    """
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier, context=_explicit_parent(trace_id, span_id))
    return carrier.get("traceparent", "")


class ControlPlaneTracer:
    """Opens ``flowmesh.control.<stage>`` spans at explicit, derived parents.

    Null-twinned: constructed from a disabled ``TelemetryConfig``, every call returns
    the same reused no-op context manager, costing one attribute read and no
    allocation. ``ledger_snapshot`` is additionally gated to the ``full`` level even
    when the other twelve stages are on.
    """

    def __init__(self, tracer: Tracer, config: TelemetryConfig) -> None:
        self._tracer = tracer
        self._config = config
        self._enabled = config.traces_enabled and config.emits(TelemetryLevel.COARSE)

    @property
    def enabled(self) -> bool:
        """Whether any stage could emit at all, for guarding expensive prep work.

        A call site that must resolve extra identifiers (an engine lookup, a work item
        id) only to build a span's parent should check this first, so that work is
        skipped entirely when telemetry is off rather than paid for a span that turns
        out to be the null twin.
        """
        return self._enabled

    def _should_emit(self, stage: ControlPlaneStage) -> bool:
        if not self._enabled:
            return False
        if stage is ControlPlaneStage.LEDGER_SNAPSHOT:
            return self._config.emits(TelemetryLevel.FULL)
        return True

    def _open(
        self,
        stage: ControlPlaneStage,
        window: ControlPlaneWindow | None,
        context: Context | None,
        attributes: Mapping[str, str],
    ) -> AbstractContextManager[Span]:
        attrs: dict[str, str] = {PHYSICAL_STAGE: str(stage)}
        if window is not None:
            attrs[PHYSICAL_WINDOW] = str(window)
        attrs.update(attributes)
        return self._tracer.start_as_current_span(
            control_span_name(stage), context=context, attributes=attrs
        )

    def workflow_stage(
        self,
        stage: ControlPlaneStage,
        window: ControlPlaneWindow,
        workflow_id: str,
        **attributes: str,
    ) -> AbstractContextManager[Span]:
        """Open a stage span parented explicitly on the workflow span.

        For the five ``compile_*`` stages, ``engine_build`` and ``ds_initial_advance``:
        all run inside the submit request, which does have ambient context, but the
        parent is still built explicitly rather than relied upon (§3.0a).
        """
        if not self._should_emit(stage):
            return _NULL_SPAN
        context = _explicit_parent(
            workflow_to_trace_id_int(workflow_id),
            derived_span_id(SpanIdKind.WORKFLOW, workflow_id),
        )
        return self._open(stage, window, context, attributes)

    def episode_stage(
        self,
        stage: ControlPlaneStage,
        window: ControlPlaneWindow,
        workflow_id: str,
        work_item_id: str,
        **attributes: str,
    ) -> AbstractContextManager[Span]:
        """Open a stage span parented explicitly on the episode (work item) span.

        For ``dispatch`` (the dispatcher loop thread) and ``ds_drive`` when a work
        item is in scope (worker-event handling) — both run with no ambient span.
        """
        if not self._should_emit(stage):
            return _NULL_SPAN
        context = _explicit_parent(
            workflow_to_trace_id_int(workflow_id),
            derived_span_id(SpanIdKind.WORK_ITEM, work_item_id),
        )
        return self._open(stage, window, context, attributes)

    def boundary_stage(
        self,
        stage: ControlPlaneStage,
        window: ControlPlaneWindow,
        trace_id: int,
        invocation_id: str,
        **attributes: str,
    ) -> AbstractContextManager[Span]:
        """Open a stage span parented explicitly on the boundary (invocation) span.

        For ``admission``, ``permit`` and ``relay`` (async control paths, no ambient
        span). ``trace_id`` is supplied by the caller rather than derived from a
        ``workflow_id`` here, because a boundary may belong to a gated serve request
        that owns no workflow and roots its own trace instead (§1.1).
        """
        if not self._should_emit(stage):
            return _NULL_SPAN
        context = _explicit_parent(
            trace_id, derived_span_id(SpanIdKind.INVOCATION, invocation_id)
        )
        return self._open(stage, window, context, attributes)

    def ledger_snapshot(
        self, workflow_id: str, **attributes: str
    ) -> AbstractContextManager[Span]:
        """Open the ``ledger_snapshot`` span, gated to the ``full`` level.

        The one carve-out from explicit-parent-always (§3.0a): where a control-stage
        span already encloses the call site (``dispatch``, ``ds_drive``), that span is
        in the same call stack and ambient context reliably names it, so this nests
        under it instead of constructing a second, disconnected parent. Where none is
        open — a settle path with no enclosing stage — it falls back to an explicit
        workflow parent. Never a root either way.
        """
        if not self._should_emit(ControlPlaneStage.LEDGER_SNAPSHOT):
            return _NULL_SPAN
        if get_current_span().get_span_context().is_valid:
            return self._open(ControlPlaneStage.LEDGER_SNAPSHOT, None, None, attributes)
        context = _explicit_parent(
            workflow_to_trace_id_int(workflow_id),
            derived_span_id(SpanIdKind.WORKFLOW, workflow_id),
        )
        return self._open(ControlPlaneStage.LEDGER_SNAPSHOT, None, context, attributes)


def _null_telemetry_config() -> TelemetryConfig:
    return TelemetryConfig(
        level=TelemetryLevel.OFF,
        traces_enabled=False,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )


NULL_CONTROL_TRACER = ControlPlaneTracer(
    build_tracer(_null_telemetry_config(), {}), _null_telemetry_config()
)
