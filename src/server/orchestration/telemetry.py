"""Synthesized-span emission over the durable orchestration ledger.

A synthesized span is built once, at a durable record's settle transition, with an
explicitly-set derived span id, explicit start/end times read from durable records, and
an explicitly-resolved parent. It is never held open across a call. Re-emission --
after a restart, or a repeated hook call within one process -- recomputes the same id
and the same timestamps from the same durable records, so it is byte-identical; the
in-process emitted-set below, and the store's own dedup, both exist to collapse it.

``TelemetrySpanEmitter`` is the sole owner of the four ledger-derived levels (operator,
episode, attempt, boundary), bound one per engine. ``WorkflowSpanEmitter`` is the sole
owner of the workflow root span, bound once per process and called from the workflow's
own completion detector rather than from the engine -- workflow status is derived on
read, so there is no ledger transition to hook for it.

Synthesis is observation, and both emitters are called from inside ledger transitions
that hold the runtime lock. Two properties keep it that way: every public method
absorbs its own failures, so no synthesis fault can interrupt a transition mid-mutation;
and every derivation reads an index kept incrementally over the ledger's append-only
collections, so no emit walks the whole ledger.
"""

import logging
from collections.abc import Callable
from functools import wraps

from opentelemetry.sdk.trace import Tracer as _SdkTracer
from opentelemetry.sdk.trace import _Span as _SdkSpan
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    Tracer,
    set_span_in_context,
)

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.telemetry.semconv import (
    LOGICAL_ACTIVATION_ID,
    LOGICAL_CHILD_INDEX,
    LOGICAL_LOOP_TIME,
    LOGICAL_OPERATOR_ID,
    LOGICAL_OPERATOR_KIND,
    LOGICAL_OUTCOME,
    LOGICAL_PARENT_ACTIVATION_ID,
    LOGICAL_SCOPE_ID,
    LOGICAL_WORKFLOW_ID,
    PHYSICAL_ALTERNATIVE_ID,
    PHYSICAL_ATTEMPT_ID,
    PHYSICAL_ATTEMPT_NO,
    PHYSICAL_INVOCATION_ID,
    PHYSICAL_TASK_ID,
    PHYSICAL_WORK_ITEM_ID,
    PHYSICAL_WORKER_ID,
    SPAN_ATTEMPT,
    SPAN_BOUNDARY,
    SPAN_EPISODE,
    SPAN_OPERATOR,
    SPAN_WORKFLOW,
)
from shared.utils.time import parse_iso_datetime

from .state import (
    Activation,
    Attempt,
    Invocation,
    InvocationState,
    OrchestrationEvent,
    Scope,
    WorkItem,
    WorkItemStatus,
)

__all__ = [
    "ActivationClassificationError",
    "NULL_SPAN_EMITTER",
    "NULL_WORKFLOW_SPAN_EMITTER",
    "TelemetrySpanEmitter",
    "WorkflowSpanEmitter",
    "build_span_emitter",
    "build_workflow_span_emitter",
]

# Operator kinds whose root activation settles inside the ledger and never
# dispatches; distinct vocabulary from an Activation's own kind
# (``child``/``iteration``/``region``).
_NO_EXTENT_OPERATOR_KINDS = frozenset(
    {"branch", "merge", "join", "spawn", "loop_context"}
)
# ``iteration`` activations (``engine.py::loop_feedback``) own neither a work item nor
# a scope: unlike a spawn child, the loop primitive materializes no dispatchable body
# for its own activation. Checked directly since it is already a first-class
# Activation.kind, not an operator id needing a cross-reference.
#
# ``leaf``/``agent`` root activations that name a spawn's ``child_template_ref`` are
# the same story under a different kind: ``engine.py::build`` excludes a child template
# from ``dispatchable`` (it is only ever instantiated as a "child"-kind activation per
# spawn, never dispatched itself), so it permanently owns neither a work item nor a
# scope. The work-item branch is checked first, so this is reachable only once that
# has already come back empty -- a genuinely dispatchable leaf/agent always owns a
# work item from the moment ``build()`` constructs it, so this never masks one that
# merely has not settled yet.
_NO_EXTENT_ACTIVATION_KINDS = frozenset({"iteration", "leaf", "agent"})

_TERMINAL_WI = frozenset({WorkItemStatus.SETTLED, WorkItemStatus.CANCELLED})
_TERMINAL_INVOCATION = frozenset(
    {
        InvocationState.TERMINAL,
        InvocationState.AMBIGUITY_TERMINAL,
        InvocationState.COMPENSATION_REQUIRED,
    }
)

_OFF_TELEMETRY_CONFIG = TelemetryConfig(
    level=TelemetryLevel.OFF,
    traces_enabled=False,
    metrics_enabled=False,
    sample_ratio=1.0,
    otlp_endpoint=None,
)

_logger = logging.getLogger("orchestration-telemetry")


class ActivationClassificationError(RuntimeError):
    """An activation's kind and shape match no known span-synthesis rule.

    Raised rather than silently deriving a wrong or missing extent, so a fifth
    activation class added later drops its span and logs instead of producing a
    plausible but incorrect trace.
    """


def _absorbs_faults[**P](method: Callable[P, None]) -> Callable[P, None]:
    """Contain every failure of one span-emitter entry point.

    Each entry point runs inside a ledger transition that has already begun mutating
    durable state, so an escaping exception would leave the transition half-applied
    and the workflow wedged. Synthesis is observation: a failure costs a span.
    """

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            method(*args, **kwargs)
        except Exception as exc:
            _logger.debug("Span synthesis failed in %s: %s", method.__name__, exc)

    return guarded


def _keep_earliest(
    index: dict[str, tuple[int, str]], key: str, stamp: tuple[int, str]
) -> None:
    current = index.get(key)
    if current is None or stamp[0] < current[0]:
        index[key] = stamp


def _keep_latest(
    index: dict[str, tuple[int, str]], key: str, stamp: tuple[int, str]
) -> None:
    current = index.get(key)
    if current is None or stamp[0] > current[0]:
        index[key] = stamp


def _appended[T](source: dict[str, T], indexed: int) -> list[T]:
    """The values added to an append-only dict beyond its first ``indexed`` entries."""
    count = len(source) - indexed
    if count <= 0:
        return []
    cursor = reversed(source)
    keys = [next(cursor) for _ in range(count)]
    return [source[key] for key in reversed(keys)]


def _iso_to_ns(value: str) -> int:
    dt = parse_iso_datetime(value)
    if dt is None:
        raise ValueError(f"empty or malformed timestamp: {value!r}")
    return int(dt.timestamp()) * 1_000_000_000 + dt.microsecond * 1_000


def _span_context(trace_id: int, span_id: int) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _export_span(
    tracer: Tracer | None,
    name: str,
    trace_id: int,
    span_id: int,
    parent_span_id: int | None,
    start_ns: int,
    end_ns: int,
    attributes: dict[str, str],
) -> None:
    """Build and export one synthesized span with an explicit id, parent, and extent.

    Bypasses ``Tracer.start_span``, which always mints a random span id from the
    provider's ``IdGenerator`` and exposes no parameter to override it, by
    constructing the span through the same ``_Span`` class the tracer itself builds
    internally, with the same sampler/resource/processor/scope it would otherwise use.
    A caller gates on its own enabled check first, but the ``None`` / non-SDK guard is
    repeated here too (the null twin, when telemetry is off) so this function is safe
    to call unconditionally.
    """
    if tracer is None or not isinstance(tracer, _SdkTracer):
        return
    own_context = _span_context(trace_id, span_id)
    parent_context = (
        _span_context(trace_id, parent_span_id) if parent_span_id is not None else None
    )
    span = _SdkSpan(
        name=name,
        context=own_context,
        parent=parent_context,
        sampler=tracer.sampler,
        resource=tracer.resource,
        attributes=attributes,
        span_processor=tracer.span_processor,
        instrumentation_info=tracer.instrumentation_info,
    )
    parent_ctx = (
        set_span_in_context(NonRecordingSpan(parent_context))
        if parent_context is not None
        else None
    )
    span.start(start_time=start_ns, parent_context=parent_ctx)
    span.end(end_time=end_ns)


class TelemetrySpanEmitter:
    """Derives and emits the operator/episode/attempt/boundary spans for one workflow.

    Constructed with just a tracer, a config, and the owning workflow id; ``attach``
    binds the engine's own live ledger collections (mutated in place, never
    reassigned, so a reference taken once stays current) and performs the rehydrate
    pass -- deriving and re-emitting every entity the loaded state already shows
    settled, which is a no-op on a freshly built engine and the restart-recovery path
    on a rehydrated one.
    """

    def __init__(
        self, tracer: Tracer | None, config: TelemetryConfig, workflow_id: str
    ) -> None:
        self._tracer = tracer
        self._config = config
        self._workflow_id = workflow_id
        self._trace_id = workflow_to_trace_id_int(workflow_id)
        self._emitted: set[tuple[int, int]] = set()

        self._activations: dict[str, Activation] = {}
        self._scopes: dict[str, Scope] = {}
        self._work_items: dict[str, WorkItem] = {}
        self._attempts: dict[str, Attempt] = {}
        self._invocations: dict[str, Invocation] = {}
        self._trace: list[OrchestrationEvent] = []
        self._released_scopes: set[str] = set()
        self._reset_indexes()

    @_absorbs_faults
    def attach(
        self,
        *,
        activations: dict[str, Activation],
        scopes: dict[str, Scope],
        work_items: dict[str, WorkItem],
        attempts: dict[str, Attempt],
        invocations: dict[str, Invocation],
        trace: list[OrchestrationEvent],
        released_scopes: set[str],
    ) -> None:
        self._activations = activations
        self._scopes = scopes
        self._work_items = work_items
        self._attempts = attempts
        self._invocations = invocations
        self._trace = trace
        self._released_scopes = released_scopes
        self._reset_indexes()
        self._rehydrate()

    # ------------------------------------------------------------------ #
    # Derived indexes over the ledger's append-only collections
    # ------------------------------------------------------------------ #

    def _reset_indexes(self) -> None:
        self._trace_seen = 0
        self._work_items_seen = 0
        self._scopes_seen = 0
        self._activations_seen = 0
        self._ready_by_work_item: dict[str, tuple[int, str]] = {}
        self._last_by_work_item: dict[str, tuple[int, str]] = {}
        self._recorded_by_invocation: dict[str, tuple[int, str]] = {}
        self._last_by_invocation: dict[str, tuple[int, str]] = {}
        self._work_item_ids_by_activation: dict[str, list[str]] = {}
        self._scope_by_owner: dict[str, str] = {}
        self._child_scope_ids: dict[str, list[str]] = {}
        self._activation_ids_by_scope: dict[str, list[str]] = {}

    def _sync_indexes(self) -> None:
        for event in self._trace[self._trace_seen :]:
            stamp = (event.seq, event.at)
            if event.work_item_id is not None:
                if event.kind == "work_item_ready":
                    _keep_earliest(self._ready_by_work_item, event.work_item_id, stamp)
                _keep_latest(self._last_by_work_item, event.work_item_id, stamp)
            if event.invocation_id is not None:
                if event.kind == "boundary_recorded":
                    _keep_earliest(
                        self._recorded_by_invocation, event.invocation_id, stamp
                    )
                _keep_latest(self._last_by_invocation, event.invocation_id, stamp)
        self._trace_seen = len(self._trace)

        for wi in _appended(self._work_items, self._work_items_seen):
            self._work_item_ids_by_activation.setdefault(wi.activation_id, []).append(
                wi.work_item_id
            )
        self._work_items_seen = len(self._work_items)

        for scope in _appended(self._scopes, self._scopes_seen):
            if scope.owner_activation_id is not None:
                self._scope_by_owner.setdefault(
                    scope.owner_activation_id, scope.scope_id
                )
            if scope.parent_scope_id is not None:
                self._child_scope_ids.setdefault(scope.parent_scope_id, []).append(
                    scope.scope_id
                )
        self._scopes_seen = len(self._scopes)

        for activation in _appended(self._activations, self._activations_seen):
            self._activation_ids_by_scope.setdefault(activation.scope_id, []).append(
                activation.activation_id
            )
        self._activations_seen = len(self._activations)

    def _rehydrate(self) -> None:
        if not self._emits(TelemetryLevel.COARSE):
            return
        for attempt in self._attempts.values():
            self.emit_attempt(attempt)
        for wi in self._work_items.values():
            self.emit_work_item(wi)
        for activation_id in list(self._activations):
            self.emit_activation(activation_id)
        for invocation in self._invocations.values():
            self.emit_boundary(invocation)

    def _emits(self, minimum: TelemetryLevel) -> bool:
        return (
            self._tracer is not None
            and self._config.traces_enabled
            and self._config.emits(minimum)
        )

    def _event_ns(self, index: dict[str, tuple[int, str]], key: str) -> int | None:
        stamp = index.get(key)
        return None if stamp is None else _iso_to_ns(stamp[1])

    # ------------------------------------------------------------------ #
    # Attempt
    # ------------------------------------------------------------------ #

    @_absorbs_faults
    def emit_attempt(self, attempt: Attempt) -> None:
        if not self._emits(TelemetryLevel.FULL):
            return
        span_id = derived_span_id(SpanIdKind.ATTEMPT, attempt.attempt_id)
        key = (self._trace_id, span_id)
        if key in self._emitted:
            return
        if attempt.started_at is None or attempt.finished_at is None:
            return
        parent_span_id = derived_span_id(SpanIdKind.WORK_ITEM, attempt.work_item_id)
        wi = self._work_items.get(attempt.work_item_id)
        attrs = {
            LOGICAL_WORKFLOW_ID: self._workflow_id,
            PHYSICAL_WORK_ITEM_ID: attempt.work_item_id,
            PHYSICAL_ATTEMPT_ID: attempt.attempt_id,
            PHYSICAL_ATTEMPT_NO: str(attempt.attempt_no),
        }
        if attempt.worker_id:
            attrs[PHYSICAL_WORKER_ID] = attempt.worker_id
        if attempt.alternative_id:
            attrs[PHYSICAL_ALTERNATIVE_ID] = attempt.alternative_id
        if wi is not None and wi.legacy_task_id:
            attrs[PHYSICAL_TASK_ID] = wi.legacy_task_id
        _export_span(
            self._tracer,
            SPAN_ATTEMPT,
            self._trace_id,
            span_id,
            parent_span_id,
            _iso_to_ns(attempt.started_at),
            _iso_to_ns(attempt.finished_at),
            attrs,
        )
        self._emitted.add(key)

    # ------------------------------------------------------------------ #
    # Episode (work item)
    # ------------------------------------------------------------------ #

    def _work_item_extent(self, wi: WorkItem) -> tuple[int, int] | None:
        if wi.status not in _TERMINAL_WI:
            return None
        start_ns = self._event_ns(self._ready_by_work_item, wi.work_item_id)
        if start_ns is None:
            return None
        attempt_ends = [
            _iso_to_ns(a.finished_at)
            for aid in wi.attempt_ids
            if (a := self._attempts.get(aid)) is not None and a.finished_at is not None
        ]
        event_end = self._event_ns(self._last_by_work_item, wi.work_item_id)
        candidates = attempt_ends + ([event_end] if event_end is not None else [])
        if not candidates:
            return None
        return start_ns, max(candidates)

    @_absorbs_faults
    def emit_work_item(self, wi: WorkItem) -> None:
        if not self._emits(TelemetryLevel.COARSE):
            return
        span_id = derived_span_id(SpanIdKind.WORK_ITEM, wi.work_item_id)
        key = (self._trace_id, span_id)
        if key in self._emitted:
            return
        self._sync_indexes()
        extent = self._work_item_extent(wi)
        if extent is None:
            return
        start_ns, end_ns = extent
        parent_span_id = derived_span_id(SpanIdKind.ACTIVATION, wi.activation_id)
        attrs = {
            LOGICAL_WORKFLOW_ID: self._workflow_id,
            LOGICAL_OPERATOR_ID: wi.operator_id,
            LOGICAL_ACTIVATION_ID: wi.activation_id,
            PHYSICAL_WORK_ITEM_ID: wi.work_item_id,
        }
        if wi.legacy_task_id:
            attrs[PHYSICAL_TASK_ID] = wi.legacy_task_id
        if wi.outcome is not None:
            attrs[LOGICAL_OUTCOME] = wi.outcome.value
        _export_span(
            self._tracer,
            SPAN_EPISODE,
            self._trace_id,
            span_id,
            parent_span_id,
            start_ns,
            end_ns,
            attrs,
        )
        self._emitted.add(key)

    # ------------------------------------------------------------------ #
    # Operator (activation)
    # ------------------------------------------------------------------ #

    def _scope_subtree_ids(self, root: str) -> list[str]:
        order = [root]
        seen = {root}
        cursor = 0
        while cursor < len(order):
            current = order[cursor]
            cursor += 1
            for scope_id in self._child_scope_ids.get(current, ()):
                if scope_id not in seen:
                    seen.add(scope_id)
                    order.append(scope_id)
        return order

    def _scope_subtree_extent(self, scope_id: str) -> tuple[int, int] | None:
        if scope_id not in self._released_scopes:
            return None
        starts: list[int] = []
        ends: list[int] = []
        for sid in self._scope_subtree_ids(scope_id):
            for activation_id in self._activation_ids_by_scope.get(sid, ()):
                extent = self._activation_extent(activation_id)
                if extent is not None:
                    starts.append(extent[0])
                    ends.append(extent[1])
        if not starts:
            return None
        return min(starts), max(ends)

    def _owned_scope_id(self, activation_id: str) -> str | None:
        return self._scope_by_owner.get(activation_id)

    def _activation_extent(self, activation_id: str) -> tuple[int, int] | None:
        """(start_ns, end_ns) once closed; ``None`` while still open or extent-less.

        Classification (not readiness) is checked first, so an activation whose shape
        this cannot recognize raises regardless of whether it happens to be closed yet.
        """
        activation = self._activations.get(activation_id)
        if activation is None:
            return None
        owned_work_items = [
            self._work_items[wi_id]
            for wi_id in self._work_item_ids_by_activation.get(activation_id, ())
        ]
        if owned_work_items:
            extents = [self._work_item_extent(wi) for wi in owned_work_items]
            if any(e is None for e in extents):
                return None
            starts = [e[0] for e in extents if e is not None]
            ends = [e[1] for e in extents if e is not None]
            return min(starts), max(ends)
        scope_id = self._owned_scope_id(activation_id)
        if scope_id is not None:
            return self._scope_subtree_extent(scope_id)
        if (
            activation.kind in _NO_EXTENT_ACTIVATION_KINDS
            or activation.kind in _NO_EXTENT_OPERATOR_KINDS
        ):
            return None
        raise ActivationClassificationError(
            f"activation {activation_id!r} of kind {activation.kind!r} owns no work "
            "item and no scope, and its kind is not a recognized no-extent shape"
        )

    def _activation_parent_span_id(self, activation: Activation) -> int:
        if activation.parent_activation_id is not None:
            return derived_span_id(
                SpanIdKind.ACTIVATION, activation.parent_activation_id
            )
        scope = self._scopes.get(activation.scope_id)
        if scope is not None and scope.owner_activation_id is not None:
            return derived_span_id(SpanIdKind.ACTIVATION, scope.owner_activation_id)
        return derived_span_id(SpanIdKind.WORKFLOW, self._workflow_id)

    @_absorbs_faults
    def emit_activation(self, activation_id: str) -> None:
        if not self._emits(TelemetryLevel.FINE):
            return
        span_id = derived_span_id(SpanIdKind.ACTIVATION, activation_id)
        key = (self._trace_id, span_id)
        if key in self._emitted:
            return
        activation = self._activations.get(activation_id)
        if activation is None:
            return
        self._sync_indexes()
        extent = self._activation_extent(activation_id)
        if extent is None:
            return
        start_ns, end_ns = extent
        parent_span_id = self._activation_parent_span_id(activation)
        attrs = {
            LOGICAL_WORKFLOW_ID: self._workflow_id,
            LOGICAL_OPERATOR_ID: activation.operator_id,
            LOGICAL_ACTIVATION_ID: activation.activation_id,
            LOGICAL_SCOPE_ID: activation.scope_id,
            LOGICAL_OPERATOR_KIND: activation.kind,
            LOGICAL_LOOP_TIME: str(activation.loop_time),
        }
        if activation.parent_activation_id is not None:
            attrs[LOGICAL_PARENT_ACTIVATION_ID] = activation.parent_activation_id
        if activation.child_index is not None:
            attrs[LOGICAL_CHILD_INDEX] = str(activation.child_index)
        _export_span(
            self._tracer,
            SPAN_OPERATOR,
            self._trace_id,
            span_id,
            parent_span_id,
            start_ns,
            end_ns,
            attrs,
        )
        self._emitted.add(key)

    # ------------------------------------------------------------------ #
    # Boundary (invocation)
    # ------------------------------------------------------------------ #

    @_absorbs_faults
    def emit_boundary(self, invocation: Invocation) -> None:
        if not self._emits(TelemetryLevel.FINE):
            return
        if invocation.state not in _TERMINAL_INVOCATION:
            return
        span_id = derived_span_id(SpanIdKind.INVOCATION, invocation.invocation_id)
        key = (self._trace_id, span_id)
        if key in self._emitted:
            return
        self._sync_indexes()
        start_ns = self._event_ns(
            self._recorded_by_invocation, invocation.invocation_id
        )
        if start_ns is None:
            return
        end_ns = self._event_ns(self._last_by_invocation, invocation.invocation_id)
        if end_ns is None:
            return
        parent_span_id = derived_span_id(SpanIdKind.WORK_ITEM, invocation.work_item_id)
        wi = self._work_items.get(invocation.work_item_id)
        attrs = {
            LOGICAL_WORKFLOW_ID: self._workflow_id,
            PHYSICAL_WORK_ITEM_ID: invocation.work_item_id,
            PHYSICAL_INVOCATION_ID: invocation.invocation_id,
        }
        if wi is not None:
            attrs[LOGICAL_OPERATOR_ID] = wi.operator_id
        _export_span(
            self._tracer,
            SPAN_BOUNDARY,
            self._trace_id,
            span_id,
            parent_span_id,
            start_ns,
            end_ns,
            attrs,
        )
        self._emitted.add(key)


NULL_SPAN_EMITTER = TelemetrySpanEmitter(None, _OFF_TELEMETRY_CONFIG, "")


class WorkflowSpanEmitter:
    """Sole owner of the ``flowmesh.workflow`` root span.

    Bound once per process, like ``ControlPlaneTracer`` -- not once per workflow, since
    it is called from a shared completion detector rather than from a per-workflow
    engine. That detector is already gated to fire at most once per workflow (a durable
    Redis key checked before it runs), so this carries no dedup bookkeeping of its own.
    """

    def __init__(self, tracer: Tracer | None, config: TelemetryConfig) -> None:
        self._tracer = tracer
        self._config = config

    @_absorbs_faults
    def emit(self, workflow_id: str, submitted_at: str, closed_at: str) -> None:
        if (
            self._tracer is None
            or not self._config.traces_enabled
            or not self._config.emits(TelemetryLevel.COARSE)
        ):
            return
        trace_id = workflow_to_trace_id_int(workflow_id)
        span_id = derived_span_id(SpanIdKind.WORKFLOW, workflow_id)
        _export_span(
            self._tracer,
            SPAN_WORKFLOW,
            trace_id,
            span_id,
            None,
            _iso_to_ns(submitted_at),
            _iso_to_ns(closed_at),
            {LOGICAL_WORKFLOW_ID: workflow_id},
        )


NULL_WORKFLOW_SPAN_EMITTER = WorkflowSpanEmitter(None, _OFF_TELEMETRY_CONFIG)


def build_span_emitter(
    tracer: Tracer | None, config: TelemetryConfig | None, workflow_id: str
) -> TelemetrySpanEmitter:
    """Build the per-workflow ledger-span emitter, or the null twin when disabled.

    The null twin is returned when telemetry is off (or no config was supplied) so a
    caller can construct it unconditionally; the emitter's own per-level gates then
    make every call a no-op.
    """
    if (
        tracer is None
        or config is None
        or not config.traces_enabled
        or not config.emits(TelemetryLevel.COARSE)
    ):
        return NULL_SPAN_EMITTER
    return TelemetrySpanEmitter(tracer, config, workflow_id)


def build_workflow_span_emitter(
    tracer: Tracer | None, config: TelemetryConfig | None
) -> WorkflowSpanEmitter:
    """Build the process-wide workflow-root emitter, or the null twin when disabled."""
    if (
        tracer is None
        or config is None
        or not config.traces_enabled
        or not config.emits(TelemetryLevel.COARSE)
    ):
        return NULL_WORKFLOW_SPAN_EMITTER
    return WorkflowSpanEmitter(tracer, config)
