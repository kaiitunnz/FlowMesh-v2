"""Synthesized-span emitter tests over the durable orchestration ledger.

These drive the orchestration engine directly (as ``test_v2_agent_harness.py`` and
``test_v2_dynamic_regions.py`` do) with a real, in-memory-recording tracer attached, and
inspect the exported spans rather than the ledger alone: the synthesis model's
correctness rests entirely on what it derives from durable state, so a wrong derivation
would otherwise produce a tree that looks plausible while being silently wrong.
"""

import asyncio
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from server.config import OrchestrationConfig
from server.orchestration import (
    OrchestrationEngine,
    ScopeBudget,
    WorkItemStatus,
)
from server.orchestration.state import Activation, BoundaryEvent, LedgerSnapshot
from server.orchestration.telemetry import (
    ActivationClassificationError,
    TelemetrySpanEmitter,
    WorkflowSpanEmitter,
)
from server.task.runtime import TaskRuntime
from server.task.v2 import FrontendWorkflowSource, PersistedV2Workflow
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    BoundarySignature,
    ChildRegionRef,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    LoopContextRegion,
    OperatorKind,
    Port,
    SpawnRegion,
)
from server.task.v2.representations.plan import PhysicalExecutionPlan, PhysicalNode
from server.task.v2.representations.results import (
    CardinalityKind,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from server.task.v2.representations.template import (
    LogicalWorkflowTemplate,
    SourceMapEntry,
    TemplateEdge,
)
from server.task.v2.representations.versioning import VersionId
from shared.tasks import TaskType
from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.telemetry.semconv import (
    SPAN_ATTEMPT,
    SPAN_BOUNDARY,
    SPAN_EPISODE,
    SPAN_OPERATOR,
)
from shared.utils.time import now_iso, parse_iso_datetime
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _WorkerRegistryStub,
)
from tests.server.telemetry_helpers import recording_tracer

_WORKFLOW_ID = "wfl-telemetry-test"
_SIGNATURE = BoundarySignature(
    events=(
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.SPAWN_SEAL,
    )
)
_GRANTED = frozenset({"model"})
_REGION_KINDS = {
    OperatorKind.BRANCH,
    OperatorKind.MERGE,
    OperatorKind.SPAWN,
    OperatorKind.JOIN,
    OperatorKind.LOOP_CONTEXT,
}


# --------------------------------------------------------------------------- #
# Bundle construction (a minimal subset of the helpers in the sibling engine
# test modules, adapted to pass an ``emitter`` through engine construction)
# --------------------------------------------------------------------------- #


def _leaf(op_id: str) -> LeafOperator:
    return LeafOperator(
        operator_id=op_id,
        source_ref=op_id,
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )


def _decl(
    output_id: str,
    source_ref: str,
    *,
    release: ReleaseConditionKind = ReleaseConditionKind.SOURCE_SETTLED,
) -> ResultDeclaration:
    return ResultDeclaration(
        output_id=output_id,
        source_ref=source_ref,
        cardinality=CardinalityKind.SINGLETON,
        release=release,
        visibility=Visibility.INTERNAL,
    )


def _bundle(
    ops: list[LogicalOperator], edges: list[TemplateEdge], results: tuple = ()
) -> PersistedV2Workflow:
    tv = VersionId(lineage="wfl-t:template", content_digest="td")
    pv = VersionId(lineage="wfl-t:plan", content_digest="pd")
    source_map = tuple(
        SourceMapEntry(
            logical_ref=op.operator_id,
            source_kind="region" if op.kind in _REGION_KINDS else "graph_node",
            source_id=op.operator_id,
        )
        for op in ops
    )
    nodes = tuple(
        PhysicalNode(
            node_id=f"phys:{op.operator_id}",
            source_ref=op.operator_id,
            logical_ref=op.operator_id,
        )
        for op in ops
    )
    template = LogicalWorkflowTemplate(
        version=tv,
        operators=tuple(ops),
        edges=tuple(edges),
        result_declarations=results,
        source_map=source_map,
    )
    plan = PhysicalExecutionPlan(plan_version=pv, template_version=tv, nodes=nodes)
    source = FrontendWorkflowSource.capture("telemetry: true", "native", name="wf")
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def _agent(op_id: str, *, regions: tuple[ChildRegionRef, ...] = ()) -> AgentOperator:
    return AgentOperator(
        operator_id=op_id,
        source_ref=op_id,
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=("model",), delegate=()),
        boundary=_SIGNATURE,
        child_region_refs=regions,
        outputs=(Port(name="out"),),
    )


def _region(
    role: str, entry: str
) -> tuple[ChildRegionRef, list[LogicalOperator], TemplateEdge]:
    spawn_id = f"{role}:spawn"
    join_id = f"{spawn_id}:join"
    spawn = SpawnRegion(
        operator_id=spawn_id,
        source_ref=spawn_id,
        outputs=(Port(name="children"),),
        child_template_ref=entry,
    )
    join = JoinRegion(
        operator_id=join_id,
        source_ref=join_id,
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ALL_SETTLED,
    )
    ref = ChildRegionRef(name=role, spawn_ref=spawn_id)
    return ref, [spawn, join], TemplateEdge(from_op=spawn_id, to_op=join_id)


def _spawning_agent_bundle() -> PersistedV2Workflow:
    """Agent "A" declaring one "worker" region whose entry is leaf "wbody"."""
    ref, region_ops, edge = _region("worker", "wbody")
    return _bundle(
        [_agent("A", regions=(ref,)), _leaf("wbody"), *region_ops],
        [edge],
        (_decl("out:A", "A"),),
    )


def _spawn_join_bundle() -> PersistedV2Workflow:
    """A plain top-level spawn/join region, owned by no agent."""
    spawn = SpawnRegion(operator_id="S", source_ref="S", child_template_ref="body")
    join = JoinRegion(
        operator_id="J", source_ref="J", completion=JoinCompletion.ALL_SETTLED
    )
    return _bundle(
        [spawn, join, _leaf("body")],
        [TemplateEdge(from_op="S", to_op="J")],
        (_decl("out:J", "J", release=ReleaseConditionKind.SCOPE_CLOSED),),
    )


def _loop_bundle() -> PersistedV2Workflow:
    loop = LoopContextRegion(operator_id="L", source_ref="L", loop_coordinate="t")
    return _bundle(
        [loop],
        [],
        (_decl("out:L", "L", release=ReleaseConditionKind.SCOPE_CLOSED),),
    )


def _engine(
    bundle: PersistedV2Workflow,
    *,
    emitter: TelemetrySpanEmitter | None = None,
    budget: ScopeBudget | None = None,
    granted_interfaces: frozenset[str] = _GRANTED,
) -> OrchestrationEngine:
    return OrchestrationEngine.build(
        _WORKFLOW_ID,
        "owner",
        "org",
        bundle,
        budget=budget,
        granted_interfaces=granted_interfaces,
        emitter=emitter,
    )


def _dispatch(eng: OrchestrationEngine, task: str, worker: str = "w1") -> None:
    eng.on_dispatched(task, worker)


def _span_ids(exporter: InMemorySpanExporter) -> set[tuple[int, int]]:
    return {
        (s.context.trace_id, s.context.span_id) for s in exporter.get_finished_spans()
    }


def _spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _sig(span: ReadableSpan) -> tuple[Any, ...]:
    """A full comparable signature for a byte-identity check."""
    return (
        span.name,
        span.context.trace_id,
        span.context.span_id,
        span.parent.span_id if span.parent is not None else None,
        span.start_time,
        span.end_time,
        tuple(sorted(_attrs(span).items())),
    )


# --------------------------------------------------------------------------- #
# Full-coverage tree: root agent, region opener, spawn child, retry, boundary
# --------------------------------------------------------------------------- #


def test_full_run_covers_every_activation_class_with_no_orphans_or_dupes() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)

    # Root agent activation, dispatched (covers "root", via work items).
    _dispatch(eng, "A")

    # SPAWN opens the region (kind="region" opener) and materializes one
    # dispatchable child leaf (covers "spawn child", via work items).
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    wi_a = eng.work_item("A")
    assert wi_a is not None
    child_task = next(
        act.activation_id
        for act in eng._activations.values()  # noqa: SLF001 - test inspection
        if act.kind == "child"
    )
    wi_child = eng.work_item(child_task)
    assert wi_child is not None

    # A retry on the child: two attempts sharing one episode span.
    _dispatch(eng, child_task)
    eng.on_failed(child_task, "transient", retryable=True)
    _dispatch(eng, child_task)
    eng.on_succeeded(child_task)

    # SPAWN_SEAL closes the region: with the sole child settled, the join
    # releases and the region-opener activation span is emitted.
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="s1",
            child_region_ref="worker",
        ),
    )
    assert eng.region_closed("worker:spawn:join")

    # A suspended mediated model boundary on the agent, settled and terminalized.
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="m0", interface="model"
        ),
    )
    eng.settle_boundary_outcome("A", "m0", value="answer")
    eng.terminalize_boundary_invocation("A", "m0")

    # Finish the agent.
    _dispatch(eng, "A")
    eng.on_succeeded("A")

    spans = exporter.get_finished_spans()
    assert spans, "expected at least one exported span"

    trace_id = workflow_to_trace_id_int(_WORKFLOW_ID)
    assert all(
        s.context.trace_id == trace_id for s in spans
    ), "every span must share the workflow's derived trace id (no orphan trace)"

    ids = [(s.context.trace_id, s.context.span_id) for s in spans]
    assert len(ids) == len(set(ids)), "no duplicate (trace_id, span_id) allowed"

    known_span_ids = {sid for _, sid in ids}
    for s in spans:
        if s.parent is not None:
            assert s.parent.span_id in known_span_ids or s.parent.span_id == (
                derived_span_id(SpanIdKind.WORKFLOW, _WORKFLOW_ID)
            ), f"orphan parent for span {s.name!r}"

    # Attempt: two attempts for the child (a retry), same parent episode span id.
    attempt_spans = _spans_named(exporter, SPAN_ATTEMPT)
    child_attempt_spans = [
        s
        for s in attempt_spans
        if _attrs(s)["flowmesh.physical.work_item_id"] == wi_child.work_item_id
    ]
    assert len(child_attempt_spans) == 2
    assert len({s.parent.span_id for s in child_attempt_spans if s.parent}) == 1
    expected_episode_span_id = derived_span_id(
        SpanIdKind.WORK_ITEM, wi_child.work_item_id
    )
    assert all(
        s.parent is not None and s.parent.span_id == expected_episode_span_id
        for s in child_attempt_spans
    )

    # Episode: agent work item + child work item.
    episode_spans = _spans_named(exporter, SPAN_EPISODE)
    assert len(episode_spans) == 2

    # Operator: root agent activation + region-opener activation.
    operator_spans = _spans_named(exporter, SPAN_OPERATOR)
    operator_activation_ids = {
        _attrs(s)["flowmesh.logical.activation_id"] for s in operator_spans
    }
    region_opener = next(
        act for act in eng._activations.values() if act.kind == "region"  # noqa: SLF001
    )
    assert wi_a.activation_id in operator_activation_ids
    assert region_opener.activation_id in operator_activation_ids
    # The region opener parents on its owning agent (parent_activation_id, step 1
    # of the precedence), not the workflow -- verifying the precedence held.
    region_span = next(
        s
        for s in operator_spans
        if _attrs(s)["flowmesh.logical.activation_id"] == region_opener.activation_id
    )
    assert region_span.parent is not None
    assert region_span.parent.span_id == derived_span_id(
        SpanIdKind.ACTIVATION, wi_a.activation_id
    )

    # Boundary: the terminalized mediated model invocation.
    boundary_spans = _spans_named(exporter, SPAN_BOUNDARY)
    assert len(boundary_spans) == 1


# --------------------------------------------------------------------------- #
# Restart: re-emission is byte-identical
# --------------------------------------------------------------------------- #


def test_restart_reemits_settled_spans_byte_identically() -> None:
    tracer_a, exporter_a, config = recording_tracer(TelemetryLevel.FULL)
    emitter_a = TelemetrySpanEmitter(tracer_a, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter_a)

    _dispatch(eng, "A")
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    child_task = next(
        act.activation_id
        for act in eng._activations.values()  # noqa: SLF001
        if act.kind == "child"
    )
    _dispatch(eng, child_task)
    eng.on_succeeded(child_task)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="s1",
            child_region_ref="worker",
        ),
    )

    before = {_sig(s) for s in exporter_a.get_finished_spans()}
    assert before, "expected spans from the driving sequence"

    # Simulate a restart: rebuild an engine from the persisted snapshot with a fresh
    # emitter instance, exactly as the runtime's rehydrate path constructs
    # OrchestrationEngine(snapshot, bundle, ...) directly.
    snapshot = eng.to_snapshot()
    tracer_b, exporter_b, config_b = recording_tracer(TelemetryLevel.FULL)
    emitter_b = TelemetrySpanEmitter(tracer_b, config_b, _WORKFLOW_ID)
    OrchestrationEngine(
        cast(LedgerSnapshot, snapshot),
        eng._bundle,  # noqa: SLF001 - the same compiled bundle, as the runtime holds it
        budget=ScopeBudget(),
        emitter=emitter_b,
    )

    after = {_sig(s) for s in exporter_b.get_finished_spans()}
    assert after, "expected spans re-emitted on rehydrate"
    assert after == before, "restart re-emission must be byte-identical"


# --------------------------------------------------------------------------- #
# Cancelled episode: closed, not open-ended
# --------------------------------------------------------------------------- #


def test_cancelled_never_dispatched_work_item_span_is_closed() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)

    _dispatch(eng, "A")
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    child_wi = next(
        wi
        for wi in eng._work_items.values()  # noqa: SLF001
        if eng._activations[wi.activation_id].kind == "child"  # noqa: SLF001
    )
    assert child_wi.status is WorkItemStatus.READY
    assert not child_wi.attempt_ids  # never dispatched: zero attempts

    # Cancel it directly, with no attempt ever having been issued.
    eng.on_cancelled(child_wi.legacy_task_id)
    assert child_wi.status is WorkItemStatus.CANCELLED

    episode_spans = _spans_named(exporter, SPAN_EPISODE)
    matching = [
        s
        for s in episode_spans
        if _attrs(s)["flowmesh.physical.work_item_id"] == child_wi.work_item_id
    ]
    assert len(matching) == 1, "a cancelled zero-attempt episode must still get a span"
    span = matching[0]
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.end_time >= span.start_time


# --------------------------------------------------------------------------- #
# Top-level (non-agent) spawn/loop region root: owns a scope, no work item
# --------------------------------------------------------------------------- #


def test_top_level_spawn_root_gets_an_operator_span_at_scope_release() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawn_join_bundle(), emitter=emitter, granted_interfaces=frozenset())

    # materialize_child (not spawn_child, which is trace-level only and never
    # admits) so the child gets a real work_item_ready event, matching how a
    # dynamic child actually becomes observable.
    advance = eng.materialize_child("S")
    child_wi = eng.work_item(advance.ready[0])
    assert child_wi is not None
    eng.settle_child(child_wi.activation_id)
    eng.seal_spawn("S")
    assert eng.region_closed("J")

    root_spawn_activation = next(
        act
        for act in eng._activations.values()  # noqa: SLF001
        if act.operator_id == "S" and act.kind == "spawn"
    )
    operator_spans = _spans_named(exporter, SPAN_OPERATOR)
    matching = [
        s
        for s in operator_spans
        if _attrs(s)["flowmesh.logical.activation_id"]
        == root_spawn_activation.activation_id
    ]
    assert len(matching) == 1


# --------------------------------------------------------------------------- #
# Loop iteration: an engine-level, no-workflow-callable no-extent shape
# --------------------------------------------------------------------------- #


def test_loop_iteration_activation_produces_no_span_and_does_not_raise() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_loop_bundle(), emitter=emitter, granted_interfaces=frozenset())

    iteration = eng.loop_feedback("L")
    eng.settle_iteration(iteration)
    eng.loop_seal("L")
    assert eng.region_closed("L")

    operator_spans = _spans_named(exporter, SPAN_OPERATOR)
    iteration_span = [
        s
        for s in operator_spans
        if _attrs(s)["flowmesh.logical.activation_id"] == iteration
    ]
    assert iteration_span == [], "an iteration activation has no observable extent"


# --------------------------------------------------------------------------- #
# Classification guard
# --------------------------------------------------------------------------- #


def test_an_unclassifiable_activation_drops_its_span_instead_of_raising() -> None:
    """The guard still refuses to guess a shape, and the refusal stays inside telemetry.

    ``attach()`` rehydrates every activation eagerly, at engine construction. A raise
    there would make a workflow holding one unclassifiable activation unrestartable,
    so the emitter absorbs it: the activation simply has no span.
    """
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    mystery = Activation(
        activation_id="act-mystery",
        instance_id=_WORKFLOW_ID,
        scope_id="scp-root",
        operator_id="op-mystery",
        kind="mystery",
    )
    emitter.attach(
        activations={mystery.activation_id: mystery},
        scopes={},
        work_items={},
        attempts={},
        invocations={},
        trace=[],
        released_scopes=set(),
    )

    assert _spans_named(exporter, SPAN_OPERATOR) == []
    with pytest.raises(ActivationClassificationError):
        emitter._activation_extent(mystery.activation_id)  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Ledger read-only-ness: no new event kind, no to_snapshot() call
# --------------------------------------------------------------------------- #


def _drive_a_representative_sequence(eng: OrchestrationEngine) -> None:
    _dispatch(eng, "A")
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    child_task = next(
        act.activation_id
        for act in eng._activations.values()  # noqa: SLF001
        if act.kind == "child"
    )
    _dispatch(eng, child_task)
    eng.on_failed(child_task, "transient", retryable=True)
    _dispatch(eng, child_task)
    eng.on_succeeded(child_task)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="s1",
            child_region_ref="worker",
        ),
    )
    _dispatch(eng, "A")
    eng.on_succeeded("A")


def test_telemetry_adds_no_ledger_event_kind() -> None:
    """The same driving sequence produces the identical event-kind trace whether a
    real emitter is attached or not -- the emitter reads the ledger and adds nothing
    to it."""
    plain = _engine(_spawning_agent_bundle())
    _drive_a_representative_sequence(plain)

    tracer, _exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    telemetered = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_a_representative_sequence(telemetered)

    plain_kinds = [k for k, _ in plain.contract_trace()]
    telemetered_kinds = [k for k, _ in telemetered.contract_trace()]
    assert telemetered_kinds == plain_kinds


def test_telemetry_never_triggers_a_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = OrchestrationEngine.to_snapshot

    def _counting_to_snapshot(self: OrchestrationEngine) -> Any:
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(OrchestrationEngine, "to_snapshot", _counting_to_snapshot)

    tracer, _exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_a_representative_sequence(eng)

    assert calls["n"] == 0, "no telemetry path may trigger to_snapshot()"


# --------------------------------------------------------------------------- #
# Every level: a connected tree, never a forest
# --------------------------------------------------------------------------- #


def _drive_every_span_kind(eng: OrchestrationEngine) -> None:
    """One sequence settling an activation, an episode, an attempt and a boundary."""
    _drive_a_representative_sequence(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="m0", interface="model"
        ),
    )
    eng.settle_boundary_outcome("A", "m0", value="answer")
    eng.terminalize_boundary_invocation("A", "m0")


@pytest.mark.parametrize("level", list(TelemetryLevel))
def test_every_level_emits_a_connected_tree(level: TelemetryLevel) -> None:
    """No level may name a parent span that level never emits.

    The workflow root is the one parent a ledger span may name without emitting it --
    a different emitter owns it -- so it is the only accepted unknown id.
    """
    tracer, exporter, config = recording_tracer(level)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_every_span_kind(eng)

    spans = exporter.get_finished_spans()
    if level is TelemetryLevel.OFF:
        assert not spans
        return

    assert spans, f"expected spans at {level}"
    workflow_span_id = derived_span_id(SpanIdKind.WORKFLOW, _WORKFLOW_ID)
    emitted = {s.context.span_id for s in spans}
    for span in spans:
        assert span.context.trace_id == workflow_to_trace_id_int(_WORKFLOW_ID)
        assert span.parent is not None, f"{span.name!r} has no parent at {level}"
        assert (
            span.parent.span_id in emitted or span.parent.span_id == workflow_span_id
        ), f"orphan parent for {span.name!r} at {level}"


def test_coarse_hangs_its_episodes_on_the_workflow_root() -> None:
    """``coarse`` emits no activation layer, so an episode parents on the workflow.

    The operator and activation ids stay on the episode as attributes, so what the
    flattening costs is nesting, not identity.
    """
    tracer, exporter, config = recording_tracer(TelemetryLevel.COARSE)
    emitter = TelemetrySpanEmitter(tracer, config, _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_every_span_kind(eng)

    assert _spans_named(exporter, SPAN_OPERATOR) == []
    episodes = _spans_named(exporter, SPAN_EPISODE)
    assert episodes
    workflow_span_id = derived_span_id(SpanIdKind.WORKFLOW, _WORKFLOW_ID)
    for span in episodes:
        assert span.parent is not None and span.parent.span_id == workflow_span_id
        assert _attrs(span)["flowmesh.logical.activation_id"]


# --------------------------------------------------------------------------- #
# Sample ratio: the trace-id-derived decision the SDK sampler never sees
# --------------------------------------------------------------------------- #


def _sampled_config(config: TelemetryConfig, ratio: float) -> TelemetryConfig:
    return replace(config, sample_ratio=ratio)


def test_an_unsampled_workflow_synthesizes_no_ledger_span() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, _sampled_config(config, 0.0), _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_every_span_kind(eng)

    assert not exporter.get_finished_spans()


def test_a_sampled_workflow_still_synthesizes_its_ledger_spans() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.FULL)
    emitter = TelemetrySpanEmitter(tracer, _sampled_config(config, 1.0), _WORKFLOW_ID)
    eng = _engine(_spawning_agent_bundle(), emitter=emitter)
    _drive_every_span_kind(eng)

    assert exporter.get_finished_spans()


def test_the_workflow_root_span_honors_the_same_sample_decision() -> None:
    tracer, exporter, config = recording_tracer(TelemetryLevel.COARSE)
    unsampled = WorkflowSpanEmitter(tracer, _sampled_config(config, 0.0))
    unsampled.emit(
        _WORKFLOW_ID, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00"
    )
    assert not exporter.get_finished_spans()

    sampled = WorkflowSpanEmitter(tracer, _sampled_config(config, 1.0))
    sampled.emit(_WORKFLOW_ID, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00")
    assert len(exporter.get_finished_spans()) == 1


# --------------------------------------------------------------------------- #
# Workflow record: submission time, not record-construction time
# --------------------------------------------------------------------------- #

_V2_LINEAR = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: linear}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


class _SubmitTimeRegistry(FakeRegistry):
    """Records the submission timestamp ``register`` stamps the workflow record with."""

    async def register_workflow_async(
        self,
        workflow_id: str,
        tasks: list[Any],
        v2: Any = None,
        submitted_at: str | None = None,
    ) -> None:
        # Falls back to stamping here, exactly as the record's own default does, so
        # the assertion measures the ordering rather than the plumbing.
        self.submitted_at = submitted_at or now_iso()
        await super().register_workflow_async(workflow_id, tasks, v2=v2)


def test_the_workflow_span_starts_no_later_than_its_earliest_child() -> None:
    """The workflow span's extent must cover the submit work it is meant to measure.

    ``submitted_at`` is what the root span starts at. A v2 submission compiles the
    template and drives the ledger's first work items before the workflow record is
    written, so a timestamp taken at record construction lands after the spans nested
    under it.
    """
    registry = _SubmitTimeRegistry()
    runtime = TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("v2-telemetry-submit"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )
    workflow_id, _ = asyncio.run(runtime.register("owner", "org", _V2_LINEAR))

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    events = engine._trace  # noqa: SLF001 - the ledger's own recorded timestamps
    assert events, "expected the build to record at least one ledger event"

    submitted_at = parse_iso_datetime(registry.submitted_at)
    child_starts = [at for event in events if (at := parse_iso_datetime(event.at))]
    assert submitted_at is not None and child_starts
    assert submitted_at <= min(child_starts)
