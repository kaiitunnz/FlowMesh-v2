"""Engine-level tests for dispatchable dynamic children.

These prove that a spawned child acquires a stable dispatchable identity, that its
settlement flows through the task-keyed lifecycle (not the static forward-record path),
and that a live spawn/join fan-out closes with contract-equivalent progress and keyed
publication versus the trace-level ``spawn_child``/``settle_child`` driving.
"""

from server.orchestration import (
    BoundaryEvent,
    OrchestrationEngine,
    PublicationOutcome,
    ScopeBudget,
    WorkItemStatus,
)
from server.orchestration.state import InvocationState, ValueRef
from server.task.v2 import FrontendWorkflowSource, PersistedV2Workflow
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    BoundaryEventKind,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
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

_REGION_KINDS = {
    OperatorKind.BRANCH,
    OperatorKind.MERGE,
    OperatorKind.SPAWN,
    OperatorKind.JOIN,
    OperatorKind.LOOP_CONTEXT,
}


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
    cardinality: CardinalityKind = CardinalityKind.SINGLETON,
    release: ReleaseConditionKind = ReleaseConditionKind.SOURCE_SETTLED,
    keying: str | None = None,
) -> ResultDeclaration:
    return ResultDeclaration(
        output_id=output_id,
        source_ref=source_ref,
        cardinality=cardinality,
        release=release,
        visibility=Visibility.INTERNAL,
        keying=keying,
    )


def _bundle(
    ops: list[LogicalOperator],
    edges: list[TemplateEdge],
    results: tuple[ResultDeclaration, ...],
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
    source = FrontendWorkflowSource.capture("regions: true", "native", name="wf")
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def _fanout_bundle() -> PersistedV2Workflow:
    planner = _leaf("planner")
    spawn = SpawnRegion(operator_id="exp", source_ref="exp", child_template_ref="trial")
    join = JoinRegion(
        operator_id="collect",
        source_ref="collect",
        completion=JoinCompletion.ALL_SUCCEED,
    )
    return _bundle(
        [planner, spawn, join, _leaf("trial")],
        [
            TemplateEdge(from_op="planner", to_op="exp"),
            TemplateEdge(from_op="exp", to_op="collect"),
        ],
        (
            _decl(
                "results",
                "exp",
                cardinality=CardinalityKind.KEYED_COLLECTION,
                keying="hypothesis",
            ),
            _decl("summary", "collect", release=ReleaseConditionKind.SCOPE_CLOSED),
        ),
    )


def _engine() -> OrchestrationEngine:
    return OrchestrationEngine.build(
        "wfl-x", "owner", "org", _fanout_bundle(), budget=ScopeBudget()
    )


def _init_ref(i: int) -> ValueRef:
    return ValueRef(kind="legacy_task_result", legacy_task_id=f"planner#{i}")


def test_materialized_child_gets_a_dispatchable_ready_identity() -> None:
    eng = _engine()
    eng.on_succeeded("planner")  # opens the experiment region
    adv = eng.materialize_child("exp", value_ref=_init_ref(0))
    # The child readies under a stable dispatchable id, not the empty trace-level id.
    assert len(adv.ready) == 1
    child_task = adv.ready[0]
    assert child_task and child_task.startswith("act-")
    wi = eng.work_item(child_task)
    assert wi is not None and wi.status is WorkItemStatus.READY
    assert wi.legacy_task_id == child_task


def test_live_fanout_closes_join_over_dispatched_children() -> None:
    eng = _engine()
    eng.on_succeeded("planner")
    children = [
        eng.materialize_child("exp", value_ref=_init_ref(i)).ready[0] for i in range(3)
    ]
    eng.seal_spawn("exp")
    assert not eng.region_closed("collect")  # sealed, but children still in flight
    for child in children[:-1]:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
        assert not eng.region_closed("collect")
    eng.on_dispatched(children[-1], "w1")
    adv = eng.on_succeeded(children[-1])
    assert eng.region_closed("collect")  # closes on capability drain
    assert "collect" in adv.ready or eng.resolve_output("summary") is not None
    summary = eng.resolve_output("summary")
    assert summary is not None and summary.outcome is PublicationOutcome.SUCCESS
    keyed = [
        p
        for slot, p in eng._publications.items()  # type: ignore[attr-defined]
        if p.output_id == "results"
    ]
    assert len(keyed) == 3  # one keyed publication per dispatched child


def test_child_retry_preserves_work_item_and_invocation_identity() -> None:
    eng = _engine()
    eng.on_succeeded("planner")
    child = eng.materialize_child("exp", value_ref=_init_ref(0)).ready[0]
    eng.on_dispatched(child, "w1")
    wi_before = eng.work_item(child)
    assert wi_before is not None
    wi_id, inv_id = wi_before.work_item_id, wi_before.invocation_id
    adv = eng.on_failed(child, "worker lost", retryable=True)
    assert adv.retry == [child] and adv.failed == []
    eng.on_dispatched(child, "w2")  # relocated attempt
    wi_after = eng.work_item(child)
    assert wi_after is not None
    assert wi_after.work_item_id == wi_id and wi_after.invocation_id == inv_id
    assert len(wi_after.attempt_ids) == 2  # a second attempt under the same identity


def _leaf_engine() -> OrchestrationEngine:
    bundle = _bundle([_leaf("solo")], [], (_decl("legacy:solo", "solo"),))
    eng = OrchestrationEngine.build("wfl-x", "o", "g", bundle)
    eng.on_dispatched("solo", "w1")  # an attempt is in flight, about to yield
    return eng


def _trace_kinds(eng: OrchestrationEngine) -> set[str]:
    return {kind for kind, _ in eng.contract_trace()}


def test_invocation_boundary_records_durable_state_before_suspending() -> None:
    eng = _leaf_engine()
    before = len(eng._invocations)  # type: ignore[attr-defined]
    adv = eng.route_boundary_event(
        "solo", BoundaryEvent(kind=BoundaryEventKind.INVOCATION, interface="search")
    )
    assert adv.ready == [] and adv.failed == []
    wi = eng.work_item("solo")
    assert wi is not None and wi.status is WorkItemStatus.BLOCKED  # worker released
    # A durable invocation is recorded ISSUED before the work item suspends.
    issued = [
        i
        for i in eng._invocations.values()  # type: ignore[attr-defined]
        if i.state is InvocationState.ISSUED and i.work_item_id == wi.work_item_id
    ]
    assert len(eng._invocations) == before + 1  # type: ignore[attr-defined]
    assert issued


def test_yield_boundary_persists_the_continuation_and_suspends() -> None:
    eng = _leaf_engine()
    eng.route_boundary_event(
        "solo", BoundaryEvent(kind=BoundaryEventKind.YIELD, continuation="cont-blob")
    )
    wi = eng.work_item("solo")
    assert wi is not None
    assert wi.status is WorkItemStatus.BLOCKED and wi.continuation_ref == "cont-blob"


def test_state_access_boundary_records_without_suspending() -> None:
    eng = _leaf_engine()
    adv = eng.route_boundary_event(
        "solo", BoundaryEvent(kind=BoundaryEventKind.STATE_ACCESS, state_ref="ref-1")
    )
    assert adv.ready == []
    assert "state_access" in _trace_kinds(eng)
    wi = eng.work_item("solo")  # a state access is a query, not a suspension
    assert wi is not None and wi.status is WorkItemStatus.DISPATCHED


def test_child_failure_fails_all_succeed_join_without_static_cascade() -> None:
    eng = _engine()
    eng.on_succeeded("planner")
    children = [
        eng.materialize_child("exp", value_ref=_init_ref(i)).ready[0] for i in range(2)
    ]
    eng.seal_spawn("exp")
    eng.on_dispatched(children[0], "w1")
    eng.on_succeeded(children[0])
    eng.on_dispatched(children[1], "w1")
    adv = eng.on_failed(children[1], "trial crashed", retryable=False)
    assert children[1] in adv.failed
    assert eng.region_closed("collect")
    summary = eng.resolve_output("summary")
    assert summary is not None
    assert summary.outcome is PublicationOutcome.DECLARED_FAILURE
