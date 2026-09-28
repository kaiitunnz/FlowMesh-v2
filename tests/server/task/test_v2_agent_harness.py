"""Engine-substrate tests for the agent-harness boundary path.

These prove a declared agent readies as a run-to-yield episode and, over the engine's
boundary machinery, suspends before a mediated model or tool action, releasing its lane
until its durable outcome is injected; the durable boundary envelope persists the
capsule, causal invocation id, and fabric-assigned idempotency key atomically before the
lane releases, and a forced re-drive maps a reissued facade call to its recorded key;
signature and authority checks reject undeclared tools, model interfaces, and child
regions through a durable typed outcome; and a facade spawn_agent selects one of an
agent's finite declared child regions, creating one child attenuated from that region's
entry, sealed per region, with recursive agent children reusing the declared region.
"""

import sys
from typing import Any

import pytest

from server.orchestration import (
    OrchestrationEngine,
    ProgressAxis,
    RegionError,
    ScopeBudget,
    WorkItemStatus,
)
from server.orchestration.state import (
    BoundaryEvent,
    DenialKind,
    InvocationState,
    WorkItem,
)
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

_SIGNATURE = BoundarySignature(
    events=(
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.EXTERNAL_EFFECT,
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.SPAWN_SEAL,
        BoundaryEventKind.YIELD,
        BoundaryEventKind.STATE_ACCESS,
    )
)
_GRANTED = frozenset({"model", "search"})
_REGION_KINDS = {
    OperatorKind.BRANCH,
    OperatorKind.MERGE,
    OperatorKind.SPAWN,
    OperatorKind.JOIN,
    OperatorKind.LOOP_CONTEXT,
}


# --------------------------------------------------------------------------- #
# Bundle construction
# --------------------------------------------------------------------------- #


def _agent(
    op_id: str,
    *,
    regions: tuple[ChildRegionRef, ...] = (),
    invoke: tuple[str, ...] = ("model", "search"),
    delegate: tuple[str, ...] = ("model",),
) -> AgentOperator:
    return AgentOperator(
        operator_id=op_id,
        source_ref=op_id,
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=invoke, delegate=delegate),
        boundary=_SIGNATURE,
        child_region_refs=regions,
        outputs=(Port(name="out"),),
    )


def _region(
    role: str,
    entry: str,
    *,
    spawn_id: str | None = None,
    invoke: tuple[str, ...] = ("model", "search"),
    delegate: tuple[str, ...] = ("model",),
) -> tuple[ChildRegionRef, list[LogicalOperator], TemplateEdge]:
    """One declared role region: a matched Spawn/Join pair over an entry target."""
    spawn_id = spawn_id or f"{role}:spawn"
    join_id = f"{spawn_id}:join"
    spawn = SpawnRegion(
        operator_id=spawn_id,
        source_ref=spawn_id,
        outputs=(Port(name="children"),),
        child_template_ref=entry,
        authority=AuthorityCeiling(invoke=invoke, delegate=delegate),
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


def _leaf(op_id: str) -> LeafOperator:
    return LeafOperator(
        operator_id=op_id,
        source_ref=op_id,
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )


def _decl(output_id: str, source_ref: str) -> ResultDeclaration:
    return ResultDeclaration(
        output_id=output_id,
        source_ref=source_ref,
        cardinality=CardinalityKind.SINGLETON,
        release=ReleaseConditionKind.SOURCE_SETTLED,
        visibility=Visibility.INTERNAL,
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
    source = FrontendWorkflowSource.capture("agent: true", "native", name="wf")
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def _engine(
    bundle: PersistedV2Workflow,
    *,
    budget: ScopeBudget | None = None,
    granted: frozenset[str] = _GRANTED,
):
    return OrchestrationEngine.build(
        "wfl-x", "owner", "org", bundle, granted_interfaces=granted, budget=budget
    )


def _multi_region_agent() -> PersistedV2Workflow:
    """An agent with two independent role regions: distinct authority and join."""
    r_ref, r_ops, r_edge = _region("researcher", "rbody")
    v_ref, v_ops, v_edge = _region("reviewer", "vbody", invoke=("model",), delegate=())
    v_ops[1] = v_ops[1].model_copy(update={"completion": JoinCompletion.ALL_SUCCEED})
    return _bundle(
        [
            _agent("A", regions=(r_ref, v_ref)),
            _leaf("rbody"),
            _leaf("vbody"),
            *r_ops,
            *v_ops,
        ],
        [r_edge, v_edge],
        (_decl("out:A", "A"),),
    )


def _solo_agent() -> PersistedV2Workflow:
    return _bundle([_agent("A")], [], (_decl("out:A", "A"),))


def _spawning_agent(
    *,
    child: LogicalOperator,
    role: str = "worker",
    region_invoke: tuple[str, ...] = ("model", "search"),
    region_delegate: tuple[str, ...] = ("model",),
) -> PersistedV2Workflow:
    ref, region_ops, edge = _region(
        role,
        child.operator_id,
        invoke=region_invoke,
        delegate=region_delegate,
    )
    return _bundle(
        [_agent("A", regions=(ref,)), child, *region_ops],
        [edge],
        (_decl("out:A", "A"),),
    )


def _dispatch_agent(eng: OrchestrationEngine, task: str = "A") -> str:
    eng.on_dispatched(task, "w1")
    return eng.work_item(task).activation_id  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Dispatch, suspend, resume
# --------------------------------------------------------------------------- #


def test_agent_dispatches_as_ready_episode() -> None:
    eng = _engine(_solo_agent())
    # The declared agent readies as a dispatchable episode at submission, not a
    # control-only settlement.
    assert eng.initial_advance().ready == ["A"]
    wi = eng.work_item("A")
    assert wi is not None and wi.operator_id == "A"


def test_agent_suspends_before_a_mediated_model_action() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model",
            continuation="after:c0",
        ),
    )
    wi = eng.work_item("A")
    # The lane releases: the work item suspends until the outcome is injected.
    assert wi is not None and wi.status is WorkItemStatus.BLOCKED
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.idempotency_key is not None
    # The causal request identity is recorded, and its durable invocation is ISSUED.
    assert env.invocation_id is not None
    model_inv = eng._invocations[env.invocation_id]  # type: ignore[attr-defined]
    assert model_inv.state is InvocationState.ISSUED
    assert env.continuation == "after:c0"  # capsule persisted before the lane released
    # The finished attempt is closed, so the work item holds no worker while it waits.
    attempt = eng._attempts[wi.attempt_ids[-1]]  # type: ignore[attr-defined]
    assert attempt.status.value == "succeeded" and attempt.finished_at is not None


def test_stranded_model_boundary_is_listed_for_resettlement() -> None:
    # A model boundary suspended with no durable outcome is stranded across a crash (the
    # off-lane settle ran in memory), so a restart can re-issue it from the envelope.
    eng = _engine(_solo_agent())
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model",
            request_payload="q",
        ),
    )
    pending = eng.pending_tool_dispatches()
    assert len(pending) == 1
    env = pending[0]
    assert (env.task_id, env.call_correlation, env.interface, env.request_payload) == (
        "A",
        "c0",
        "model",
        "q",
    )
    assert env.invocation_id
    # Once settled, it is no longer stranded.
    eng.settle_boundary_outcome("A", "c0", value="answer")
    assert eng.pending_tool_dispatches() == []


def test_a_stranded_search_boundary_carries_its_tool_interface() -> None:
    # A crash-stranded search invocation is re-listed with interface "search/v1", so
    # recovery routes it to the tool broker rather than the model settler.
    eng = _engine(
        _bundle(
            [_agent("A", invoke=("search/v1", "model"), delegate=())],
            [],
            (_decl("out:A", "A"),),
        ),
        granted=frozenset({"search/v1", "model"}),
    )
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="s0",
            interface="search/v1",
            request_payload='{"query": "q"}',
        ),
    )
    pending = eng.pending_tool_dispatches()
    assert len(pending) == 1
    assert pending[0].interface == "search/v1"
    assert pending[0].request_payload == '{"query": "q"}'


def test_only_the_durable_outcome_re_readies_a_suspended_episode() -> None:
    eng = _engine(_solo_agent())
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="model"
        ),
    )
    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.BLOCKED
    # Only the durable outcome re-readies the lane; then the episode settles terminally
    # and its declared output is readable.
    assert eng.deliver_boundary_outcome("A", "c0").ready == ["A"]
    eng.on_dispatched("A", "w1")
    eng.on_succeeded("A")
    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.SETTLED
    pub = eng.output_publication("out:A")
    assert pub is not None and pub.outcome.value == "success"


# --------------------------------------------------------------------------- #
# Idempotency key and re-drive correlation
# --------------------------------------------------------------------------- #


def test_redrive_maps_to_the_recorded_idempotency_key() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    request = BoundaryEvent(
        kind=BoundaryEventKind.EXTERNAL_EFFECT,
        call_correlation="c0",
        interface="search",
    )
    eng.route_boundary_event("A", request)
    env = eng.boundary_envelope(act, "c0")
    assert env is not None
    key, invocations = env.idempotency_key, len(eng._invocations)  # type: ignore[attr-defined]
    # A forced re-drive of the same facade call under a fresh attempt reissues the
    # request; it maps to the recorded key and creates no second target effect.
    eng.on_dispatched("A", "w2")
    eng.route_boundary_event("A", request)
    again = eng.boundary_envelope(act, "c0")
    assert again is not None and again.idempotency_key == key
    assert len(eng._invocations) == invocations  # type: ignore[attr-defined]
    assert "boundary_redriven" in {k for k, _ in eng.contract_trace()}


def test_boundary_envelope_survives_rehydration() -> None:
    bundle = _solo_agent()
    eng = _engine(bundle)
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="model"
        ),
    )
    key = eng.boundary_envelope(act, "c0").idempotency_key  # type: ignore[union-attr]
    restored = OrchestrationEngine(eng.to_snapshot(), bundle)
    env = restored.boundary_envelope(act, "c0")
    assert env is not None and env.idempotency_key == key


# --------------------------------------------------------------------------- #
# Signature and authority validation
# --------------------------------------------------------------------------- #


def test_undeclared_tool_is_denied_without_creating_work() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    before = len(eng._invocations)  # type: ignore[attr-defined]
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="danger"
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    # An undeclared tool is a durable typed denial, not a silent no-op — no invocation.
    assert env is not None and env.denial is DenialKind.AUTHORITY
    assert len(eng._invocations) == before  # type: ignore[attr-defined]
    assert "authority_denied" in {k for k, _ in eng.contract_trace()}


def test_denied_boundary_redrive_is_idempotent() -> None:
    eng = _engine(_solo_agent())
    _dispatch_agent(eng)
    request = BoundaryEvent(
        kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="danger"
    )
    eng.route_boundary_event("A", request)
    decisions = len(eng._decisions)  # type: ignore[attr-defined]
    # A re-driven denial maps to the recorded call rather than re-denying it.
    eng.on_dispatched("A", "w2")
    eng.route_boundary_event("A", request)
    assert len(eng._decisions) == decisions  # type: ignore[attr-defined]
    assert "boundary_redriven" in {k for k, _ in eng.contract_trace()}


def test_undeclared_boundary_kind_is_denied() -> None:
    # An agent whose signature omits SPAWN cannot yield a spawn boundary.
    ref, region_ops, edge = _region("worker", "child")
    narrow = _agent("A", regions=(ref,)).model_copy(
        update={"boundary": BoundarySignature(events=(BoundaryEventKind.INVOCATION,))}
    )
    eng = _engine(
        _bundle([narrow, _leaf("child"), *region_ops], [edge], (_decl("out:A", "A"),))
    )
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="c0",
            child_region_ref="worker",
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.denial is DenialKind.AUTHORITY


def test_undeclared_region_is_denied() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="c0",
            child_region_ref="other",
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.denial is DenialKind.AUTHORITY


def test_raw_operator_id_cannot_select_a_region() -> None:
    # A spawn that names a raw operator id rather than a declared role is denied: it
    # cannot install topology or reach a target the agent did not declare a region for.
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.denial is DenialKind.AUTHORITY


# --------------------------------------------------------------------------- #
# spawn_agent facade
# --------------------------------------------------------------------------- #


def test_spawn_agent_creates_one_attenuated_child_with_a_seal() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    adv = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="c0",
            child_region_ref="worker",
        ),
    )
    # Exactly one declared child activation with a dispatchable identity.
    assert len(adv.ready) == 1 and adv.ready[0].startswith("act-")
    grant = eng.grant_for("worker:spawn")
    assert grant is not None and grant.parent_grant_id == eng.instance.root_grant_id
    # The delegated grant is attenuated: it cannot widen the pinned envelope, and its
    # delegate face cannot widen its own invoke face.
    assert set(grant.invoke) <= _GRANTED
    assert set(grant.delegate) <= set(grant.invoke)
    # A spawn does not seal child-init; an explicit spawn seal for the region closes it.
    region_scope = eng.region_scope_for(act, "worker")
    cap = eng.capability(region_scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "open"
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="c1",
            child_region_ref="worker",
        ),
    )
    cap = eng.capability(region_scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "sealed"


def test_agent_terminal_completion_seals_its_child_init() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="c0",
            child_region_ref="worker",
        ),
    ).ready[0]

    eng.on_dispatched(child, "w1")
    eng.on_succeeded(child)
    eng.on_succeeded("A")  # terminal completion supplies the seal
    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.closed


def _recursive_agent_bundle() -> PersistedV2Workflow:
    # A declares a region whose entry is agent ``child``; ``child`` declares a region
    # whose entry is itself, so each spawn re-enters the finite declared region.
    child = _agent("child", regions=(ChildRegionRef(name="self", spawn_ref="rec"),))
    child_ref, child_region, child_edge = _region("self", "child", spawn_id="rec")
    a_ref, a_region, a_edge = _region("worker", "child")
    return _bundle(
        [_agent("A", regions=(a_ref,)), child, *a_region, *child_region],
        [a_edge, child_edge],
        (_decl("out:A", "A"),),
    )


def test_recursive_agent_child_reuses_the_declared_region() -> None:
    eng = _engine(_recursive_agent_bundle(), budget=ScopeBudget(max_scope_depth=8))
    act = _dispatch_agent(eng)
    lvl1 = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="c0",
            child_region_ref="worker",
        ),
    ).ready[0]
    # The materialized child agent is itself dispatchable and can spawn its own child.
    eng.on_dispatched(lvl1, "w1")
    lvl2 = eng.route_boundary_event(
        lvl1,
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_region_ref="self"
        ),
    ).ready[0]
    assert lvl2.startswith("act-") and lvl2 != lvl1
    # The template still holds exactly the declared operators: recursion reused the
    # region rather than growing the topology.
    assert {op.operator_id for op in eng._bundle.template.operators} == {  # type: ignore[attr-defined]
        "A",
        "child",
        "worker:spawn",
        "worker:spawn:join",
        "rec",
        "rec:join",
    }
    # A's worker region owns a child-init scope (for lvl1); lvl1's self region owns a
    # nested one (for lvl2), one level deeper — recursion nests scopes by depth.
    sa = eng.region_scope_for(act, "worker")
    s1 = eng.region_scope_for(lvl1, "self")
    assert sa is not None and s1 is not None and sa != s1
    by_id = {s.scope_id: s for s in eng.to_snapshot().scopes}
    assert by_id[s1].depth == by_id[sa].depth + 1


def test_recursive_agent_depth_budget_is_enforced() -> None:
    eng = _engine(_recursive_agent_bundle(), budget=ScopeBudget(max_scope_depth=1))
    _dispatch_agent(eng)
    # The finite SCC / depth budget trips: materializing a recursive agent child
    # reserves a level for the child's own region scope, exceeding the depth budget.
    try:
        eng.route_boundary_event(
            "A",
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN,
                call_correlation="c0",
                child_region_ref="worker",
            ),
        )
    except RegionError:
        assert "scope_budget_exhausted" in {k for k, _ in eng.contract_trace()}
        return
    raise AssertionError("expected the depth budget to reject the nested agent scope")


# --------------------------------------------------------------------------- #
# Latent-path safety: rejection, correlation, and query boundaries
# --------------------------------------------------------------------------- #


def test_rejected_spawn_records_no_envelope_and_no_phantom_ack() -> None:
    # The depth budget rejects the nested agent scope: the spawn raises and records no
    # envelope, so a later re-drive has no phantom-accepted record to deliver.
    eng = _engine(_recursive_agent_bundle(), budget=ScopeBudget(max_scope_depth=1))
    act = _dispatch_agent(eng)
    spawn = BoundaryEvent(
        kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_region_ref="worker"
    )
    with pytest.raises(RegionError):
        eng.route_boundary_event("A", spawn)
    assert eng.boundary_envelope(act, "c0") is None  # no phantom-accepted record
    # A re-drive is not short-circuited into a success — the rejection stands.
    with pytest.raises(RegionError):
        eng.route_boundary_event("A", spawn)


def test_state_access_is_resolved_inline_without_suspending() -> None:
    eng = _engine(_solo_agent())
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.STATE_ACCESS,
            call_correlation="c0",
            state_ref="ref-1",
        ),
    )
    # A state access is a query the engine resolves inline, not a deferred boundary: the
    # lane is not suspended and the access is traced.
    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.DISPATCHED
    assert "state_access" in {k for k, _ in eng.contract_trace()}


def test_dedup_capable_boundary_without_correlation_is_rejected() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    # A mediated dedup-capable boundary must carry a stable correlation, or a re-drive
    # could duplicate a target effect; the engine refuses one without it.
    for event in (
        BoundaryEvent(kind=BoundaryEventKind.SPAWN, child_region_ref="worker"),
        BoundaryEvent(kind=BoundaryEventKind.INVOCATION, interface="model"),
        BoundaryEvent(kind=BoundaryEventKind.EXTERNAL_EFFECT, interface="search"),
    ):
        with pytest.raises(RegionError):
            eng.route_boundary_event("A", event)


def test_request_payload_round_trips_on_the_envelope() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model",
            request_payload='{"q": "hi"}',
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.request_payload == '{"q": "hi"}'


# --------------------------------------------------------------------------- #
# Finite child-region contract: multiple roles, per-entry attenuation, per-
# region seal
# --------------------------------------------------------------------------- #


def _spawn(role: str, call: str) -> BoundaryEvent:
    return BoundaryEvent(
        kind=BoundaryEventKind.SPAWN, call_correlation=call, child_region_ref=role
    )


def test_multiple_role_regions_each_spawn_and_settle_with_the_agent() -> None:
    eng = _engine(_multi_region_agent())
    act = _dispatch_agent(eng)
    # One child per declared role, selected by role rather than a raw operator id.
    c1 = eng.route_boundary_event("A", _spawn("researcher", "c0")).ready[0]
    c2 = eng.route_boundary_event("A", _spawn("reviewer", "c1")).ready[0]
    assert c1 != c2
    for child in (c1, c2):
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    eng.on_succeeded("A")  # terminal completion settles every still-open region
    # Observable end-to-end outcome: the agent's declared output resolves SUCCESS ...
    pub = eng.output_publication("out:A")
    assert pub is not None and pub.outcome.value == "success"
    # ... and each role region's child-init progress closes independently.
    for role in ("researcher", "reviewer"):
        cap = eng.capability(eng.region_scope_for(act, role), ProgressAxis.CHILD_INIT)
        assert cap is not None and cap.closed


def test_authority_attenuates_from_the_selected_region_not_the_parent() -> None:
    # The parent broadly holds "broad"; the narrow region's ceiling does not. A child
    # spawned through that region is denied "broad" even though it declares it — the
    # attenuation comes from the selected entry, not the parent's blanket ceiling.
    granted = frozenset({"model", "search", "broad"})
    sub = _agent("sub", invoke=("model", "broad"), delegate=("model",))
    ref, region_ops, edge = _region(
        "narrow", "sub", invoke=("model",), delegate=("model",)
    )
    agent = _agent(
        "A", invoke=("model", "search", "broad"), delegate=("model", "broad")
    )
    agent = agent.model_copy(update={"child_region_refs": (ref,)})
    bundle = _bundle([agent, sub, *region_ops], [edge], (_decl("out:A", "A"),))
    eng = _engine(bundle, granted=granted)
    _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("narrow", "c0")).ready[0]
    # The region's delegated grant is attenuated below the parent: it drops "broad".
    grant = eng.grant_for("narrow:spawn")
    assert grant is not None and "broad" not in grant.invoke and "model" in grant.invoke
    eng.on_dispatched(child, "w1")
    eng.route_boundary_event(
        child,
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="i0", interface="broad"
        ),
    )
    denied = eng.boundary_envelope(child, "i0")
    assert denied is not None and denied.denial is not None
    # An interface the region does grant is admitted for the same child.
    eng.route_boundary_event(
        child,
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="i1", interface="model"
        ),
    )
    admitted = eng.boundary_envelope(child, "i1")
    assert admitted is not None and admitted.denial is None


def test_spawn_seal_closes_only_its_region() -> None:
    eng = _engine(_multi_region_agent())
    act = _dispatch_agent(eng)
    eng.route_boundary_event("A", _spawn("researcher", "r0"))
    eng.route_boundary_event("A", _spawn("reviewer", "v0"))
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="rs",
            child_region_ref="researcher",
        ),
    )
    # A late child in the sealed region is rejected (per-region late-child prevention).
    with pytest.raises(RegionError):
        eng.route_boundary_event("A", _spawn("researcher", "r1"))
    # The other region is untouched: it still admits a child.
    assert len(eng.route_boundary_event("A", _spawn("reviewer", "v1")).ready) == 1
    researcher = eng.capability(
        eng.region_scope_for(act, "researcher"), ProgressAxis.CHILD_INIT
    )
    reviewer = eng.capability(
        eng.region_scope_for(act, "reviewer"), ProgressAxis.CHILD_INIT
    )
    assert researcher is not None and researcher.status.value == "sealed"
    assert reviewer is not None and reviewer.status.value == "open"


def test_region_state_survives_rehydration() -> None:
    bundle = _spawning_agent(child=_leaf("child"))
    eng = _engine(bundle)
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("worker", "c0")).ready[0]
    eng.on_dispatched(child, "w1")
    eng.on_succeeded(child)
    # The region opener, scope, and child-init account rebuild from the snapshot.
    restored = OrchestrationEngine(eng.to_snapshot(), bundle)
    scope = restored.region_scope_for(act, "worker")
    assert scope is not None
    restored.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    cap = restored.capability(scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.closed


def _normalized_legacy_bundle(
    *, invoke: tuple[str, ...], delegate: tuple[str, ...]
) -> PersistedV2Workflow:
    """A legacy child_template_ref agent, run through the compiler normalization."""
    from server.task.v2.compiler.project import LoweringAccumulator
    from server.task.v2.compiler.regions import normalize_agent_child_regions

    legacy = AgentOperator(
        operator_id="A",
        source_ref="A",
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=invoke, delegate=delegate),
        boundary=_SIGNATURE,
        child_template_ref="child",
        outputs=(Port(name="out"),),
    )
    acc = LoweringAccumulator()
    acc.operators.extend([legacy, _leaf("child")])
    normalize_agent_child_regions(acc)
    return _bundle(acc.operators, acc.edges, (_decl("out:A", "A"),))


def test_normalized_legacy_agent_child_is_bounded_by_delegate() -> None:
    # A legacy agent that invokes more than it delegates normalizes to a compat region
    # whose child grant is bounded by the delegate face, never the fuller invoke face.
    eng = _engine(
        _normalized_legacy_bundle(invoke=("model", "search"), delegate=("model",))
    )
    _dispatch_agent(eng)
    eng.route_boundary_event("A", _spawn("child", "c0"))
    grant = eng.grant_for("A:child")
    assert grant is not None
    assert "model" in grant.invoke and "search" not in grant.invoke


def test_terminal_completion_settles_a_never_entered_region() -> None:
    eng = _engine(_multi_region_agent())
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("researcher", "c0")).ready[0]
    eng.on_dispatched(child, "w1")
    eng.on_succeeded(child)
    eng.on_succeeded("A")
    # The entered region closes on drain; the never-entered region opens as zero-child
    # and its join releases, so a declared region never hangs a downstream consumer.
    for role in ("researcher", "reviewer"):
        cap = eng.capability(eng.region_scope_for(act, role), ProgressAxis.CHILD_INIT)
        assert cap is not None and cap.closed
    assert eng.region_closed("reviewer:spawn:join")


def test_terminal_failure_fails_a_never_entered_region() -> None:
    eng = _engine(_multi_region_agent())
    act = _dispatch_agent(eng)

    failed = eng.on_failed("A", "boom", retryable=False).failed

    # A failed agent's unused region is not an empty one: it opens no scope, seals
    # nothing, and its template and join fail with it.
    assert failed[0] == "A" and {"rbody", "vbody"} <= set(failed)
    for role in ("researcher", "reviewer"):
        assert eng.region_scope_for(act, role) is None
    assert set(eng.to_snapshot().failed_regions) == {
        "researcher:spawn",
        "researcher:spawn:join",
        "reviewer:spawn",
        "reviewer:spawn:join",
    }
    kinds = {kind for kind, _ in eng.contract_trace()}
    assert "child_init_sealed" not in kinds and "join_released" not in kinds


def test_an_ambiguity_terminal_fails_a_never_entered_region() -> None:
    eng = _engine(_multi_region_agent())
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model",
            request_digest="sha256:abc",
        ),
    )

    failed = eng.on_uncertain("A").failed

    assert failed[0] == "A" and {"rbody", "vbody"} <= set(failed)
    assert eng.region_scope_for(act, "researcher") is None
    assert "reviewer:spawn:join" in eng.to_snapshot().failed_regions


def test_terminalizing_leaves_an_already_terminal_invocation_alone() -> None:
    eng = _engine(_multi_region_agent())
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model",
            request_digest="sha256:abc",
        ),
    )
    (invocation_id,) = eng.terminalize_unsettled_invocations()

    assert eng.terminalize_unsettled_invocations() == []
    assert eng.boundary_invocation_completed(invocation_id) is False


def test_terminal_failure_fails_an_entered_region_while_its_children_drain() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("worker", "c0")).ready[0]
    eng.on_dispatched(child, "w1")
    # The agent fails terminally while its region is open with an in-flight child.
    eng.on_failed("A", "boom", retryable=False)
    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "sealed" and not cap.closed
    assert "worker:spawn:join" in eng.to_snapshot().failed_regions
    # The child drains under the drain residual, and the join never releases.
    eng.on_succeeded(child)
    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.closed
    assert "join_released" not in {kind for kind, _ in eng.contract_trace()}


def test_terminal_failure_cancels_an_entered_region_under_a_cancel_residual() -> None:
    bundle = _spawning_agent(child=_leaf("child"))
    ops = [
        (
            op.model_copy(update={"residual_policy": "cancel"})
            if isinstance(op, JoinRegion)
            else op
        )
        for op in bundle.template.operators
    ]
    template = bundle.template.model_copy(update={"operators": tuple(ops)})
    eng = _engine(bundle.model_copy(update={"template": template}))
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("worker", "c0")).ready[0]
    eng.on_dispatched(child, "w1")

    eng.on_failed("A", "boom", retryable=False)

    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "revoked"
    assert _work_item(eng, child).status is WorkItemStatus.CANCELLED
    assert "join_released" not in {kind for kind, _ in eng.contract_trace()}


def test_a_failed_agent_instance_fails_only_its_own_region_scope() -> None:
    bundle = _recursive_agent_bundle()
    eng = _engine(bundle, budget=ScopeBudget(max_scope_depth=8))
    _dispatch_agent(eng)
    failing = _spawn_in(eng, "A", "c0", "worker")
    sibling = _spawn_in(eng, "A", "c1", "worker")
    for instance in (failing, sibling):
        eng.on_dispatched(instance, "w1")
    grandchild = _spawn_in(eng, failing, "c0", "self")
    eng.on_dispatched(grandchild, "w1")
    failing_act = _work_item(eng, failing).activation_id

    eng.on_failed(failing, "boom", retryable=False)
    eng.on_succeeded(grandchild)

    failed_scope = eng.region_scope_for(failing_act, "self")
    snapshot = eng.to_snapshot()
    assert snapshot.failed_scopes == [failed_scope]
    assert failed_scope not in snapshot.released_scopes
    assert not snapshot.failed_regions
    # A sibling instance of the same template still spawns and releases its own region.
    nephew = _spawn_in(eng, sibling, "c0", "self")
    eng.on_dispatched(nephew, "w1")
    eng.on_succeeded(nephew)
    eng.on_succeeded(sibling)
    sibling_act = _work_item(eng, sibling).activation_id
    assert eng.region_scope_for(sibling_act, "self") in (
        eng.to_snapshot().released_scopes
    )
    restored = OrchestrationEngine(eng.to_snapshot(), bundle)
    assert restored.to_snapshot().failed_scopes == [failed_scope]


def _self_recursive_agent() -> PersistedV2Workflow:
    """Agent A spawns instances of itself; ``after`` consumes A's region."""
    ref, ops, edge = _region("self", "A")
    return _bundle(
        [_agent("A", regions=(ref,)), *ops, _leaf("after")],
        [edge, TemplateEdge(from_op="self:spawn:join", to_op="after")],
        (_decl("out:A", "A"), _decl("out:after", "after")),
    )


def _nested_level(eng: OrchestrationEngine) -> tuple[str, str]:
    """A's instance I1 spawns a grandchild G into I1's own scope of the region."""
    _dispatch_agent(eng)
    i1 = _spawn_in(eng, "A", "c0", "self")
    eng.on_dispatched(i1, "w1")
    grandchild = _spawn_in(eng, i1, "c0", "self")
    eng.on_dispatched(grandchild, "w1")
    return i1, grandchild


def test_a_failed_agent_leaves_its_released_region_released() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    i1, _ = _nested_level(eng)
    eng.on_succeeded(i1)
    seal = BoundaryEvent(
        kind=BoundaryEventKind.SPAWN_SEAL,
        call_correlation="s0",
        child_region_ref="self",
    )
    assert eng.route_boundary_event("A", seal).ready == ["after"]

    assert eng.on_failed("A", "boom", retryable=False).failed == ["A"]

    assert _work_item(eng, "after").status is WorkItemStatus.READY
    assert not eng.to_snapshot().failed_regions
    assert eng.output_publication("out:after") is None


def test_a_failed_agent_fails_its_own_region_while_a_nested_level_released() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    i1, grandchild = _nested_level(eng)
    eng.on_succeeded(grandchild)
    # I1's scope of the shared region releases; A's own scope stays open.
    eng.on_succeeded(i1)

    failed = eng.on_failed("A", "boom", retryable=False).failed

    assert failed == ["A", "after"]
    assert "self:spawn:join" in eng.to_snapshot().failed_regions


_SELF_SEAL = BoundaryEvent(
    kind=BoundaryEventKind.SPAWN_SEAL, call_correlation="s0", child_region_ref="self"
)


def test_a_nested_levels_release_delivers_nothing_downstream() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    _dispatch_agent(eng)
    instance = _spawn_in(eng, "A", "c0", "self")
    eng.on_dispatched(instance, "w1")

    # The instance's own, never-entered region closes as it completes.
    assert eng.on_succeeded(instance).ready == []
    assert eng.output_publication("self:spawn:join") is None

    assert eng.route_boundary_event("A", _SELF_SEAL).ready == ["after"]


def test_a_failed_agent_fails_what_a_nested_level_released() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    _dispatch_agent(eng)
    instance = _spawn_in(eng, "A", "c0", "self")
    eng.on_dispatched(instance, "w1")
    assert eng.on_succeeded(instance).ready == []

    assert eng.on_failed("A", "boom", retryable=False).failed == ["A", "after"]


def test_a_nested_level_closes_after_its_join_failed_at_the_root() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    i1, grandchild = _nested_level(eng)
    eng.on_failed("A", "boom", retryable=False)

    eng.on_succeeded(grandchild)
    eng.on_succeeded(i1)

    nested = eng.region_scope_for(_work_item(eng, i1).activation_id, "self")
    assert nested in eng.to_snapshot().released_scopes


def test_a_cancelled_instances_region_delivers_nothing_downstream() -> None:
    eng = _engine(_self_recursive_agent(), budget=ScopeBudget(max_scope_depth=8))
    join = eng._operators["self:spawn:join"]
    assert isinstance(join, JoinRegion)
    eng._operators["self:spawn:join"] = join.model_copy(
        update={"residual_policy": "cancel"}
    )
    i1, grandchild = _nested_level(eng)

    # A completes while I1 and I1's own child run: the cancel reaches both, and only
    # A's level of the shared region releases downstream.
    advance = eng.on_succeeded("A")

    assert advance.cancelled == [i1, grandchild]
    assert advance.ready == ["after"]
    nested = eng.region_scope_for(_work_item(eng, i1).activation_id, "self")
    assert nested in eng.to_snapshot().released_scopes


def test_the_root_level_aggregate_survives_a_later_nested_release() -> None:
    bundle = _self_recursive_agent()
    eng = _engine(bundle, budget=ScopeBudget(max_scope_depth=8))
    i1, grandchild = _nested_level(eng)
    # I1 completes while its own child runs, so the root level releases first.
    eng.on_succeeded(i1)
    assert eng.route_boundary_event("A", _SELF_SEAL).ready == ["after"]
    root_members = [
        member.child_activation_id
        for member in eng._aggregate_by_join["self:spawn:join"].members
    ]
    assert root_members == [_work_item(eng, i1).activation_id]

    eng.on_succeeded(grandchild)

    snapshot = eng.to_snapshot()
    assert len(snapshot.region_aggregates) == 1
    # A ledger stored with a nested aggregate after the root's restores the root's.
    nested = snapshot.region_aggregates[0].model_copy(
        update={
            "members": tuple(
                member.model_copy(
                    update={
                        "child_activation_id": _work_item(eng, grandchild).activation_id
                    }
                )
                for member in snapshot.region_aggregates[0].members
            )
        }
    )
    stored = snapshot.model_copy(
        update={"region_aggregates": [*snapshot.region_aggregates, nested]}
    )
    restored = OrchestrationEngine(stored, bundle)
    assert [
        member.child_activation_id
        for member in restored._aggregate_by_join["self:spawn:join"].members
    ] == root_members


def _work_item(eng: OrchestrationEngine, task: str) -> WorkItem:
    wi = eng.work_item(task)
    assert wi is not None
    return wi


def _spawn_in(eng: OrchestrationEngine, task: str, call: str, role: str) -> str:
    return eng.route_boundary_event(
        task,
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation=call, child_region_ref=role
        ),
    ).ready[0]


def _numbered_by_scope_order(eng: OrchestrationEngine) -> bool:
    """Each child's index counts every activation its scope held before it."""
    seen: dict[str, int] = {}
    for act in eng.to_snapshot().activations:
        if act.kind == "child" and act.child_index != seen.get(act.scope_id, 0):
            return False
        seen[act.scope_id] = seen.get(act.scope_id, 0) + 1
    return True


def test_a_child_index_counts_every_activation_in_its_scope_across_a_restart() -> None:
    eng = _engine(_recursive_agent_bundle(), budget=ScopeBudget(max_scope_depth=8))
    _dispatch_agent(eng)
    lvl1 = _spawn_in(eng, "A", "c0", "worker")
    eng.on_dispatched(lvl1, "w1")
    # The nested spawn mints lvl1's region opener inside lvl1's own scope.
    _spawn_in(eng, lvl1, "c0", "self")
    second = _spawn_in(eng, "A", "c1", "worker")

    activations = {a.activation_id: a for a in eng.to_snapshot().activations}
    assert activations[second].child_index == 2
    assert _numbered_by_scope_order(eng)

    restored = OrchestrationEngine(eng.to_snapshot(), eng._bundle)
    third = _spawn_in(restored, "A", "c2", "worker")
    activations = {a.activation_id: a for a in restored.to_snapshot().activations}
    assert activations[third].child_index == 3
    assert _numbered_by_scope_order(restored)


def test_the_activation_budget_holds_across_a_restart() -> None:
    eng = _engine(
        _spawning_agent(child=_leaf("child")), budget=ScopeBudget(max_activations=2)
    )
    _dispatch_agent(eng)
    _spawn_in(eng, "A", "c0", "worker")
    restored = OrchestrationEngine(
        eng.to_snapshot(), eng._bundle, budget=ScopeBudget(max_activations=2)
    )
    _spawn_in(restored, "A", "c1", "worker")
    with pytest.raises(RegionError):
        _spawn_in(restored, "A", "c2", "worker")


def test_spawning_a_child_never_rescans_every_activation() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    activations = eng._activations
    scans = 0

    def count(_frame: Any, event: str, arg: Any) -> None:
        nonlocal scans
        if (
            event == "c_call"
            and getattr(arg, "__self__", None) is activations
            and arg.__name__ == "values"
        ):
            scans += 1

    sys.setprofile(count)
    try:
        for i in range(8):
            _spawn_in(eng, "A", f"c{i}", "worker")
    finally:
        sys.setprofile(None)
    assert scans == 0


def test_a_restart_restores_only_the_spawn_site_denials() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="x"
        ),
    )
    eng.deny_spawn("worker:spawn", "x")
    live = set(eng._denied_spawns)

    restored = OrchestrationEngine(eng.to_snapshot(), eng._bundle)
    assert restored._denied_spawns == live == {"worker:spawn"}
