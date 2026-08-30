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

import pytest

from server.orchestration import (
    OrchestrationEngine,
    ProgressAxis,
    RegionError,
    ScopeBudget,
    WorkItemStatus,
)
from server.orchestration.state import BoundaryEvent, DenialKind, InvocationState
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
    pub = eng.resolve_output("out:A")
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
    pub = eng.resolve_output("out:A")
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


def test_terminal_failure_settles_an_open_region() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    child = eng.route_boundary_event("A", _spawn("worker", "c0")).ready[0]
    eng.on_dispatched(child, "w1")
    # The agent fails terminally while its region is open with an in-flight child.
    eng.on_failed("A", "boom", retryable=False)
    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "sealed" and not cap.closed
    # The child drains and the region's join then releases.
    eng.on_succeeded(child)
    cap = eng.capability(eng.region_scope_for(act, "worker"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.closed
    assert eng.region_closed("worker:spawn:join")
