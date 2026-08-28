"""Engine-substrate tests for the agent-harness path, driven by a test-double adapter.

These prove a declared agent dispatches as a run-to-yield episode and suspends before a
mediated model or tool action, releasing its lane until its durable outcome is injected;
the durable boundary envelope persists the capsule, causal invocation id, and
fabric-assigned idempotency key atomically before the lane releases, and a forced
re-drive maps a reissued facade call to its recorded key; signature and authority checks
reject undeclared tools, model interfaces, and child targets through a durable typed
outcome; and a facade spawn_agent creates exactly one attenuated child with an explicit
seal, with recursive agent children reusing the finite declared region.
"""

from collections.abc import Sequence

from server.orchestration import (
    OrchestrationEngine,
    ProgressAxis,
    RegionError,
    ScopeBudget,
    WorkItemStatus,
)
from server.orchestration.harness import (
    AgentEpisode,
    DeliveredOutcome,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessConfigError,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
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
    LeafOperator,
    LogicalOperator,
    OperatorKind,
    Port,
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
    child_ref: str | None = None,
    invoke: tuple[str, ...] = ("model", "search"),
    delegate: tuple[str, ...] = ("model",),
) -> AgentOperator:
    return AgentOperator(
        operator_id=op_id,
        source_ref=op_id,
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=invoke, delegate=delegate),
        boundary=_SIGNATURE,
        child_template_ref=child_ref,
        outputs=(Port(name="out"),),
    )


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


def _engine(bundle: PersistedV2Workflow, *, budget: ScopeBudget | None = None):
    return OrchestrationEngine.build(
        "wfl-x", "owner", "org", bundle, granted_interfaces=_GRANTED, budget=budget
    )


def _solo_agent() -> PersistedV2Workflow:
    return _bundle([_agent("A")], [], (_decl("out:A", "A"),))


def _spawning_agent(*, child: LogicalOperator) -> PersistedV2Workflow:
    return _bundle(
        [_agent("A", child_ref=child.operator_id), child],
        [],
        (_decl("out:A", "A"),),
    )


# --------------------------------------------------------------------------- #
# Test-double adapter
# --------------------------------------------------------------------------- #


def _capsule(blob: str) -> HarnessCapsule:
    return HarnessCapsule(
        backend=HarnessBackendKey(backend="test-double", version="v1"), blob=blob
    )


class _DoubleAdapter(HarnessAdapter):
    """A scripted harness: each start returns the next result and records injections."""

    def __init__(self, steps: Sequence[HarnessResult]) -> None:
        self._steps = list(steps)
        self.cursor = 0
        self.injected: list[list[DeliveredOutcome]] = []
        self.cancelled: list[str] = []

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend="test-double", version="v1")

    def start(self, activation_id, *, capsule, outcomes):
        self.injected.append(list(outcomes))
        step = self._steps[self.cursor]
        self.cursor += 1
        return step

    def cancel(self, activation_id: str) -> None:
        self.cancelled.append(activation_id)


class _NativeBypassAdapter(_DoubleAdapter):
    def bypass_disabled(self) -> bool:
        return False


def _boundary(kind: BoundaryEventKind, call: str, **kw) -> HarnessResult:
    return HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryEvent(kind=kind, call_correlation=call, **kw),
        capsule=_capsule(f"after:{call}"),
    )


def _completion(value: str = "done") -> HarnessResult:
    return HarnessResult(kind=HarnessResultKind.COMPLETION, value=value)


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
    ep = AgentEpisode(
        eng,
        _DoubleAdapter(
            [_boundary(BoundaryEventKind.INVOCATION, "c0", interface="model")]
        ),
    )
    ep.resume("A", act)
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


def test_injected_outcome_resumes_the_episode_to_completion() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    adapter = _DoubleAdapter(
        [
            _boundary(BoundaryEventKind.INVOCATION, "c0", interface="model"),
            _completion(),
        ]
    )
    ep = AgentEpisode(eng, adapter)
    ep.resume("A", act)
    env = eng.boundary_envelope(act, "c0")
    assert env is not None
    # Safety: the suspend released the lane — the work item holds no worker (its attempt
    # finished) and is not occupying a ready/dispatched slot while it waits.
    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.BLOCKED
    attempt = eng._attempts[wi.attempt_ids[-1]]  # type: ignore[attr-defined]
    assert attempt.status.value == "succeeded" and attempt.finished_at is not None
    # Liveness gate: only the durable outcome re-readies the lane — the work item stays
    # suspended until the mediated outcome is delivered.
    resumed = eng.deliver_boundary_outcome("A", "c0")
    assert resumed.ready == ["A"]
    ep.deliver(
        DeliveredOutcome(
            call_correlation="c0", idempotency_key=env.idempotency_key, value="answer"
        )
    )
    eng.on_dispatched("A", "w1")
    result = ep.resume("A", act)
    assert result.kind is HarnessResultKind.COMPLETION
    # The resumed episode received the injected outcome at its originating call.
    assert adapter.injected[-1][0].value == "answer"
    eng.on_succeeded("A")
    # Observable end-to-end outcome: the episode settles terminally and the declared
    # output is readable, driven the whole way by the test-double adapter.
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


def test_undeclared_boundary_kind_is_denied() -> None:
    # An agent whose signature omits SPAWN cannot yield a spawn boundary.
    agent = _agent("A", child_ref="child")
    narrow = agent.model_copy(
        update={"boundary": BoundarySignature(events=(BoundaryEventKind.INVOCATION,))}
    )
    eng = _engine(_bundle([narrow, _leaf("child")], [], (_decl("out:A", "A"),)))
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.denial is DenialKind.AUTHORITY


def test_undeclared_child_target_is_denied() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="other"
        ),
    )
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.denial is DenialKind.AUTHORITY


def test_denial_is_delivered_back_through_the_mediation() -> None:
    eng = _engine(_solo_agent())
    act = _dispatch_agent(eng)
    adapter = _DoubleAdapter(
        [
            _boundary(BoundaryEventKind.INVOCATION, "c0", interface="danger"),
            _completion(),
        ]
    )
    ep = AgentEpisode(eng, adapter)
    ep.resume("A", act)
    eng.deliver_boundary_outcome("A", "c0")
    eng.on_dispatched("A", "w1")
    ep.resume("A", act)
    injected = adapter.injected[-1]
    assert injected and injected[0].kind is OutcomeKind.DENIED
    assert injected[0].denial is DenialKind.AUTHORITY


def test_mediation_refuses_a_native_bypass_adapter() -> None:
    eng = _engine(_solo_agent())
    try:
        AgentEpisode(eng, _NativeBypassAdapter([_completion()]))
    except HarnessConfigError:
        return
    raise AssertionError("expected a native-bypass adapter to be refused")


# --------------------------------------------------------------------------- #
# spawn_agent facade
# --------------------------------------------------------------------------- #


def test_spawn_agent_creates_one_attenuated_child_with_a_seal() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    adv = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    )
    # Exactly one declared child activation with a dispatchable identity.
    assert len(adv.ready) == 1 and adv.ready[0].startswith("act-")
    grant = eng.grant_for("A")
    assert grant is not None and grant.parent_grant_id == eng.instance.root_grant_id
    # The delegated grant is attenuated: it cannot widen the pinned envelope, and its
    # delegate face cannot widen its own invoke face.
    assert set(grant.invoke) <= _GRANTED
    assert set(grant.delegate) <= set(grant.invoke)
    # A spawn does not seal child-init; an explicit spawn seal closes the producer.

    cap = eng.capability(eng.scope_for("A"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "open"
    eng.route_boundary_event(
        "A", BoundaryEvent(kind=BoundaryEventKind.SPAWN_SEAL, call_correlation="c1")
    )
    cap = eng.capability(eng.scope_for("A"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "sealed"


def test_agent_terminal_completion_seals_its_child_init() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    child = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    ).ready[0]

    eng.on_dispatched(child, "w1")
    eng.on_succeeded(child)
    eng.on_succeeded("A")  # terminal completion supplies the seal
    cap = eng.capability(eng.scope_for("A"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.closed


def test_recursive_agent_child_reuses_the_declared_region() -> None:
    # An agent whose declared child region is another agent that recurses into itself:
    # each spawn re-enters the finite declared region, nesting scopes by depth.
    child = _agent("child", child_ref="child")
    eng = _engine(_spawning_agent(child=child), budget=ScopeBudget(max_scope_depth=8))
    _dispatch_agent(eng)
    lvl1 = eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    ).ready[0]
    # The materialized child agent is itself dispatchable and can spawn its own child.
    eng.on_dispatched(lvl1, "w1")
    lvl2 = eng.route_boundary_event(
        lvl1,
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
        ),
    ).ready[0]
    assert lvl2.startswith("act-") and lvl2 != lvl1
    # The template still holds exactly the two declared operators: recursion reused the
    # region rather than growing the topology.
    assert {op.operator_id for op in eng._bundle.template.operators} == {"A", "child"}  # type: ignore[attr-defined]
    # A owns a child-init scope (for lvl1); lvl1 owns a nested one (for lvl2), one
    # level deeper — recursion nests scopes over the finite topology.
    sa = eng.scope_for("A")
    s1 = eng.scope_for(lvl1)
    assert sa is not None and s1 is not None and sa != s1
    by_id = {s.scope_id: s for s in eng.to_snapshot().scopes}
    assert by_id[s1].depth == by_id[sa].depth + 1


def test_recursive_agent_depth_budget_is_enforced() -> None:
    child = _agent("child", child_ref="child")
    eng = _engine(_spawning_agent(child=child), budget=ScopeBudget(max_scope_depth=1))
    _dispatch_agent(eng)

    try:
        eng.route_boundary_event(
            "A",
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN, call_correlation="c0", child_ref="child"
            ),
        )
    except RegionError:
        assert "scope_budget_exhausted" in {k for k, _ in eng.contract_trace()}
        return
    raise AssertionError("expected the depth budget to reject the nested agent scope")
