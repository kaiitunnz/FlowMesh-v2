"""Deterministic trace tests for structured dynamic regions.

These drive the orchestration engine directly over hand-built transparent-region
bundles. They prove closure from combined child-init and loop-time capability
accounting, monotone authority attenuation, and the invocation FSM extension, plus
small autoresearch-like and RLVR-like controllers over generic regions.
"""

import pytest

from server.orchestration import (
    CapabilityStatus,
    OrchestrationEngine,
    ProgressAxis,
    PublicationOutcome,
    RegionError,
    ScopeBudget,
)
from server.orchestration.state import ResultPublication, ValueRef
from server.task.v2 import FrontendWorkflowSource, PersistedV2Workflow
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    AuthorityCeiling,
    BindingKey,
    BranchRegion,
    DeterminismClass,
    EffectBoundary,
    EffectClass,
    EffectReplayContract,
    InputProvenanceKind,
    JoinCompletion,
    JoinPredicate,
    JoinRegion,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    LoopContextRegion,
    MergeRegion,
    ModelRef,
    OperatorKind,
    Port,
    RecoveryClass,
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


def _leaf(
    op_id: str, *, deps: bool = False, profile: LeafProfile | None = None
) -> LeafOperator:
    return LeafOperator(
        operator_id=op_id,
        source_ref=op_id,
        inputs=(Port(name="in"),) if deps else (),
        outputs=(Port(name="out"),),
        profile=profile or leaf_profile(TaskType.ECHO),
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
    *,
    results: tuple[ResultDeclaration, ...] = (),
    boundaries: tuple[EffectBoundary, ...] = (),
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
        effect_boundaries=boundaries,
        source_map=source_map,
    )
    plan = PhysicalExecutionPlan(plan_version=pv, template_version=tv, nodes=nodes)
    source = FrontendWorkflowSource.capture("regions: true", "native", name="wf")
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def _engine(
    bundle: PersistedV2Workflow,
    *,
    budget: ScopeBudget | None = None,
    granted_interfaces: frozenset[str] | None = None,
) -> OrchestrationEngine:
    return OrchestrationEngine.build(
        "wfl-x",
        "owner",
        "org",
        bundle,
        budget=budget,
        granted_interfaces=granted_interfaces,
    )


def _kinds(eng: OrchestrationEngine) -> set[str]:
    return {kind for kind, _ in eng.contract_trace()}


# --------------------------------------------------------------------------- #
# Progress capability: acquisition, seal, revoke, zero-child, late child
# --------------------------------------------------------------------------- #


def _spawn_join(
    *,
    completion: JoinCompletion = JoinCompletion.ALL_SETTLED,
    results: tuple[ResultDeclaration, ...] = (),
) -> PersistedV2Workflow:
    spawn = SpawnRegion(operator_id="S", source_ref="S", child_template_ref="body")
    join = JoinRegion(operator_id="J", source_ref="J", completion=completion)
    body = _leaf("body")
    return _bundle(
        [spawn, join, body],
        [TemplateEdge(from_op="S", to_op="J")],
        results=results
        or (_decl("out:J", "J", release=ReleaseConditionKind.SCOPE_CLOSED),),
    )


def test_child_init_capability_acquired_on_spawn_open() -> None:
    eng = _engine(_spawn_join())
    scope = eng.scope_for("S")
    assert scope is not None
    cap = eng.capability(scope, ProgressAxis.CHILD_INIT)
    assert (
        cap is not None and cap.status is CapabilityStatus.OPEN and cap.outstanding == 0
    )
    assert "child_init_acquired" in _kinds(eng)


def test_zero_child_spawn_closes_only_after_seal() -> None:
    eng = _engine(_spawn_join())
    # An empty child set is never closure by itself; the join stays open until the seal.
    assert not eng.region_closed("J")
    eng.seal_spawn("S")
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY
    assert {"child_init_sealed", "join_released", "frontier_closed"} <= _kinds(eng)


def test_late_child_prevented_after_seal() -> None:
    eng = _engine(_spawn_join())
    eng.seal_spawn("S")
    with pytest.raises(RegionError):
        eng.spawn_child("S")


def test_child_init_revoke_drains_then_closes() -> None:
    eng = _engine(_spawn_join())
    child = eng.spawn_child("S")
    scope = eng.scope_for("S")
    assert scope is not None
    eng.revoke_spawn("S")
    cap = eng.capability(scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.REVOKED
    assert cap.outstanding == 1 and not cap.closed  # a materialized child still drains
    eng.settle_child(child)
    assert cap.closed
    assert "child_init_revoked" in _kinds(eng)


def test_all_settled_join_releases_success_over_children() -> None:
    eng = _engine(_spawn_join(completion=JoinCompletion.ALL_SETTLED))
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a)
    assert not eng.region_closed("J")  # child-init not sealed yet
    eng.settle_child(b)
    assert not eng.region_closed("J")
    eng.seal_spawn("S")
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.SUCCESS


def test_all_succeed_join_fails_on_a_child_failure() -> None:
    eng = _engine(_spawn_join(completion=JoinCompletion.ALL_SUCCEED))
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a, outcome=PublicationOutcome.SUCCESS)
    eng.settle_child(b, outcome=PublicationOutcome.DECLARED_FAILURE)
    eng.seal_spawn("S")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE


def test_join_waits_for_seal_not_observed_empty() -> None:
    eng = _engine(_spawn_join())
    a = eng.spawn_child("S")
    eng.settle_child(a)
    # Every known child has settled, but the producer has not sealed: not closed.
    assert not eng.region_closed("J")
    eng.spawn_child("S")  # a late child is still legal before the seal
    eng.seal_spawn("S")
    assert not eng.region_closed("J")  # the late child is still outstanding
    outstanding = eng.capability(eng.scope_for("S"), ProgressAxis.CHILD_INIT)
    assert outstanding is not None and outstanding.outstanding == 1


# --------------------------------------------------------------------------- #
# Nested call and scope lineage
# --------------------------------------------------------------------------- #


def test_nested_call_closes_inner_then_outer() -> None:
    outer_s = SpawnRegion(operator_id="So", source_ref="So", child_template_ref="Si")
    outer_j = JoinRegion(
        operator_id="Jo", source_ref="Jo", completion=JoinCompletion.ALL_SUCCEED
    )
    inner_s = SpawnRegion(operator_id="Si", source_ref="Si", child_template_ref="leaf")
    inner_j = JoinRegion(
        operator_id="Ji", source_ref="Ji", completion=JoinCompletion.ALL_SUCCEED
    )
    eng = _engine(
        _bundle(
            [outer_s, outer_j, inner_s, inner_j, _leaf("leaf")],
            [
                TemplateEdge(from_op="So", to_op="Jo"),
                TemplateEdge(from_op="Si", to_op="Ji"),
            ],
            results=(_decl("out:Jo", "Jo", release=ReleaseConditionKind.SCOPE_CLOSED),),
        )
    )
    outer_child = eng.spawn_child("So", operator_id="Si")  # opens the nested Si scope
    inner_child = eng.spawn_child("Si")
    eng.settle_child(inner_child)
    eng.seal_spawn("Si")
    assert eng.region_closed("Ji")  # inner call closes first
    assert not eng.region_closed("Jo")
    eng.settle_child(outer_child)
    eng.seal_spawn("So")
    assert eng.region_closed("Jo")

    # The inner scope is a child of the outer scope, one level deeper.
    outer_scope = eng.scope_for("So")
    inner_scope = eng.scope_for("Si")
    assert inner_scope is not None and outer_scope is not None
    assert eng.grant_for("Si") is not None


# --------------------------------------------------------------------------- #
# Authority attenuation (second generation)
# --------------------------------------------------------------------------- #


def test_second_generation_attenuation_cannot_redelegate() -> None:
    # root delegates {x, y}; spawn A may invoke {x, y} but only delegate {x}; spawn B,
    # a child region of A, declares a wide ceiling but is bounded by A's delegate face.
    spawn_a = SpawnRegion(
        operator_id="A",
        source_ref="A",
        child_template_ref="B",
        authority=AuthorityCeiling(invoke=("x", "y"), delegate=("x",)),
    )
    spawn_b = SpawnRegion(
        operator_id="B",
        source_ref="B",
        child_template_ref="leaf",
        authority=AuthorityCeiling(invoke=("x", "y"), delegate=("x", "y")),
    )
    eng = _engine(
        _bundle([spawn_a, spawn_b, _leaf("leaf")], []),
        granted_interfaces=frozenset({"x", "y"}),
    )
    eng.spawn_child("A", operator_id="B")  # opens B's nested scope, minting its grant

    grant_a = eng.grant_for("A")
    grant_b = eng.grant_for("B")
    assert grant_a is not None and grant_b is not None
    assert grant_a.invoke == ("x", "y") and grant_a.delegate == ("x",)
    # B is bounded by A's delegate face: it cannot delegate y though the root could.
    assert grant_b.invoke == ("x",) and grant_b.delegate == ("x",)
    assert eng.can_delegate("A", "x") and not eng.can_delegate("A", "y")
    assert eng.can_delegate("B", "x") and not eng.can_delegate("B", "y")
    assert "grant_delegated" in _kinds(eng)


def test_denied_spawn_creates_no_child_and_seal_stays_separate() -> None:
    eng = _engine(_spawn_join())
    eng.deny_spawn("S", "x")
    with pytest.raises(RegionError):
        eng.spawn_child("S")
    # Denial does not seal the child-init capability: grant and cardinality stay apart.
    cap = eng.capability(eng.scope_for("S"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.OPEN
    assert "authority_denied" in _kinds(eng)


# --------------------------------------------------------------------------- #
# LoopContext: well-founded time, loop-carried ModelRef, delayed completion
# --------------------------------------------------------------------------- #


def _loop_bundle() -> PersistedV2Workflow:
    loop = LoopContextRegion(operator_id="L", source_ref="L", loop_coordinate="t")
    return _bundle(
        [loop],
        [],
        results=(_decl("out:L", "L", release=ReleaseConditionKind.SCOPE_CLOSED),),
    )


def test_loop_time_is_well_founded_and_bounded() -> None:
    eng = _engine(_loop_bundle(), budget=ScopeBudget(max_loop_iterations=2))
    t1 = eng.loop_feedback("L")
    t2 = eng.loop_feedback("L")
    assert eng._activations[t1].loop_time == 1  # strictly increasing
    assert eng._activations[t2].loop_time == 2
    with pytest.raises(RegionError):
        eng.loop_feedback("L")  # exceeds the iteration budget


def test_loop_closes_under_delayed_completion_with_carried_model_ref() -> None:
    eng = _engine(_loop_bundle())
    rounds = []
    for version in ("v1", "v2", "v3"):
        rounds.append(
            eng.loop_feedback(
                "L",
                value_ref=ValueRef(
                    kind="model_ref",
                    model_ref=ModelRef(architecture="m", version=version),
                ),
            )
        )
    # Iterations settle out of order (a later rollout finishes before an earlier one).
    eng.settle_iteration(rounds[1])
    eng.loop_seal("L")
    assert not eng.region_closed("L")  # rounds 0 and 2 still outstanding
    eng.settle_iteration(rounds[0])
    assert not eng.region_closed("L")
    eng.settle_iteration(rounds[2])
    assert eng.region_closed("L")
    # Egress carries the latest loop-time ModelRef version.
    pub = eng.resolve_output("out:L")
    assert pub is not None and pub.value_ref is not None
    assert pub.value_ref.model_ref is not None
    assert pub.value_ref.model_ref.version == "v3"


# --------------------------------------------------------------------------- #
# Branch / merge record routing
# --------------------------------------------------------------------------- #


def test_branch_routes_selected_port_and_empties_the_rest() -> None:
    branch = BranchRegion(operator_id="B", source_ref="B", selection=None)
    eng = _engine(
        _bundle(
            [_leaf("A"), branch, _leaf("x", deps=True), _leaf("y", deps=True)],
            [
                TemplateEdge(from_op="A", to_op="B"),
                TemplateEdge(from_op="B", to_op="x", from_port="p1"),
                TemplateEdge(from_op="B", to_op="y", from_port="p2"),
            ],
            results=(_decl("legacy:y", "y"),),
        )
    )
    eng.on_succeeded("A")  # fires the branch; awaits a data-driven route
    adv = eng.route_branch("B", "p1")
    assert adv.ready == ["x"]  # the selected port readies its successor
    assert eng.work_item("x").status.value == "ready"  # type: ignore[union-attr]
    ypub = eng.resolve_output("legacy:y")
    assert ypub is not None and ypub.outcome is PublicationOutcome.EXPLICIT_EMPTY


def test_merge_combines_all_inputs_before_releasing() -> None:
    merge = MergeRegion(operator_id="M", source_ref="M")
    eng = _engine(
        _bundle(
            [_leaf("A"), _leaf("B"), merge, _leaf("C", deps=True)],
            [
                TemplateEdge(from_op="A", to_op="M"),
                TemplateEdge(from_op="B", to_op="M"),
                TemplateEdge(from_op="M", to_op="C"),
            ],
        )
    )
    eng.on_succeeded("A")
    assert eng.work_item("C").status.value == "blocked"  # type: ignore[union-attr]
    adv = eng.on_succeeded("B")  # merge fires only when both inputs arrive
    assert adv.ready == ["C"]
    assert "merge_combined" in _kinds(eng)


# --------------------------------------------------------------------------- #
# Invocation FSM extension for structured regions
# --------------------------------------------------------------------------- #


def _external_leaf(
    op_id: str, contract: EffectReplayContract
) -> tuple[LeafOperator, EffectBoundary]:
    profile = LeafProfile(
        determinism=DeterminismClass.SAMPLED,
        effect=EffectClass.EXTERNAL_EFFECT,
        recovery=RecoveryClass.RECORD,
        input_provenance=InputProvenanceKind.LIVE_INPUT,
        binding=BindingKey(task_type=TaskType.API),
    )
    boundary = EffectBoundary(
        effect_class=EffectClass.EXTERNAL_EFFECT,
        replay_contract=contract,
        source_ref=op_id,
    )
    return _leaf(op_id, profile=profile), boundary


def test_uncertain_compensable_effect_is_compensation_required() -> None:
    leaf, boundary = _external_leaf("eff", EffectReplayContract.COMPENSABLE)
    eng = _engine(
        _bundle(
            [leaf], [], results=(_decl("legacy:eff", "eff"),), boundaries=(boundary,)
        )
    )
    eng.on_dispatched("eff", None)
    eng.on_started("eff")
    adv = eng.on_uncertain("eff")
    # A compensable effect left uncertain never infers success or silently retries.
    assert adv.failed == ["eff"] and adv.retry == []
    inv = eng.invocation_for_task("eff")
    assert inv is not None and inv.state.value == "compensation_required"
    pub = eng.resolve_legacy_task("eff")
    assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE


def test_replayable_effect_reissues_same_invocation_and_records_one_receipt() -> None:
    leaf, boundary = _external_leaf("eff", EffectReplayContract.REPLAYABLE_DEDUP)
    eng = _engine(
        _bundle(
            [leaf], [], results=(_decl("legacy:eff", "eff"),), boundaries=(boundary,)
        )
    )
    eng.on_dispatched("eff", None)
    invocation = eng.invocation_for_task("eff")
    assert invocation is not None
    invocation_id = invocation.invocation_id
    eng.on_started("eff")
    adv = eng.on_uncertain("eff")  # a replayable effect reissues, never infers success
    assert adv.retry == ["eff"] and adv.failed == []
    eng.on_dispatched("eff", None)  # reissue as a fresh attempt under the same identity
    assert eng.invocation_for_task("eff").invocation_id == invocation_id  # type: ignore[union-attr]
    eng.on_succeeded("eff")
    snap = eng.to_snapshot()
    wi = eng.work_item("eff")
    assert wi is not None
    receipts = [r for r in snap.effect_receipts if r.work_item_id == wi.work_item_id]
    assert len(receipts) == 1  # the recorded outcome is preserved exactly once
    pub = eng.resolve_legacy_task("eff")
    assert pub is not None and pub.outcome is PublicationOutcome.SUCCESS


def test_uncertain_ambiguity_terminal_effect_never_succeeds() -> None:
    leaf, boundary = _external_leaf("eff", EffectReplayContract.AMBIGUITY_TERMINAL)
    eng = _engine(_bundle([leaf], [], boundaries=(boundary,)))
    eng.on_dispatched("eff", None)
    adv = eng.on_uncertain("eff")
    assert adv.failed == ["eff"] and adv.retry == []
    inv = eng.invocation_for_task("eff")
    assert inv is not None and inv.state.value == "ambiguity_terminal"


# --------------------------------------------------------------------------- #
# Workload controllers over generic regions
# --------------------------------------------------------------------------- #


def test_autoresearch_controller_fans_out_experiments() -> None:
    # A planner leaf feeds a spawn/join region; a controller creates one experiment
    # child per hypothesis through the generic region, with no special expansion API.
    planner = _leaf("planner")
    spawn = SpawnRegion(operator_id="exp", source_ref="exp", child_template_ref="trial")
    join = JoinRegion(
        operator_id="collect",
        source_ref="collect",
        completion=JoinCompletion.ALL_SETTLED,
    )
    eng = _engine(
        _bundle(
            [planner, spawn, join, _leaf("trial")],
            [
                TemplateEdge(from_op="planner", to_op="exp"),
                TemplateEdge(from_op="exp", to_op="collect"),
            ],
            results=(
                _decl(
                    "results",
                    "exp",
                    cardinality=CardinalityKind.KEYED_COLLECTION,
                    keying="hypothesis",
                ),
                _decl("summary", "collect", release=ReleaseConditionKind.SCOPE_CLOSED),
            ),
        )
    )
    eng.on_succeeded("planner")  # planner settles, opening the experiment region
    children = [eng.spawn_child("exp") for _ in range(3)]
    for child in children:
        eng.settle_child(child)
    eng.seal_spawn("exp")
    assert eng.region_closed("collect")
    summary = eng.resolve_output("summary")
    assert summary is not None and summary.outcome is PublicationOutcome.SUCCESS
    # Each experiment publishes into the keyed collection under its own logical key.
    keyed = [p for k, p in _collection_publications(eng, "results")]
    assert len(keyed) == 3


def test_rlvr_loop_pins_model_version_per_round() -> None:
    # A repeated rollout/evaluation loop carries an updated ModelRef each round and
    # closes only after every delayed rollout completes.
    eng = _engine(_loop_bundle())
    versions = ["p1", "p2", "p3", "p4"]
    rounds = [
        eng.loop_feedback(
            "L",
            value_ref=ValueRef(
                kind="model_ref", model_ref=ModelRef(architecture="policy", version=v)
            ),
        )
        for v in versions
    ]
    # Evaluations return in a scrambled order.
    for idx in (2, 0, 3, 1):
        assert not eng.region_closed("L")
        eng.settle_iteration(rounds[idx])
    eng.loop_seal("L")
    assert eng.region_closed("L")
    pub = eng.resolve_output("out:L")
    assert pub is not None and pub.value_ref is not None
    assert (
        pub.value_ref.model_ref is not None and pub.value_ref.model_ref.version == "p4"
    )


# --------------------------------------------------------------------------- #
# Guardrails and skipped subtrees
# --------------------------------------------------------------------------- #


def test_activation_budget_breach_is_scope_budget_exhausted() -> None:
    eng = _engine(_spawn_join(), budget=ScopeBudget(max_activations=1))
    eng.spawn_child("S")
    with pytest.raises(RegionError):
        eng.spawn_child("S")  # exceeds the activation budget
    # A budget breach is a durable terminal distinct from an authority denial.
    assert "scope_budget_exhausted" in _kinds(eng)
    assert "authority_denied" not in _kinds(eng)


def test_depth_budget_breach_leaves_no_half_open_child() -> None:
    # A nested opener child would breach the depth cap; the child must not materialize.
    outer = SpawnRegion(operator_id="So", source_ref="So", child_template_ref="Si")
    inner = SpawnRegion(operator_id="Si", source_ref="Si", child_template_ref="leaf")
    eng = _engine(
        _bundle(
            [outer, inner, _leaf("leaf")],
            [],
        ),
        budget=ScopeBudget(max_scope_depth=1),
    )
    before = len(eng.to_snapshot().activations)
    with pytest.raises(RegionError):
        eng.spawn_child("So", operator_id="Si")  # opening Si's scope would be depth 2
    cap = eng.capability(eng.scope_for("So"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.outstanding == 0  # nothing half-materialized
    assert len(eng.to_snapshot().activations) == before


def test_branch_skips_a_non_selected_control_subtree() -> None:
    branch = BranchRegion(operator_id="B", source_ref="B", selection=None)
    spawn = SpawnRegion(operator_id="Sk", source_ref="Sk", child_template_ref="body")
    eng = _engine(
        _bundle(
            [_leaf("A"), branch, _leaf("x", deps=True), spawn, _leaf("body")],
            [
                TemplateEdge(from_op="A", to_op="B"),
                TemplateEdge(from_op="B", to_op="x", from_port="p1"),
                TemplateEdge(from_op="B", to_op="Sk", from_port="p2"),
            ],
        )
    )
    eng.on_succeeded("A")
    eng.route_branch("B", "p1")
    # The non-selected spawn subtree is skipped, not left waiting on the branch.
    assert "region_skipped" in _kinds(eng)
    assert eng.work_item("x").status.value == "ready"  # type: ignore[union-attr]


def test_branch_skips_a_shared_downstream_op_once() -> None:
    branch = BranchRegion(operator_id="B", source_ref="B", selection=None)
    spawn = SpawnRegion(operator_id="Z", source_ref="Z", child_template_ref="body")
    eng = _engine(
        _bundle(
            [
                _leaf("A"),
                branch,
                _leaf("x", deps=True),
                _leaf("m", deps=True),
                _leaf("n", deps=True),
                spawn,
                _leaf("body"),
            ],
            [
                TemplateEdge(from_op="A", to_op="B"),
                TemplateEdge(from_op="B", to_op="x", from_port="p1"),
                TemplateEdge(from_op="B", to_op="m", from_port="p2"),
                TemplateEdge(from_op="B", to_op="n", from_port="p3"),
                TemplateEdge(from_op="m", to_op="Z"),
                TemplateEdge(from_op="n", to_op="Z"),
            ],
        )
    )
    eng.on_succeeded("A")
    eng.route_branch("B", "p1")
    skips = [op for kind, op in eng.contract_trace() if kind == "region_skipped"]
    assert skips.count("Z") == 1  # both non-selected ports reach Z; it skips once
    assert eng.work_item("x").status.value == "ready"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Rehydration of a mid-flight dynamic region
# --------------------------------------------------------------------------- #


def test_mid_loop_snapshot_rehydrates_and_egresses() -> None:
    bundle = _loop_bundle()
    eng = _engine(bundle)
    first = eng.loop_feedback(
        "L",
        value_ref=ValueRef(
            kind="model_ref", model_ref=ModelRef(architecture="m", version="v1")
        ),
    )
    second = eng.loop_feedback(
        "L",
        value_ref=ValueRef(
            kind="model_ref", model_ref=ModelRef(architecture="m", version="v2")
        ),
    )
    eng.settle_iteration(first)
    # Rebuild from a snapshot taken after feedback but before egress: the per-iteration
    # feedback records must not be mistaken for an already-egressed loop.
    restored = OrchestrationEngine(eng.to_snapshot(), bundle)
    assert not restored.region_closed("L")
    restored.settle_iteration(second)
    restored.loop_seal("L")
    assert restored.region_closed("L")
    pub = restored.resolve_output("out:L")
    assert pub is not None and pub.value_ref is not None
    assert (
        pub.value_ref.model_ref is not None and pub.value_ref.model_ref.version == "v2"
    )


def test_denied_spawn_survives_rehydration() -> None:
    bundle = _spawn_join()
    eng = _engine(bundle)
    eng.deny_spawn("S", "x")
    # The denial is reconstructed from durable facts, so no child appears after restart.
    restored = OrchestrationEngine(eng.to_snapshot(), bundle)
    with pytest.raises(RegionError):
        restored.spawn_child("S")


def _collection_publications(
    eng: OrchestrationEngine, output_id: str
) -> list[tuple[str | None, ResultPublication]]:
    snap = eng.to_snapshot()
    return [
        (slot.logical_key, pub)
        for slot in snap.result_slots
        if slot.output_id == output_id
        for pub in snap.result_publications
        if pub.slot_key == slot.slot_key
    ]


# --------------------------------------------------------------------------- #
# Early-join completion (any / first-k / predicate)
# --------------------------------------------------------------------------- #


def _early_join(
    completion: JoinCompletion,
    *,
    first_k: int | None = None,
    predicate: JoinPredicate | None = None,
    residual: str = "continue",
    no_winner_failure: bool = False,
) -> PersistedV2Workflow:
    spawn = SpawnRegion(operator_id="S", source_ref="S", child_template_ref="body")
    join = JoinRegion(
        operator_id="J",
        source_ref="J",
        completion=completion,
        residual_policy=residual,
        first_k=first_k,
        predicate=predicate,
        no_winner_failure=no_winner_failure,
    )
    return _bundle(
        [spawn, join, _leaf("body")],
        [TemplateEdge(from_op="S", to_op="J")],
        results=(_decl("out:J", "J", release=ReleaseConditionKind.JOIN_WINNER),),
    )


def _value(tag: str) -> ValueRef:
    return ValueRef(kind="legacy_task_result", legacy_task_id=tag)


def _child_status(eng: OrchestrationEngine, activation_id: str) -> str:
    snap = eng.to_snapshot()
    wi = next(w for w in snap.work_items if w.activation_id == activation_id)
    return wi.status.value


def test_any_join_releases_on_first_qualifier_and_ignores_residual() -> None:
    eng = _engine(_early_join(JoinCompletion.ANY, residual="continue"))
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a, value_ref=_value("a"))
    # An early join releases before the producer seals, on the first qualifier.
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.SUCCESS
    assert pub.value_ref is not None and pub.value_ref.legacy_task_id == "a"
    # A residual child keeps settling into the ledger but never becomes the output.
    eng.settle_child(b, value_ref=_value("b"))
    again = eng.resolve_output("out:J")
    assert again is not None and again.value_ref is not None
    assert again.value_ref.legacy_task_id == "a"


def test_first_k_releases_at_threshold_with_lowest_index_winner() -> None:
    eng = _engine(_early_join(JoinCompletion.FIRST_K, first_k=2, residual="continue"))
    eng.spawn_child("S")  # index 0, never settles
    b = eng.spawn_child("S")  # index 1
    c = eng.spawn_child("S")  # index 2
    eng.settle_child(c, value_ref=_value("c"))
    assert not eng.region_closed("J")  # one qualifier, threshold is two
    eng.settle_child(b, value_ref=_value("b"))
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    # Winner is the lowest child_index among the qualifiers {b: 1, c: 2}.
    assert pub is not None and pub.value_ref is not None
    assert pub.value_ref.legacy_task_id == "b"


def test_predicate_monotone_releases_on_witness() -> None:
    eng = _engine(
        _early_join(
            JoinCompletion.PREDICATE,
            predicate=JoinPredicate(min_qualifiers=2, monotone=True),
            residual="drain",
        )
    )
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.spawn_child("S")
    eng.settle_child(a)
    assert not eng.region_closed("J")
    eng.settle_child(b)
    assert eng.region_closed("J")  # monotone witness at the threshold releases


def test_predicate_non_monotone_waits_for_frontier_closure() -> None:
    eng = _engine(
        _early_join(
            JoinCompletion.PREDICATE,
            predicate=JoinPredicate(min_qualifiers=2, monotone=False),
            residual="continue",
        )
    )
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a)
    eng.settle_child(b)
    # A non-monotone predicate never releases early even once its threshold is met.
    assert not eng.region_closed("J")
    eng.seal_spawn("S")
    assert eng.region_closed("J")  # evaluated once at frontier closure
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.SUCCESS


def test_early_join_no_winner_is_explicit_empty() -> None:
    eng = _engine(_early_join(JoinCompletion.ANY, residual="continue"))
    a = eng.spawn_child("S")
    eng.settle_child(a, outcome=PublicationOutcome.DECLARED_FAILURE)
    assert not eng.region_closed("J")  # a failed child is not a qualifier
    eng.seal_spawn("S")
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY


def test_early_join_no_winner_declared_failure_when_opted_in() -> None:
    eng = _engine(
        _early_join(JoinCompletion.ANY, residual="continue", no_winner_failure=True)
    )
    a = eng.spawn_child("S")
    eng.settle_child(a, outcome=PublicationOutcome.DECLARED_FAILURE)
    eng.seal_spawn("S")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE


def test_residual_continue_leaves_child_init_open() -> None:
    eng = _engine(_early_join(JoinCompletion.ANY, residual="continue"))
    a = eng.spawn_child("S")
    eng.settle_child(a, value_ref=_value("a"))
    cap = eng.capability(eng.scope_for("S"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.OPEN
    late = eng.spawn_child("S")  # a late child is still legal under continue
    eng.settle_child(late, value_ref=_value("late"))
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.value_ref is not None
    assert pub.value_ref.legacy_task_id == "a"  # residual never overtakes the winner


def test_residual_drain_seals_child_init() -> None:
    eng = _engine(_early_join(JoinCompletion.ANY, residual="drain"))
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a)
    cap = eng.capability(eng.scope_for("S"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.SEALED
    with pytest.raises(RegionError):
        eng.spawn_child("S")  # no new child after a drain seal
    eng.settle_child(b)  # a materialized residual still settles


def test_residual_cancel_revokes_and_cancels_residual_children() -> None:
    eng = _engine(_early_join(JoinCompletion.ANY, residual="cancel"))
    a = eng.spawn_child("S")
    b = eng.spawn_child("S")
    eng.settle_child(a)
    cap = eng.capability(eng.scope_for("S"), ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.REVOKED
    assert _child_status(eng, b) == "cancelled"  # residual child withdrawn
    assert _child_status(eng, a) == "settled"  # the qualifier stands


# --------------------------------------------------------------------------- #
# Recursive-region entry
# --------------------------------------------------------------------------- #


def _recursive_bundle() -> PersistedV2Workflow:
    spawn = SpawnRegion(operator_id="R", source_ref="R", child_template_ref="leaf")
    join = JoinRegion(
        operator_id="Rj", source_ref="Rj", completion=JoinCompletion.ALL_SETTLED
    )
    return _bundle(
        [spawn, join, _leaf("leaf")],
        [TemplateEdge(from_op="R", to_op="Rj")],
        results=(_decl("out:Rj", "Rj", release=ReleaseConditionKind.SCOPE_CLOSED),),
    )


def test_recursion_nests_scopes_at_increasing_depth() -> None:
    eng = _engine(_recursive_bundle(), budget=ScopeBudget(max_scope_depth=4))
    lvl2 = eng.spawn_child("R", operator_id="R")  # re-enters R one level down
    lvl3 = eng.spawn_child(lvl2, operator_id="R")
    # A recursive operator owns a scope per level, addressed by the entry activation.
    s2 = eng.scope_for(lvl2)
    s3 = eng.scope_for(lvl3)
    assert s2 is not None and s3 is not None
    snap = eng.to_snapshot()
    by_id = {s.scope_id: s for s in snap.scopes}
    s2_parent = by_id[s2].parent_scope_id
    assert s2_parent is not None
    # Each recursive entry nests one scope deeper over the finite template topology:
    # the top R scope is depth 1, then the two nested re-entries at depth 2 and 3.
    assert by_id[s2_parent].depth == 1
    assert by_id[s2].depth == 2
    assert by_id[s3].depth == 3
    assert by_id[s3].parent_scope_id == s2


def test_recursion_depth_limit_is_scope_budget_exhausted() -> None:
    eng = _engine(_recursive_bundle(), budget=ScopeBudget(max_scope_depth=3))
    lvl2 = eng.spawn_child("R", operator_id="R")
    lvl3 = eng.spawn_child(lvl2, operator_id="R")
    before = len(eng.to_snapshot().activations)
    with pytest.raises(RegionError):
        eng.spawn_child(lvl3, operator_id="R")  # a fourth level breaches the depth cap
    assert "scope_budget_exhausted" in _kinds(eng)
    assert "authority_denied" not in _kinds(eng)  # distinct from an authority denial
    assert len(eng.to_snapshot().activations) == before  # nothing half-materialized


def test_recursion_preserves_identity_across_rehydration() -> None:
    bundle = _recursive_bundle()
    eng = _engine(bundle, budget=ScopeBudget(max_scope_depth=5))
    lvl2 = eng.spawn_child("R", operator_id="R")
    lvl3 = eng.spawn_child(lvl2, operator_id="R")
    scope3 = eng.scope_for(lvl3)
    # A mid-recursion snapshot round-trips: recursive scopes and their opener
    # activations survive, and recursion continues from the restored deepest level.
    restored = OrchestrationEngine(
        eng.to_snapshot(), bundle, budget=ScopeBudget(max_scope_depth=5)
    )
    assert restored.scope_for(lvl3) == scope3
    lvl4 = restored.spawn_child(lvl3, operator_id="R")
    assert restored.scope_for(lvl4) is not None
    # Re-settling a materialized child is idempotent, never a second logical child.
    child = restored.spawn_child(lvl4, operator_id="leaf")
    restored.settle_child(child)
    restored.settle_child(child)
    matching = [
        a for a in restored.to_snapshot().activations if a.activation_id == child
    ]
    assert len(matching) == 1


def test_recursive_cross_level_join_release_survives_rehydration() -> None:
    bundle = _recursive_bundle()
    eng = _engine(bundle, budget=ScopeBudget(max_scope_depth=5))
    outer_scope = eng.scope_for("R")
    assert outer_scope is not None
    outer_owner = next(
        s.owner_activation_id
        for s in eng.to_snapshot().scopes
        if s.scope_id == outer_scope
    )
    assert outer_owner is not None
    # Re-enter R one level down and close the inner level's join.
    lvl2 = eng.spawn_child("R", operator_id="R")
    inner_scope = eng.scope_for(lvl2)
    assert inner_scope is not None
    inner_child = eng.spawn_child(lvl2, operator_id="leaf")
    eng.settle_child(inner_child)
    eng.seal_spawn(lvl2)
    # Close the outer level's join (lvl2 is the outer scope's child).
    eng.settle_child(lvl2)
    eng.seal_spawn(outer_owner)
    assert {outer_scope, inner_scope} <= set(eng.to_snapshot().released_scopes)
    releases = [k for k, _ in eng.contract_trace() if k == "join_released"]

    # Both levels of the one recursive join operator are restored directly, not
    # misattributed to the deepest scope, so neither released level is lost or re-run.
    restored = OrchestrationEngine(
        eng.to_snapshot(), bundle, budget=ScopeBudget(max_scope_depth=5)
    )
    assert {outer_scope, inner_scope} <= set(restored.to_snapshot().released_scopes)
    assert [k for k, _ in restored.contract_trace() if k == "join_released"] == releases


# --------------------------------------------------------------------------- #
# Cancellation as a durable semantic event
# --------------------------------------------------------------------------- #


def test_cancellation_revokes_child_init_distinct_from_seal() -> None:
    eng = _engine(_spawn_join())
    child = eng.spawn_child("S")
    scope = eng.scope_for("S")
    eng.on_cancelled(scope)  # type: ignore[arg-type]
    cap = eng.capability(scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status is CapabilityStatus.REVOKED
    kinds = _kinds(eng)
    assert "scope_cancelled" in kinds
    assert "child_init_revoked" in kinds
    assert "child_init_sealed" not in kinds  # revoke is never conflated with seal
    assert _child_status(eng, child) == "cancelled"  # default residual cancels children


def test_cancellation_revokes_grant_distinct_from_child_init_revoke() -> None:
    eng = _engine(_spawn_join())
    eng.spawn_child("S")
    grant_before = eng.grant_for("S")
    assert grant_before is not None and not grant_before.revoked
    eng.on_cancelled("S")
    grant_after = eng.grant_for("S")
    assert grant_after is not None and grant_after.revoked
    kinds = [k for k, _ in eng.contract_trace()]
    # Grant revocation and child-init revocation are separate, ordered transitions.
    assert "child_init_revoked" in kinds and "grant_revoked" in kinds
    assert kinds.index("child_init_revoked") < kinds.index("grant_revoked")


def test_cancellation_recorded_before_join_release() -> None:
    eng = _engine(_spawn_join())
    eng.spawn_child("S")
    eng.on_cancelled("S")
    kinds = [k for k, _ in eng.contract_trace()]
    assert kinds.index("scope_cancelled") < kinds.index("join_released")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY


def test_inner_scope_cancel_resolves_join_and_readies_downstream() -> None:
    spawn = SpawnRegion(operator_id="S", source_ref="S", child_template_ref="body")
    join = JoinRegion(
        operator_id="J", source_ref="J", completion=JoinCompletion.ALL_SETTLED
    )
    eng = _engine(
        _bundle(
            [spawn, join, _leaf("body"), _leaf("D", deps=True)],
            [
                TemplateEdge(from_op="S", to_op="J"),
                TemplateEdge(from_op="J", to_op="D"),
            ],
            results=(_decl("out:J", "J", release=ReleaseConditionKind.SCOPE_CLOSED),),
        )
    )
    eng.spawn_child("S")
    advance = eng.on_cancelled("S")
    assert eng.region_closed("J")
    pub = eng.resolve_output("out:J")
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY
    # The join's no-winner record readies the downstream leaf: the outer flow continues.
    assert advance.ready == ["D"]
    assert eng.work_item("D").status.value == "ready"  # type: ignore[union-attr]


def test_cancellation_residual_drain_lets_materialized_children_settle() -> None:
    spawn = SpawnRegion(operator_id="S", source_ref="S", child_template_ref="body")
    join = JoinRegion(
        operator_id="J",
        source_ref="J",
        completion=JoinCompletion.ANY,
        residual_policy="drain",
    )
    eng = _engine(
        _bundle(
            [spawn, join, _leaf("body")],
            [TemplateEdge(from_op="S", to_op="J")],
            results=(_decl("out:J", "J", release=ReleaseConditionKind.JOIN_WINNER),),
        )
    )
    child = eng.spawn_child("S")
    eng.on_cancelled("S")
    # A drain residual policy leaves a materialized child to settle, not cancelled.
    assert _child_status(eng, child) != "cancelled"
    eng.settle_child(child)
    assert _child_status(eng, child) == "settled"
