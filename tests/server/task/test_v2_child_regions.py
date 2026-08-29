"""Compiler-level checks for the finite agent child-region contract.

These cover normalizing the legacy single child target into one declared region and the
static validation of the declared region set: reject-both, unique roles, resolvable
references, matched joins, and per-region authority within the agent's delegate face.
"""

from typing import Any

from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.compiler.project import LoweringAccumulator
from server.task.v2.compiler.regions import normalize_agent_child_regions
from server.task.v2.compiler.validation import validate_compilation
from server.task.v2.representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    ChildRegionRef,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    Port,
    SpawnRegion,
)
from server.task.v2.representations.plan import PhysicalExecutionPlan, PhysicalNode
from server.task.v2.representations.template import (
    LogicalWorkflowTemplate,
    SourceMapEntry,
    TemplateEdge,
)
from server.task.v2.representations.versioning import VersionId
from shared.tasks import TaskType


def _agent(op_id: str, **kw: Any) -> AgentOperator:
    return AgentOperator(
        operator_id=op_id,
        source_ref=op_id,
        binding=BindingKey(task_type=TaskType.AGENT),
        outputs=(Port(name="out"),),
        **kw,
    )


def _leaf(op_id: str) -> LeafOperator:
    return LeafOperator(
        operator_id=op_id,
        source_ref=op_id,
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )


def _region(
    role: str, entry: str, **kw: Any
) -> tuple[list[LogicalOperator], TemplateEdge]:
    spawn_id = f"{role}:spawn"
    join_id = f"{spawn_id}:join"
    spawn = SpawnRegion(
        operator_id=spawn_id,
        source_ref=spawn_id,
        outputs=(Port(name="children"),),
        child_template_ref=entry,
        **kw,
    )
    join = JoinRegion(
        operator_id=join_id,
        source_ref=join_id,
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ALL_SETTLED,
    )
    return [spawn, join], TemplateEdge(from_op=spawn_id, to_op=join_id)


def _diagnose(ops: list[LogicalOperator], edges: list[TemplateEdge]) -> set[str]:
    tv = VersionId(lineage="wfl:template", content_digest="td")
    pv = VersionId(lineage="wfl:plan", content_digest="pd")
    template = LogicalWorkflowTemplate(
        version=tv,
        operators=tuple(ops),
        edges=tuple(edges),
        source_map=tuple(
            SourceMapEntry(
                logical_ref=op.operator_id,
                source_kind="region",
                source_id=op.operator_id,
            )
            for op in ops
        ),
    )
    plan = PhysicalExecutionPlan(
        plan_version=pv,
        template_version=tv,
        nodes=tuple(
            PhysicalNode(
                node_id=f"phys:{op.operator_id}",
                source_ref=op.operator_id,
                logical_ref=op.operator_id,
            )
            for op in ops
        ),
    )
    return {d.code for d in validate_compilation(template, plan)}


# --------------------------------------------------------------------------- #
# Normalization of the legacy single child target
# --------------------------------------------------------------------------- #


def test_legacy_child_template_ref_normalizes_to_one_region() -> None:
    acc = LoweringAccumulator()
    acc.operators.extend(
        [
            _agent(
                "A",
                authority=AuthorityCeiling(invoke=("model",), delegate=("model",)),
                child_template_ref="child",
            ),
            _leaf("child"),
        ]
    )
    normalize_agent_child_regions(acc)

    agent = next(op for op in acc.operators if isinstance(op, AgentOperator))
    # Exactly one declared region replaces the legacy field.
    assert agent.child_template_ref is None
    assert len(agent.child_region_refs) == 1
    ref = agent.child_region_refs[0]
    assert ref.name == "child" and ref.spawn_ref == "A:child"
    # A matched Spawn/Join pair and their binding edge were synthesized.
    spawn = next(op for op in acc.operators if op.operator_id == "A:child")
    assert isinstance(spawn, SpawnRegion) and spawn.child_template_ref == "child"
    # The compat region inherits the agent's own ceiling as its per-site authority.
    assert spawn.authority.invoke == ("model",)
    assert any(op.operator_id == "A:child:join" for op in acc.operators)
    assert TemplateEdge(from_op="A:child", to_op="A:child:join") in acc.edges
    # The synthesized operators carry source-map and plan-node entries.
    assert {"A:child", "A:child:join"} <= {e.logical_ref for e in acc.source_map}
    assert {"A:child", "A:child:join"} <= {n.logical_ref for n in acc.nodes}


def test_normalization_skips_an_agent_that_already_declares_regions() -> None:
    acc = LoweringAccumulator()
    ref = ChildRegionRef(name="r", spawn_ref="r:spawn")
    acc.operators.append(
        _agent("A", child_template_ref="child", child_region_refs=(ref,))
    )
    before = len(acc.operators)
    normalize_agent_child_regions(acc)
    # A both-forms declaration is left intact for validation to reject.
    assert len(acc.operators) == before


# --------------------------------------------------------------------------- #
# Static validation of the declared region set
# --------------------------------------------------------------------------- #


def test_both_forms_declaration_is_rejected() -> None:
    ops, edge = _region("r", "child")
    agent = _agent(
        "A",
        child_template_ref="child",
        child_region_refs=(ChildRegionRef(name="r", spawn_ref="r:spawn"),),
    )
    codes = _diagnose([agent, _leaf("child"), *ops], [edge])
    assert "region.child-both-forms" in codes


def test_duplicate_role_is_rejected() -> None:
    ops, edge = _region("r", "child")
    refs = (
        ChildRegionRef(name="dup", spawn_ref="r:spawn"),
        ChildRegionRef(name="dup", spawn_ref="r:spawn"),
    )
    agent = _agent("A", child_region_refs=refs)
    codes = _diagnose([agent, _leaf("child"), *ops], [edge])
    assert "region.duplicate-role" in codes


def test_unresolved_region_ref_is_rejected() -> None:
    agent = _agent(
        "A", child_region_refs=(ChildRegionRef(name="r", spawn_ref="ghost"),)
    )
    codes = _diagnose([agent], [])
    assert "region.unresolved-region-ref" in codes


def test_region_without_a_matched_join_is_rejected() -> None:
    spawn = SpawnRegion(
        operator_id="r:spawn",
        source_ref="r:spawn",
        outputs=(Port(name="children"),),
        child_template_ref="child",
    )
    agent = _agent(
        "A", child_region_refs=(ChildRegionRef(name="r", spawn_ref="r:spawn"),)
    )
    codes = _diagnose([agent, _leaf("child"), spawn], [])
    assert "region.region-no-join" in codes


def test_region_ceiling_beyond_agent_delegate_is_rejected() -> None:
    ops, edge = _region(
        "r", "child", authority=AuthorityCeiling(invoke=("broad",), delegate=())
    )
    agent = _agent(
        "A",
        authority=AuthorityCeiling(invoke=("model",), delegate=("model",)),
        child_region_refs=(ChildRegionRef(name="r", spawn_ref="r:spawn"),),
    )
    codes = _diagnose([agent, _leaf("child"), *ops], [edge])
    assert "region.region-exceeds-delegate" in codes


def test_well_formed_region_set_passes() -> None:
    ops, edge = _region(
        "r", "child", authority=AuthorityCeiling(invoke=("model",), delegate=("model",))
    )
    agent = _agent(
        "A",
        authority=AuthorityCeiling(invoke=("model",), delegate=("model",)),
        child_region_refs=(ChildRegionRef(name="r", spawn_ref="r:spawn"),),
    )
    codes = _diagnose([agent, _leaf("child"), *ops], [edge])
    assert not {c for c in codes if c.startswith("region.")}


def test_legacy_agent_with_invoke_over_delegate_normalizes_and_compiles() -> None:
    # A legacy agent that can invoke more than it delegates (the case the compat exists
    # to preserve) must normalize to a region whose ceiling is the delegate face, so it
    # passes the region-within-delegate check rather than tripping its own validation.
    acc = LoweringAccumulator()
    acc.operators.extend(
        [
            _agent(
                "A",
                authority=AuthorityCeiling(
                    invoke=("model", "search"), delegate=("model",)
                ),
                child_template_ref="child",
            ),
            _leaf("child"),
        ]
    )
    normalize_agent_child_regions(acc)
    spawn = next(op for op in acc.operators if op.operator_id == "A:child")
    assert isinstance(spawn, SpawnRegion) and spawn.authority.invoke == ("model",)
    codes = _diagnose(acc.operators, acc.edges)
    assert not {c for c in codes if c.startswith("region.")}


def test_agent_selected_and_producer_fed_region_is_rejected() -> None:
    ops, edge = _region("r", "child")
    feed = TemplateEdge(from_op="producer", to_op="r:spawn")
    agent = _agent(
        "A", child_region_refs=(ChildRegionRef(name="r", spawn_ref="r:spawn"),)
    )
    codes = _diagnose([agent, _leaf("child"), _leaf("producer"), *ops], [edge, feed])
    assert "region.region-dual-entry" in codes
