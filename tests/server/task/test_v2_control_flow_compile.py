"""Compilation of branch, merge and loop regions and the region definitions they
enter: the finite structure, dependency classification and the refused shapes."""

from collections.abc import Iterator
from typing import Any

import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow
from server.task.v2.compiler import regions
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.inspect import build_inspection
from server.task.v2.representations.operators import (
    BranchRegion,
    LeafOperator,
    LoopContextRegion,
    MergeCombination,
    MergeRegion,
    SelectionCase,
    SpawnRegion,
)
from server.task.v2.representations.results import CardinalityKind
from server.task.v2.representations.template import (
    DefinitionKind,
    DependencyUse,
    EntryRole,
    LogicalWorkflowTemplate,
    ReturnKind,
)

_BINDINGS = AgentBindingDefaults(default_backend="codex")
_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


@pytest.fixture(autouse=True)
def _runnable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(regions, "CONTROL_FLOW_RUNNABLE", True)
    yield


def _compile(text: str) -> LogicalWorkflowTemplate:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl-test", parsed, source, bindings=_BINDINGS)
    return template


def _codes(text: str) -> list[str]:
    with pytest.raises(CompileError) as caught:
        _compile(text)
    return [diag.code for diag in caught.value.diagnostics]


def _workflow(nodes: str, templates: str = "") -> str:
    body = f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: cf}}
spec:
  graph:
{templates}
    nodes:
{nodes}
"""
    return body


def _op(template: LogicalWorkflowTemplate, source_id: str) -> Any:
    ids = {e.source_id: e.logical_ref for e in template.source_map}
    return next(op for op in template.operators if op.operator_id == ids[source_id])


_DIAMOND = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection:
            input: input
            field: [label]
            cases: {{accepted: left, rejected: right}}
      - name: left_work
        dependsOn: [{{node: decide, port: left, input: in}}]
        spec: {_ECHO}
      - name: right_work
        dependsOn: [{{node: decide, port: right}}]
        spec: {_ECHO}
      - name: merged
        dependsOn:
          - {{node: left_work, input: left}}
          - {{node: right_work, input: right}}
        region:
          kind: merge
          combination: one_live
          result: {{visibility: published}}
"""


def test_a_branch_diamond_compiles_to_ported_routes_and_a_one_live_merge() -> None:
    template = _compile(_workflow(_DIAMOND))
    decide = _op(template, "decide")
    assert isinstance(decide, BranchRegion)
    assert decide.rule is not None
    assert decide.rule.field == ("label",)
    assert decide.rule.cases == (
        SelectionCase(value="accepted", port="left"),
        SelectionCase(value="rejected", port="right"),
    )
    left, right = _op(template, "left_work"), _op(template, "right_work")
    into = {e.to_op: e for e in template.edges if e.from_op == decide.operator_id}
    assert into[left.operator_id].from_port == "left"
    assert into[left.operator_id].use is DependencyUse.VALUE_REQUIRED
    # An unread arm still gates its consumer as a route.
    assert into[right.operator_id].from_port == "right"
    assert into[right.operator_id].use is DependencyUse.ROUTE_REQUIRED
    merged = _op(template, "merged")
    assert isinstance(merged, MergeRegion)
    assert merged.combination is MergeCombination.ONE_LIVE
    assert [p.name for p in merged.inputs] == ["left", "right"]
    (decl,) = [d for d in template.result_declarations if d.source_ref == "merged"]
    assert decl.cardinality is CardinalityKind.SINGLETON
    assert all(edge.edge_id for edge in template.edges if edge.to_op == "merged")


def test_a_concat_merge_and_a_join_publish_keyed_collections() -> None:
    template = _compile(_workflow(f"""
      - name: a
        spec: {_ECHO}
      - name: b
        spec: {_ECHO}
      - name: both
        dependsOn: [a, b]
        region: {{kind: merge, result: {{visibility: published}}}}
"""))
    merge = _op(template, "both")
    assert merge.combination is None  # concat: every live input
    (decl,) = [d for d in template.result_declarations if d.source_ref == "both"]
    assert decl.cardinality is CardinalityKind.KEYED_COLLECTION
    assert decl.keying == "member"


_LOOP_TEMPLATE = f"""
    templates:
      - name: refinement_body
        inputs:
          - {{name: state, kind: state_reference, role: carried}}
          - {{name: dataset, role: invariant}}
        nodes:
          - name: step
            dependsOn:
              - {{node: $ingress, port: state, input: state}}
              - {{node: $ingress, port: dataset, input: dataset}}
            spec: {_ECHO}
          - name: route
            dependsOn: [{{node: step, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: continue}}, {{name: finish}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: continue}}
            project: [state]
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: finish}}
            project: [state]
            to: {{node: $egress, port: state}}
"""

_LOOP_NODES = f"""
      - name: seed
        spec: {_ECHO}
      - name: dataset_source
        spec: {_ECHO}
      - name: refine
        dependsOn:
          - {{node: seed, input: state}}
          - {{node: dataset_source, input: dataset}}
        region:
          kind: loop
          body_ref: refinement_body
          loop_coordinate: refinement
          carried: [{{name: state, kind: state_reference}}]
          invariants: [{{name: dataset}}]
          result: {{visibility: published, source_port: state}}
      - name: consume
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {_ECHO}
"""


def test_a_loop_compiles_to_one_body_definition_with_routed_returns() -> None:
    template = _compile(_workflow(_LOOP_NODES, _LOOP_TEMPLATE))
    loop = _op(template, "refine")
    assert isinstance(loop, LoopContextRegion)
    assert loop.body_ref == "refinement_body"
    assert [p.name for p in loop.carried] == ["state"]
    assert [p.name for p in loop.invariants] == ["dataset"]
    (body,) = template.definitions
    assert body.kind is DefinitionKind.LOOP_BODY
    assert {p.name: p.role for p in body.inputs} == {
        "state": EntryRole.CARRIED,
        "dataset": EntryRole.INVARIANT,
    }
    step = _op(template, "step")
    route = _op(template, "refinement_body/route")
    assert set(body.members) == {step.operator_id, route.operator_id}
    assert {(e.port, e.to_op, e.to_port) for e in body.entries} == {
        ("state", step.operator_id, "state"),
        ("dataset", step.operator_id, "dataset"),
    }
    returns = {(b.kind, b.from_port, b.projection) for b in body.return_bindings}
    assert returns == {
        (ReturnKind.FEEDBACK, "continue", ("state",)),
        (ReturnKind.EGRESS, "finish", ("state",)),
    }
    # The body never reaches the root through an edge; feedback lives on the body.
    owner = template.definition_of()
    assert all(owner.get(e.from_op) == owner.get(e.to_op) for e in template.edges)
    (decl,) = [d for d in template.result_declarations if d.source_ref == "refine"]
    assert decl.source_port == "state"


def test_a_spawn_enters_a_multi_operator_child_holding_a_loop() -> None:
    templates = _LOOP_TEMPLATE.replace("    templates:\n", "") + """
      - name: researcher
        inputs:
          - {name: topic, role: param}
          - {name: dataset, role: capture}
        returns: [{name: answer}]
        nodes:
          - name: inner
            dependsOn:
              - {node: $ingress, port: topic, input: state}
              - {node: $ingress, port: dataset, input: dataset}
            region:
              kind: loop
              body_ref: refinement_body
              loop_coordinate: refinement
              carried: [{name: state, kind: state_reference}]
              invariants: [{name: dataset}]
        edges:
          - from: {node: inner, port: state}
            to: {node: $return, port: answer}
"""
    template = _compile(
        _workflow(
            f"""
      - name: plan
        spec: {_ECHO}
      - name: dataset_source
        spec: {_ECHO}
      - name: fanout
        dependsOn: [plan, {{node: dataset_source, input: dataset}}]
        region: {{kind: spawn, child: researcher}}
      - name: collect
        dependsOn: [fanout]
        region: {{kind: join, completion: all_settled}}
""",
            "    templates:\n" + templates,
        )
    )
    spawn = _op(template, "fanout")
    assert isinstance(spawn, SpawnRegion)
    assert spawn.child_definition_ref == "researcher"
    child = next(d for d in template.definitions if d.definition_id == "researcher")
    assert child.kind is DefinitionKind.CHILD
    assert child.members == ("researcher/inner",)
    assert {(e.port, e.to_port) for e in child.entries} == {
        ("topic", "state"),
        ("dataset", "dataset"),
    }
    (ret,) = child.return_bindings
    assert (ret.kind, ret.from_op, ret.port) == (
        ReturnKind.RETURN,
        "researcher/inner",
        "answer",
    )


@pytest.mark.parametrize(
    ("spec", "use"),
    [
        ("{taskType: echo, data: {type: list, items: ['${up.out}']}}", "value"),
        ("{taskType: echo, data: {type: list, items: ['${up.task_id}']}}", "route"),
        ("{taskType: echo, data: {type: list, expr: up.items}}", "value"),
        (_ECHO, "order"),
    ],
)
def test_a_dependency_is_classified_by_what_the_spec_reads(spec: str, use: str) -> None:
    template = _compile(_workflow(f"""
      - name: up
        spec: {_ECHO}
      - name: down
        dependsOn: [up]
        spec: {spec}
"""))
    up, down = _op(template, "up"), _op(template, "down")
    (edge,) = [e for e in template.edges if e.to_op == down.operator_id]
    assert edge.use.value.startswith(use)
    assert (up.operator_id in down.value_reads) is (use == "value")


def test_a_read_through_a_task_ancestor_is_recorded() -> None:
    template = _compile(_workflow(f"""
      - name: a
        spec: {_ECHO}
      - name: b
        dependsOn: [a]
        spec: {_ECHO}
      - name: c
        dependsOn: [b]
        spec: {{taskType: echo, data: {{type: list, items: ['${{a.out}}']}}}}
"""))
    c = _op(template, "c")
    assert isinstance(c, LeafOperator)
    assert c.value_reads == (_op(template, "a").operator_id,)


def test_control_flow_is_refused_while_not_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(regions, "CONTROL_FLOW_RUNNABLE", False)
    assert _codes(_workflow(_DIAMOND)) == ["region.unknown-kind"]
    assert _codes(_workflow(_LOOP_NODES, _LOOP_TEMPLATE)) == ["region.unknown-kind"]


def _diamond_with(old: str, new: str) -> str:
    assert old in _DIAMOND
    return _workflow(_DIAMOND.replace(old, new))


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        # A consumer of a branch names the arm it takes.
        (
            "{node: decide, port: right}",
            "decide",
            "branch.unported-dependency",
        ),
        ("rejected: right", "rejected: nowhere", "branch.bad-case"),
        ("accepted: left", "yes: left", "branch.bad-case"),
        ("combination: one_live", "combination: first", "region.bad-combination"),
        (
            "input: input\n            field",
            "input: other\n            field",
            "branch.bad-selection",
        ),
    ],
)
def test_a_malformed_diamond_is_refused(old: str, new: str, code: str) -> None:
    assert code in _codes(_diamond_with(old, new))


def test_a_one_live_merge_of_inputs_that_can_both_run_is_refused() -> None:
    codes = _codes(_workflow(f"""
      - name: a
        spec: {_ECHO}
      - name: b
        spec: {_ECHO}
      - name: both
        dependsOn: [a, b]
        region: {{kind: merge, combination: one_live}}
"""))
    assert codes == ["merge.not-exclusive"]


def test_reading_both_arms_of_one_branch_is_refused() -> None:
    nodes = _DIAMOND + (
        "      - name: both\n"
        "        dependsOn:\n"
        "          - {node: left_work, input: l}\n"
        "          - {node: right_work, input: r}\n"
        f"        spec: {_ECHO}\n"
    )
    assert _codes(_workflow(nodes)) == ["reads.exclusive"]


def test_an_unresolvable_reference_is_refused_at_submission() -> None:
    with pytest.raises(CompileError) as caught:
        _compile(_workflow(f"""
      - name: up
        spec: {_ECHO}
      - name: down
        dependsOn: [up]
        spec: {{taskType: echo, data: {{type: list, items: ['${{nowhere.out}}']}}}}
"""))
    assert [d.code for d in caught.value.diagnostics] == ["reads.unresolved"]


def test_a_reference_without_dependencies_is_left_to_render_as_written() -> None:
    template = _compile(
        _workflow(
            "      - name: solo\n"
            "        spec: {taskType: echo, data: {type: list, items: ['${KEY}']}}\n"
        )
    )
    assert _op(template, "solo").value_reads == ()


def _loop_with(old: str, new: str, *, in_template: bool = True) -> str:
    source = _LOOP_TEMPLATE if in_template else _LOOP_NODES
    assert old in source
    replaced = source.replace(old, new)
    return (
        _workflow(_LOOP_NODES, replaced)
        if in_template
        else _workflow(replaced, _LOOP_TEMPLATE)
    )


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        # Feedback cannot write an invariant.
        (
            "to: {node: $feedback, port: state}",
            "to: {node: $feedback, port: dataset}",
            "definition.bad-return",
        ),
        # An exit from the same arm as the feedback.
        (
            "from: {node: route, port: finish}",
            "from: {node: route, port: continue}",
            "loop.feedback-exit",
        ),
        (
            "role: invariant",
            "role: carried",
            "loop.body-inputs",
        ),
    ],
)
def test_a_malformed_loop_body_is_refused(old: str, new: str, code: str) -> None:
    assert code in _codes(_loop_with(old, new))


def test_a_loop_body_that_never_exits_is_refused() -> None:
    text = _loop_with(
        """
          - from: {node: route, port: finish}
            project: [state]
            to: {node: $egress, port: state}
""",
        "\n",
    )
    assert "loop.no-exit" in _codes(text)


def test_a_loop_input_must_be_bound_exactly_once() -> None:
    text = _loop_with(
        "          - {node: dataset_source, input: dataset}\n", "", in_template=False
    )
    assert "loop.input-binding" in _codes(text)


def test_a_template_node_cannot_read_a_root_node_by_name() -> None:
    text = _loop_with(
        "{node: $ingress, port: dataset, input: dataset}",
        "{node: seed, input: dataset}",
    )
    with pytest.raises(ValueError, match="names no node of template"):
        _compile(text)


def test_a_template_nothing_enters_is_refused() -> None:
    text = _workflow(f"      - name: solo\n        spec: {_ECHO}\n", _LOOP_TEMPLATE)
    assert _codes(text) == ["definition.orphan"]


def test_stored_region_forms_decode_tolerantly() -> None:
    merge = MergeRegion.model_validate(
        {"operator_id": "m", "source_ref": "m", "combination": "zip"}
    )
    assert merge.combination is None
    branch = BranchRegion.model_validate(
        {"operator_id": "b", "source_ref": "b", "selection": "x"}
    )
    assert branch.rule is None
    assert branch.selection == "x"


def test_inspection_renders_definitions_and_dependency_uses() -> None:
    text = _workflow(_LOOP_NODES, _LOOP_TEMPLATE)
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    rendered = build_inspection("wfl-x", parsed, source, bindings=_BINDINGS)
    assert rendered.ok
    lines = rendered.render_text().splitlines()
    assert "  template refinement_body [loop_body]" in lines
    assert "    input dataset [invariant]" in lines
    assert any(line.endswith("--> $feedback.state") for line in lines)
    assert any("[value_required]" in line for line in lines)
