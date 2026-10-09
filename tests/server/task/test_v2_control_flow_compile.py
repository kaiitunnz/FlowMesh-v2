"""Compilation of branch, merge and loop regions and the region definitions they
enter: the finite structure, dependency classification and the refused shapes."""

from typing import Any

import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.inspect import build_inspection
from server.task.v2.representations.operators import (
    BranchRegion,
    LoopContextRegion,
    MergeCombination,
    MergeRegion,
    SelectionCase,
    SpawnRegion,
)
from server.task.v2.representations.results import (
    CardinalityKind,
    ReleaseConditionKind,
)
from server.task.v2.representations.template import (
    BoundaryKind,
    DefinitionKind,
    DependencyUse,
    EntryRole,
    LogicalWorkflowTemplate,
)

_BINDINGS = AgentBindingDefaults(default_backend="codex")
_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


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
    step = _op(template, "refinement_body/step")
    route = _op(template, "refinement_body/route")
    assert set(body.members) == {step.operator_id, route.operator_id}
    entries = template.boundary_edges("refinement_body", BoundaryKind.ENTRY)
    assert {(e.from_port, e.to_op, e.to_port) for e in entries} == {
        ("state", step.operator_id, "state"),
        ("dataset", step.operator_id, "dataset"),
    }
    returns = {
        (e.boundary, e.from_port, e.projection, e.to_op)
        for e in template.boundary_edges(
            "refinement_body", BoundaryKind.FEEDBACK, BoundaryKind.EGRESS
        )
    }
    assert returns == {
        (BoundaryKind.FEEDBACK, "continue", ("state",), "$feedback"),
        (BoundaryKind.EGRESS, "finish", ("state",), "$egress"),
    }
    # The body never reaches the root through a forward edge; it crosses only
    # through its boundary edges, which belong to the body.
    owner = template.definition_of()
    assert all(
        owner.get(e.from_op) == owner.get(e.to_op) == e.definition
        for e in template.edges
        if e.is_forward
    )
    assert len({e.edge_id for e in template.edges}) == len(template.edges)
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
    entries = template.boundary_edges("researcher", BoundaryKind.ENTRY)
    assert {(e.from_port, e.to_port) for e in entries} == {
        ("topic", "state"),
        ("dataset", "dataset"),
    }
    (ret,) = template.boundary_edges("researcher", BoundaryKind.RETURN)
    assert (ret.from_op, ret.to_op, ret.to_port) == (
        "researcher/inner",
        "$return",
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
    assert edge.from_op == up.operator_id
    assert edge.use.value.startswith(use)
    assert not edge.derived


@pytest.mark.parametrize(
    ("read", "use"),
    [("${a.out}", DependencyUse.VALUE_REQUIRED), ("${a.task_id}", "route_required")],
)
def test_a_read_through_a_task_ancestor_is_a_derived_edge(read: str, use: str) -> None:
    template = _compile(_workflow(f"""
      - name: a
        spec: {_ECHO}
      - name: b
        dependsOn: [a]
        spec: {_ECHO}
      - name: c
        dependsOn: [b]
        spec: {{taskType: echo, data: {{type: list, items: ['{read}']}}}}
"""))
    a, c = _op(template, "a"), _op(template, "c")
    (derived,) = [e for e in template.edges if e.to_op == c.operator_id and e.derived]
    assert (derived.from_op, derived.use) == (a.operator_id, use)
    assert derived not in [
        e for e in template.edges if e.to_op == c.operator_id and not e.derived
    ]


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
    solo = _op(template, "solo").operator_id
    assert [e for e in template.edges if e.to_op == solo] == []


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
            "definition.return-not-exclusive",
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
    assert any("==> $feedback.state in refinement_body" in line for line in lines)
    assert any("[value_required]" in line for line in lines)


def test_a_guard_in_a_template_reads_its_own_template() -> None:
    guarded = _LOOP_TEMPLATE.replace(
        "          - name: route\n",
        "          - name: probe\n"
        "            dependsOn: [step]\n"
        "            spec:\n"
        "              taskType: echo\n"
        "              data: {type: list, items: [x]}\n"
        "              condition: {node: step, field: ok, equals: 'yes'}\n"
        "          - name: route\n",
    )
    nodes = _LOOP_NODES + f"      - name: step\n        spec: {_ECHO}\n"
    template = _compile(_workflow(nodes, guarded))
    probe = _op(template, "refinement_body/probe")
    assert probe.guard.node == _op(template, "refinement_body/step").operator_id
    # Without a root namesake the guard still resolves inside its template.
    assert _compile(_workflow(_LOOP_NODES, guarded))


_ARMS = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {_ECHO}
      - name: on_b
        dependsOn: [{{node: decide, port: b}}]
        spec: {_ECHO}
"""


def test_a_value_read_through_an_ancestor_gates_on_that_ancestor() -> None:
    text = _workflow(_ARMS + f"""
      - name: c
        dependsOn: [on_a, on_b]
        spec: {_ECHO}
      - name: d
        dependsOn: [c]
        spec: {{taskType: echo, data: {{type: list, items: ['${{on_a.out}}']}}}}
""")
    template = _compile(text)
    on_a, d = _op(template, "on_a"), _op(template, "d")
    (derived,) = [e for e in template.edges if e.derived]
    assert (derived.from_op, derived.to_op) == (on_a.operator_id, d.operator_id)
    assert derived.use is DependencyUse.VALUE_REQUIRED
    rendered = build_inspection(
        "wfl-x",
        parse_workflow(text, "native"),
        FrontendWorkflowSource.capture(text, "native", name="wf"),
        bindings=_BINDINGS,
    )
    assert any(
        line.endswith("[value_required, derived]")
        for line in rendered.render_text().splitlines()
    )


def _agent(name: str, inputs: str, depends: str = "") -> str:
    return f"""
      - name: {name}
        {depends}
        spec:
          taskType: agent
          task: read
          harness: {{backend: scripted, version: v1, params: {{script: []}}}}
          v2: {{inputs: {inputs}}}
"""


def test_agent_inputs_bind_like_dependencies() -> None:
    template = _compile(
        _workflow(_ARMS + _agent("reader", "[{name: findings, from: on_a}]"))
    )
    reader, on_a = _op(template, "reader"), _op(template, "on_a")
    (edge,) = [e for e in template.edges if e.to_op == reader.operator_id]
    assert (edge.from_op, edge.to_port, edge.use) == (
        on_a.operator_id,
        "findings",
        DependencyUse.VALUE_REQUIRED,
    )
    assert edge.edge_id


def test_agent_inputs_reading_both_exclusive_arms_are_refused() -> None:
    text = _workflow(
        _ARMS + _agent("reader", "[{name: x, from: on_a}, {name: y, from: on_b}]")
    )
    assert "reads.exclusive" in _codes(text)


def test_an_input_bound_twice_is_refused() -> None:
    both = _agent(
        "reader",
        "[{name: x, from: on_b}]",
        "dependsOn: [{node: on_a, input: x}]",
    )
    assert "edge.duplicate-input" in _codes(_workflow(_ARMS + both))
    twice = f"""
      - name: a
        spec: {_ECHO}
      - name: b
        spec: {_ECHO}
      - name: c
        dependsOn: [{{node: a, input: x}}, {{node: b, input: x}}]
        spec: {_ECHO}
"""
    assert "edge.duplicate-input" in _codes(_workflow(twice))


def test_feedback_from_an_undeclared_port_is_refused() -> None:
    text = _loop_with(
        "from: {node: route, port: continue}", "from: {node: route, port: contnue}"
    )
    assert "ports.unknown-output" in _codes(text)


_CHILD = """
    templates:
      - name: one
        inputs: [{name: e, role: param}]
        returns: [{name: out}]
        nodes:
          - name: work
            dependsOn: [{node: $ingress, port: e, input: e}]
            spec: {taskType: echo, data: {type: list, items: [x]}}
        edges:
          - from: {node: work}
            to: {node: $return, port: out}
"""

_FANOUT = f"""
      - name: plan
        spec: {_ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""


def test_a_spawn_returned_raw_from_a_template_is_refused() -> None:
    nested = (
        _CHILD.replace("          - name: work\n", "          - name: inner\n")
        .replace("[{node: $ingress, port: e, input: e}]", "[{node: $ingress, port: e}]")
        .replace(
            "            spec: {taskType: echo, data: {type: list, items: [x]}}\n",
            "            region: {kind: spawn, child: leaf}\n",
        )
        .replace("from: {node: work}", "from: {node: inner}")
    )
    nodes = _FANOUT + f"      - name: leaf\n        spec: {_ECHO}\n"
    assert "dataflow.spawn-dependent" in _codes(_workflow(nodes, nested))


def test_a_template_spawn_fans_out_over_its_entry_value() -> None:
    nested = _CHILD + """
      - name: outer
        inputs: [{name: items, role: param}]
        returns: [{name: out}]
        nodes:
          - name: inner
            dependsOn: [{node: $ingress, port: items}]
            region: {kind: spawn, child: one}
          - name: gather
            dependsOn: [inner]
            region: {kind: join, completion: all_settled}
        edges:
          - from: {node: gather}
            to: {node: $return, port: out}
"""
    template = _compile(
        _workflow(_FANOUT.replace("child: one", "child: outer"), nested)
    )
    inner = _op(template, "outer/inner")
    (entry,) = [e for e in template.edges if e.to_op == inner.operator_id]
    assert entry.boundary is BoundaryKind.ENTRY


def test_every_edge_has_one_identity_and_a_join_runs_on_its_spawn() -> None:
    template = _compile(_workflow(_FANOUT, _CHILD))
    assert len({e.edge_id for e in template.edges}) == len(template.edges)
    collect = _op(template, "collect")
    (membership,) = [e for e in template.edges if e.to_op == collect.operator_id]
    assert membership.use is DependencyUse.ROUTE_REQUIRED


def test_a_join_dependency_on_a_branch_arm_is_a_route() -> None:
    nodes = _ARMS + f"""
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: plan
        spec: {_ECHO}
      - name: collect
        dependsOn: [fan, {{node: decide, port: a}}]
        region: {{kind: join, completion: all_settled}}
"""
    template = _compile(_workflow(nodes, _CHILD))
    collect, decide = _op(template, "collect"), _op(template, "decide")
    (arm,) = [
        e
        for e in template.edges
        if e.to_op == collect.operator_id and e.from_op == decide.operator_id
    ]
    assert arm.use is DependencyUse.ROUTE_REQUIRED


_TWO_FEEDBACKS = """
          - from: {node: step}
            project: [state]
            to: {node: $feedback, port: state}
"""


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        # Two feedback routes at one time.
        (
            "        edges:\n",
            "        edges:\n" + _TWO_FEEDBACKS,
            "definition.return-not-exclusive",
        ),
        # A third arm leads to neither feedback nor exit.
        (
            "outputs: [{name: continue}, {name: finish}]",
            "outputs: [{name: continue}, {name: finish}, {name: stop}]",
            "definition.unrouted-arm",
        ),
        # The continue arm leads nowhere.
        (
            """          - from: {node: route, port: continue}
            project: [state]
            to: {node: $feedback, port: state}
""",
            "",
            "definition.unrouted-arm",
        ),
    ],
)
def test_return_routes_are_exclusive_and_cover_every_arm(
    old: str, new: str, code: str
) -> None:
    assert code in _codes(_loop_with(old, new))


def test_two_unconditional_returns_are_refused() -> None:
    twice = _CHILD + """          - from: {node: work}
            to: {node: $return, port: out}
""".replace("{node: work}", "{node: work, port: out}")
    assert "definition.return-not-exclusive" in _codes(_workflow(_FANOUT, twice))


@pytest.mark.parametrize("key", ["default: left", "until: done"])
def test_unknown_branch_and_loop_keys_are_refused(key: str) -> None:
    if key.startswith("default"):
        text = _workflow(
            _DIAMOND.replace(
                "          kind: branch\n", f"          kind: branch\n          {key}\n"
            )
        )
    else:
        text = _loop_with(
            "          kind: loop\n",
            f"          kind: loop\n          {key}\n",
            in_template=False,
        )
    assert _codes(text) == ["region.unknown-field"]


def test_a_call_returns_only_what_its_template_returns() -> None:
    nodes = f"""
      - name: plan
        spec: {_ECHO}
      - name: ask
        dependsOn: [plan]
        region: {{kind: call, child: one, returns: [answer]}}
"""
    assert "call.unknown-return" in _codes(_workflow(nodes, _CHILD))
    assert _compile(_workflow(nodes.replace("[answer]", "[out]"), _CHILD))


def test_an_input_name_cannot_hide_another_node() -> None:
    nodes = f"""
      - name: a
        spec: {_ECHO}
      - name: b
        spec: {_ECHO}
      - name: c
        dependsOn: [b, {{node: a, input: b}}]
        spec: {{taskType: echo, data: {{type: list, items: ['${{b.out}}']}}}}
"""
    assert _codes(_workflow(nodes)) == ["reads.input-shadows-node"]


def test_a_definition_input_is_readable_by_its_port_name() -> None:
    reading = _CHILD.replace(
        "[{node: $ingress, port: e, input: e}]\n"
        "            spec: {taskType: echo, data: {type: list, items: [x]}}",
        "[{node: $ingress, port: e}]\n"
        "            spec: {taskType: echo, data: {type: list, items: ['${e.x}']}}",
    )
    template = _compile(_workflow(_FANOUT, reading))
    (entry,) = template.boundary_edges("one", BoundaryKind.ENTRY)
    assert entry.use is DependencyUse.VALUE_REQUIRED


def test_an_early_published_join_releases_on_its_winner() -> None:
    nodes = _FANOUT.replace(
        "completion: all_settled}",
        "completion: any, residual: cancel, result: {visibility: published}}",
    )
    template = _compile(_workflow(nodes, _CHILD))
    (decl,) = [d for d in template.result_declarations if d.source_ref == "collect"]
    assert decl.release is ReleaseConditionKind.JOIN_WINNER


def test_merge_inputs_from_one_branch_are_named_apart() -> None:
    nodes = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: either
        dependsOn: [{{node: decide, port: a}}, {{node: decide, port: b}}]
        region: {{kind: merge, combination: one_live}}
"""
    merge = _op(_compile(_workflow(nodes)), "either")
    assert [p.name for p in merge.inputs] == ["decide.a", "decide.b"]


def test_template_members_are_named_by_their_template() -> None:
    template = _compile(_workflow(_LOOP_NODES, _LOOP_TEMPLATE))
    sources = {e.source_id for e in template.source_map}
    assert {"refinement_body/step", "refinement_body/route"} <= sources
    assert "step" not in sources


def test_a_loop_may_take_an_ordering_dependency() -> None:
    binding = "          - {node: dataset_source, input: dataset}\n"
    nodes = (
        _LOOP_NODES.replace(binding, binding + "          - consume_first\n")
        + f"      - name: consume_first\n        spec: {_ECHO}\n"
    )
    template = _compile(_workflow(nodes, _LOOP_TEMPLATE))
    loop, first = _op(template, "refine"), _op(template, "consume_first")
    (edge,) = [
        e
        for e in template.edges
        if e.to_op == loop.operator_id and e.from_op == first.operator_id
    ]
    assert edge.use is DependencyUse.ORDER_ONLY


def test_a_spawn_may_capture_a_join_aggregate() -> None:
    capturing = _CHILD.replace(
        "inputs: [{name: e, role: param}]",
        "inputs: [{name: e, role: param}, {name: ctx, role: capture}]",
    )
    nodes = _FANOUT + """
      - name: second
        dependsOn: [plan, {node: collect, input: ctx}]
        region: {kind: spawn, child: one}
      - name: gather
        dependsOn: [second]
        region: {kind: join, completion: all_settled}
"""
    nodes = nodes.replace(
        "      - name: fan\n        dependsOn: [plan]",
        "      - name: fan\n        dependsOn: [plan, {node: plan, input: ctx}]",
    )
    assert _compile(_workflow(nodes, capturing))


_SELECT_ON = f"""
      - name: decide
        dependsOn: [{{node: SOURCE, port: out, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: l}}, {{name: r}}]
          selection: {{input: input}}
      - name: on_l
        dependsOn: [{{node: decide, port: l}}]
        spec: {_ECHO}
"""


@pytest.mark.parametrize(
    ("nodes", "source"),
    [
        (
            f"""
      - name: a
        spec: {_ECHO}
      - name: b
        spec: {_ECHO}
      - name: both
        dependsOn: [{{node: a, input: x}}, {{node: b, input: y}}]
        region: {{kind: merge, combination: concat}}
""",
            "both",
        ),
        (_FANOUT, "collect"),
        (
            f"""
      - name: plan
        spec: {_ECHO}
      - name: called
        dependsOn: [plan]
        region: {{kind: call, child: one, returns: [out]}}
""",
            "called",
        ),
    ],
    ids=["concat-merge", "join", "call"],
)
def test_a_branch_selecting_on_an_aggregate_is_refused(nodes: str, source: str) -> None:
    templates = _CHILD if source != "both" else ""
    text = _workflow(nodes + _SELECT_ON.replace("SOURCE", source), templates)
    assert "dataflow.region-input" in _codes(text)


_PROJECTED = f"""
      - name: a
        spec: {_ECHO}
"""


@pytest.mark.parametrize(
    "consumer",
    [
        f"""      - name: b
        dependsOn: [{{node: a, project: [inner]}}]
        spec: {_ECHO}
""",
        """      - name: refine
        dependsOn: [{node: a, input: state}, {node: a, project: [inner]}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
""",
        f"""      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [a]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan, {{node: a, project: [inner]}}]
        region: {{kind: join, completion: all_settled}}
""",
    ],
    ids=["task", "loop", "join"],
)
def test_a_projection_no_input_names_is_refused(consumer: str) -> None:
    body = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
        edges:
          - from: {{node: step}}
            to: {{node: $egress, port: state}}
"""
    assert _codes(_workflow(_PROJECTED + consumer, body)) == [
        "reads.unnamed-projection"
    ]


def test_a_template_member_projection_no_input_names_is_refused() -> None:
    body = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: next
            dependsOn: [{{node: step, project: [items]}}]
            spec: {_ECHO}
        edges:
          - from: {{node: next}}
            to: {{node: $egress, port: state}}
"""
    nodes = _PROJECTED + """      - name: refine
        dependsOn: [{node: a, input: state}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
"""
    assert _codes(_workflow(nodes, body)) == ["reads.unnamed-projection"]


@pytest.mark.parametrize(
    "consumer",
    [
        f"""      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [{{node: a, project: [inner]}}]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
""",
        """      - name: either
        dependsOn: [{node: a, project: [inner]}]
        region: {kind: merge, combination: concat}
""",
        f"""      - name: route
        dependsOn: [{{node: a, project: [inner]}}]
        region:
          kind: branch
          outputs: [{{name: p}}, {{name: q}}]
          selection: {{field: [label]}}
      - name: on_p
        dependsOn: [{{node: route, port: p}}]
        spec: {_ECHO}
      - name: on_q
        dependsOn: [{{node: route, port: q}}]
        spec: {_ECHO}
""",
    ],
    ids=["spawn", "merge", "branch"],
)
def test_a_region_whose_input_port_carries_a_projection_may_leave_it_unnamed(
    consumer: str,
) -> None:
    assert _compile(_workflow(_PROJECTED + consumer))
