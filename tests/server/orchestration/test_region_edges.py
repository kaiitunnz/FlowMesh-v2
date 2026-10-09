"""Branch routing and dead-path resolution through the orchestration engine."""

import pytest

from server.orchestration.state import (
    ControlStatus,
    PublicationOutcome,
    ValueRef,
    WorkItemStatus,
)

from .control_flow import ECHO, Driver, workflow

_DIAMOND = f"""
      - name: classify
        spec: {ECHO}
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
        spec: {ECHO}
      - name: left_more
        dependsOn: [left_work]
        spec: {ECHO}
      - name: right_work
        dependsOn: [{{node: decide, port: right}}]
        spec: {ECHO}
      - name: merged
        dependsOn:
          - {{node: left_more, input: left}}
          - {{node: right_work, input: right}}
        region:
          kind: merge
          combination: one_live
          result: {{visibility: published}}
      - name: after
        dependsOn: [{{node: merged, input: m}}]
        spec: {ECHO}
"""


@pytest.mark.parametrize(
    ("label", "live", "dead"),
    [
        ("accepted", ["left_work", "left_more"], ["right_work"]),
        ("rejected", ["right_work"], ["left_work", "left_more"]),
    ],
)
def test_a_diamond_runs_only_its_selected_arm_and_its_merge_once(
    label: str, live: list[str], dead: list[str]
) -> None:
    run = Driver(workflow(_DIAMOND))
    run.run_one("classify")
    assert run.ready == []
    run.select(label)
    # The dead arm settles before the live arm runs, and the merge waits for it.
    assert [run.name(t) for t in run.skipped] == dead
    for name in dead:
        assert run.status(name) is WorkItemStatus.SKIPPED
    merged = run.engine.control_state("merged")
    assert merged is None or merged.status is ControlStatus.PENDING
    for name in live:
        run.run_one(name)
    merged = run.engine.control_state("merged")
    assert merged is not None and merged.status is ControlStatus.LIVE
    run.run_one("after")
    assert run.ran == ["classify", *live, "after"]
    (publication,) = [
        run.engine.output_publication(decl.output_id)
        for _, decl in run.engine.published_outputs()
    ]
    assert publication is not None
    assert publication.outcome is PublicationOutcome.SUCCESS


def test_a_shared_node_behind_two_dead_arms_settles_dead_once() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}, {{name: c}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {ECHO}
      - name: on_b
        dependsOn: [{{node: decide, port: b}}]
        spec: {ECHO}
      - name: on_c
        dependsOn: [{{node: decide, port: c}}]
        spec: {ECHO}
      - name: shared
        dependsOn: [on_a, on_b]
        spec: {ECHO}
"""
    run = Driver(workflow(nodes))
    run.run_one("classify")
    run.select("c")
    assert sorted(run.name(t) for t in run.skipped) == ["on_a", "on_b", "shared"]
    assert run.skipped.count(run.ops["shared"]) == 1
    run.run_one("on_c")
    assert run.ran == ["classify", "on_c"]


def test_ordering_only_dependencies_run_on_any_live_arm() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {ECHO}
      - name: on_b
        dependsOn: [{{node: decide, port: b}}]
        spec: {ECHO}
      - name: either
        dependsOn: [on_a, on_b]
        spec: {ECHO}
"""
    # Neither dependency is read, so the live one alone lets it run.
    run = Driver(workflow(nodes.replace("  - name: shared", "")))
    run.run_one("classify")
    run.select("b")
    run.run_one("on_b")
    run.run_one("either")
    assert run.ran == ["classify", "on_b", "either"]


def test_a_required_read_of_a_dead_arm_is_inactive_beside_a_live_helper() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: helper
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {ECHO}
      - name: reader
        dependsOn: [on_a, helper]
        spec: {{taskType: echo, data: {{type: list, items: ['${{on_a.x}}']}}}}
"""
    run = Driver(workflow(nodes))
    run.run_one("helper")
    run.run_one("classify")
    run.select("b")
    assert sorted(run.name(t) for t in run.skipped) == ["on_a", "reader"]
    assert run.ready == []


def test_success_empty_dead_failure_and_cancellation_stay_distinct() -> None:
    run = Driver(workflow(_DIAMOND))
    # A failed input fails every potential route instead of choosing one.
    run.run_one("classify", fail=True)
    assert run.engine.pending_branch_reads() == []
    decide = run.engine.control_state("decide")
    assert decide is None or decide.status is not ControlStatus.DEAD
    assert run.skipped == []
    assert {run.name(t) for t in run.failed} >= {"left_work", "right_work", "after"}

    cancelled = Driver(workflow(_DIAMOND))
    cancelled.run_one("classify")
    cancelled.apply(cancelled.engine.cancel_instance())
    assert cancelled.engine.pending_branch_reads() == []
    assert cancelled.status("left_work") is WorkItemStatus.CANCELLED


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("maybe", "matches no case"),
        (None, "missing"),
        (["yes"], "list"),
        (3, "int"),
    ],
)
def test_a_selection_naming_no_port_fails_the_branch(
    value: object, reason: str
) -> None:
    run = Driver(workflow(_DIAMOND))
    run.run_one("classify")
    run.select(value)
    decide = run.engine.control_state("decide")
    assert decide is not None and decide.status is ControlStatus.FAILED
    assert reason in (decide.reason or "")
    assert run.skipped == []
    assert {run.name(t) for t in run.failed} >= {"left_work", "right_work", "after"}
    assert run.engine.pending_branch_reads() == []


def test_an_unreadable_input_fails_the_branch() -> None:
    run = Driver(workflow(_DIAMOND))
    run.run_one("classify")
    run.select(None, error="selection input is corrupt")
    decide = run.engine.control_state("decide")
    assert decide is not None and decide.status is ControlStatus.FAILED


def test_an_accepted_decision_survives_restart_and_is_never_revised() -> None:
    run = Driver(workflow(_DIAMOND))
    run.run_one("classify")
    run.select("accepted")
    run.restore()
    decision = run.engine.branch_decision("decide")
    assert decision is not None and (decision.port, decision.case) == (
        "left",
        "accepted",
    )
    # A redelivered selection changes nothing.
    assert run.engine.accept_branch_selection("decide", "rejected").ready == []
    assert run.engine.branch_decision("decide") == decision
    assert run.status("right_work") is WorkItemStatus.SKIPPED
    run.run_one("left_work")
    run.run_one("left_more")
    run.run_one("after")


def test_a_concat_merge_freezes_its_live_members_in_input_order() -> None:
    nodes = f"""
      - name: a
        spec: {ECHO}
      - name: b
        spec: {ECHO}
      - name: both
        dependsOn: [{{node: b, input: second}}, {{node: a, input: first}}]
        region: {{kind: merge, combination: concat}}
"""
    run = Driver(workflow(nodes))
    run.run_one("a")
    run.run_one("b")
    state = run.engine.control_state("both")
    assert state is not None and state.status is ControlStatus.LIVE
    aggregate = state.outputs["out"]
    assert [m.key for m in aggregate.members] == ["second", "first"]


def test_a_one_live_merge_with_two_live_inputs_fails() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {ECHO}
      - name: on_b
        dependsOn: [{{node: decide, port: b}}]
        spec: {ECHO}
      - name: merged
        dependsOn: [on_a, on_b]
        region: {{kind: merge, combination: one_live}}
"""
    run = Driver(workflow(nodes))
    run.run_one("classify")
    run.select("a")
    run.run_one("on_a")
    state = run.engine.control_state("merged")
    assert state is not None and state.status is ControlStatus.LIVE


_ARMS = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input}}
      - name: on_a
        dependsOn: [{{node: decide, port: a}}]
        spec: {ECHO}
      - name: on_b
        dependsOn: [{{node: decide, port: b}}]
        spec: {ECHO}
"""


def test_a_value_read_past_a_live_dependency_on_a_dead_arm_is_inactive() -> None:
    nodes = _ARMS + f"""
      - name: c
        dependsOn: [on_a, on_b]
        spec: {ECHO}
      - name: d
        dependsOn: [c]
        spec: {{taskType: echo, data: {{type: list, items: ['${{on_a.out}}']}}}}
"""
    run = Driver(workflow(nodes))
    run.run_one("classify")
    run.select("b")
    run.run_one("on_b")
    run.run_one("c")
    assert run.status("d") is WorkItemStatus.SKIPPED
    assert run.ran == ["classify", "on_b", "c"]


def test_an_agent_input_bound_to_a_dead_arm_leaves_the_agent_inactive() -> None:
    # A live ordering dependency beside the dead input does not activate the agent.
    nodes = _ARMS + """
      - name: reader
        dependsOn: [classify]
        spec:
          taskType: agent
          task: read
          harness: {backend: scripted, version: v1, params: {script: []}}
          v2: {inputs: [{name: findings, from: on_a}]}
"""
    run = Driver(workflow(nodes))
    run.run_one("classify")
    run.select("b")
    assert run.status("reader") is WorkItemStatus.SKIPPED


_CHILD = f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: work
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {ECHO}
        edges:
          - from: {{node: work}}
            to: {{node: $return, port: out}}
"""


@pytest.mark.parametrize(
    ("fan_input", "join_deps"),
    [
        # The spawn itself is on the dead arm, beside a live ordering input.
        ("{node: decide, port: a}", "[fan, on_b]"),
        # The spawn is live, but the join runs only on the dead arm.
        ("plan", "[fan, {node: decide, port: a}]"),
    ],
)
def test_a_join_on_a_dead_route_is_dead(fan_input: str, join_deps: str) -> None:
    nodes = _ARMS + f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [{fan_input}]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: {join_deps}
        region: {{kind: join, completion: all_settled}}
      - name: after
        dependsOn: [collect]
        spec: {ECHO}
"""
    run = Driver(workflow(nodes, _CHILD))
    run.run_one("plan")
    run.run_one("classify")
    run.select("b")
    run.run_one("on_b")
    collect = run.engine.control_state("collect")
    assert collect is not None and collect.status is ControlStatus.DEAD
    assert run.status("after") is WorkItemStatus.SKIPPED


def test_a_branch_reads_a_merge_through_its_named_port() -> None:
    nodes = f"""
      - name: a
        spec: {ECHO}
      - name: b
        spec: {ECHO}
      - name: both
        dependsOn: [{{node: a, input: x}}, {{node: b, input: y}}]
        region: {{kind: merge, combination: concat}}
      - name: decide
        dependsOn: [{{node: both, port: out, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: l}}, {{name: r}}]
          selection: {{input: input}}
      - name: on_l
        dependsOn: [{{node: decide, port: l}}]
        spec: {ECHO}
"""
    run = Driver(workflow(nodes))
    run.run_one("a")
    run.run_one("b")
    ((key, value),) = run.engine.pending_branch_reads()
    assert key == "decide" and value.kind == "aggregate"


def test_a_merge_forwards_a_join_read_through_its_named_port() -> None:
    templates = f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: kidwork
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {ECHO}
        edges:
          - from: {{node: kidwork}}
            to: {{node: $return, port: out}}
"""
    nodes = f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: m
        dependsOn: [{{node: collect, port: out, input: all}}]
        region: {{kind: merge, combination: one_live}}
      - name: after
        dependsOn: [{{node: m, input: x}}]
        spec: {ECHO}
"""
    run = Driver(workflow(nodes, templates))
    run.run_one("plan")
    element = ValueRef(kind="inline", literal="e")
    run.apply(run.engine.enter_definition_child("fan", element))
    run.apply(run.engine.seal_spawn("fan"))
    run.run_one("kidwork")
    m = run.engine.control_state("m")
    assert m is not None and m.status is ControlStatus.LIVE
    assert m.outputs["out"].kind == "aggregate"
    run.run_one("after")


def test_an_agent_takes_a_merge_read_through_its_named_port() -> None:
    templates = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: a
            dependsOn: [{{node: $ingress, port: state, input: s}}]
            spec: {ECHO}
          - name: b
            dependsOn: [{{node: $ingress, port: state, input: s}}]
            spec: {ECHO}
          - name: both
            dependsOn: [{{node: a, input: x}}, {{node: b, input: y}}]
            region: {{kind: merge, combination: concat}}
          - name: think
            dependsOn: [{{node: both, port: out, input: findings}}]
            spec:
              taskType: agent
              task: think
              harness: {{backend: scripted, version: v1, params: {{script: []}}}}
          - name: route
            dependsOn: [{{node: think, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: continue}}, {{name: finish}}]
              selection: {{input: input}}
        edges:
          - from: {{node: route, port: continue}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: finish}}
            to: {{node: $egress, port: state}}
"""
    nodes = f"""
      - name: seed
        spec: {ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
"""
    run = Driver(workflow(nodes, templates))
    run.run_one("seed")
    run.run_one("a")
    run.run_one("b")
    assert run.ready_named("think") != []
