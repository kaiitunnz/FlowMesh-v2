"""Branch routing and dead-path resolution through the orchestration engine."""

import pytest

from server.orchestration.state import (
    ControlStatus,
    PublicationOutcome,
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
    aggregate = state.outputs[""]
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
