"""Admission authority for fixed leaf effects and agent faces wherever they run."""

import pytest

from server.orchestration.state import AuthorityDecisionKind, BoundaryEvent, ValueRef
from shared.harness.boundary import BoundaryEventKind

from .control_flow import ECHO, Driver, workflow

_EFFECT = (
    "{taskType: ssh, interactive: false, image: 'alpine:3', command: [sh, -c, 'true']}"
)

_ROOT = f"""
      - name: act
        spec: {_EFFECT}
"""

_SHORTHAND = f"""
      - name: plan
        spec: {ECHO}
      - name: act
        spec: {_EFFECT}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: act}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""

_DEFINITION_NODES = f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""
_DEFINITION = f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: act
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {_EFFECT}
        edges:
          - from: {{node: act}}
            to: {{node: $return, port: out}}
"""

_LOOP_NODES = f"""
      - name: seed
        spec: {ECHO}
      - name: repeat
        dependsOn: [{{node: seed, input: s}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: t
          carried: [{{name: s}}]
"""
_LOOP = f"""
    templates:
      - name: body
        inputs: [{{name: s, role: carried}}]
        nodes:
          - name: act
            dependsOn: [{{node: $ingress, port: s, input: s}}]
            spec: {_EFFECT}
          - name: route
            dependsOn: [{{node: act, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: s}}
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: s}}
"""


def _act_decisions(where: str, granted: frozenset[str] | None) -> list[str]:
    """Bring the effect leaf ``act`` to admission where it runs; return its
    authority decisions."""
    element = ValueRef(kind="inline", literal="e")
    match where:
        case "root":
            run = Driver(workflow(_ROOT), granted=granted)
        case "shorthand":
            run = Driver(workflow(_SHORTHAND), granted=granted)
            run.run_one("plan")
            run.apply(run.engine.materialize_child("fan", value_ref=element))
        case "definition":
            run = Driver(workflow(_DEFINITION_NODES, _DEFINITION), granted=granted)
            run.run_one("plan")
            run.apply(run.engine.enter_definition_child("fan", element))
        case _:
            run = Driver(workflow(_LOOP_NODES, _LOOP), granted=granted)
            run.run_one("seed")
    act_op = run.ops["act"]
    snapshot = run.engine.to_snapshot()
    acts = {w.work_item_id for w in snapshot.work_items if w.operator_id == act_op}
    return [
        d.kind.value for d in snapshot.authority_decisions if d.work_item_id in acts
    ]


@pytest.mark.parametrize("where", ["root", "shorthand", "definition", "loop"])
def test_a_fixed_leaf_effect_is_decided_by_the_policy_wherever_it_runs(
    where: str,
) -> None:
    assert _act_decisions(where, None) == ["granted"]
    assert _act_decisions(where, frozenset()) == ["denied"]


_TEAM = f"""
    templates:
      - name: team
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: act
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {_EFFECT}
          - name: helper
            dependsOn: [act]
            spec:
              taskType: agent
              task: research
              harness: {{backend: scripted, version: v1, params: {{script: []}}}}
              v2:
                authority: {{invoke: [search]}}
                tools: [{{name: search}}]
        edges:
          - from: {{node: helper}}
            to: {{node: $return, port: out}}
"""


def test_a_child_agent_runs_under_its_delegated_face_beside_an_admitted_leaf() -> None:
    nodes = _DEFINITION_NODES.replace("child: one", "child: team")
    run = Driver(workflow(nodes, _TEAM))
    run.run_one("plan")
    run.apply(
        run.engine.enter_definition_child("fan", ValueRef(kind="inline", literal="e"))
    )
    (act,) = run.ready
    run.run(act)
    (helper,) = run.ready
    face = run.engine.effective_invoke_face(helper)
    # The spawn delegates nothing, so the agent's face is empty; the leaf it runs
    # beside was admitted by policy, and naming that leaf grants the agent nothing.
    assert "search" not in face
    assert run.ops["act"] not in face
    assert face == ()


def test_an_agent_in_a_child_definition_reports_its_delegated_grant() -> None:
    nodes = _DEFINITION_NODES.replace(
        "child: one", "child: team, authority: {invoke: [search], delegate: [search]}"
    )
    run = Driver(workflow(nodes, _TEAM))
    run.run_one("plan")
    run.apply(
        run.engine.enter_definition_child("fan", ValueRef(kind="inline", literal="e"))
    )
    run.run(run.ready[0])
    (helper,) = run.ready
    run.engine.on_dispatched(helper, "w1")
    run.engine.route_boundary_event(
        helper,
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="search",
            request_payload="{}",
        ),
    )
    envelope = run.engine.tool_dispatch_envelope(helper, "c0")
    assert envelope is not None and envelope.grant_snapshot is not None
    (delegated,) = run.engine.to_snapshot().delegated_grants
    assert envelope.grant_snapshot.grant_id == delegated.grant_id


def test_a_dead_child_entry_mints_no_grant() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: go}}, {{name: skip}}]
          selection: {{input: input}}
      - name: fan
        dependsOn: [{{node: decide, port: go}}]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""
    run = Driver(workflow(nodes, _DEFINITION))
    run.run_one("classify")
    run.select("skip")
    trace = run.engine.contract_trace()
    assert ("grant_delegated", "fan") not in trace
    assert run.engine.control_state("collect").status.value == "dead"  # type: ignore[union-attr]


def test_admission_is_decided_once_across_restart_and_redelivery() -> None:
    run = Driver(workflow(_LOOP_NODES, _LOOP))
    run.run_one("seed")
    (act,) = run.ready
    run.restore()
    run.apply(run.engine.reconsider_admission(act))
    snapshot = run.engine.to_snapshot()
    wi = run.engine.work_item(act)
    assert wi is not None
    decisions = [
        d for d in snapshot.authority_decisions if d.work_item_id == wi.work_item_id
    ]
    assert [d.kind for d in decisions] == [AuthorityDecisionKind.GRANTED]
