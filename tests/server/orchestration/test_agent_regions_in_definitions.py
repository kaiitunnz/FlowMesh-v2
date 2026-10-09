"""An agent that declares a child region runs inside a loop body or a child
definition, and its region resolves in the agent's own context and time."""

import pytest

from server.orchestration.state import ControlStatus, LoopInstanceStatus, ValueRef

from .control_flow import ECHO, Driver, workflow

_AGENT = """
            spec:
              taskType: agent
              task: think
              harness: {backend: scripted, version: v1, params: {script: []}}
              v2: {child: [kid]}
"""

_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: kid
            spec: {ECHO}
          - name: think
            dependsOn: [{{node: $ingress, port: state, input: state}}]
{_AGENT}
          - name: route
            dependsOn: [{{node: think, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: finish}}]
              selection: {{input: input}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: finish}}
            to: {{node: $egress, port: state}}
"""

_LOOP = f"""
      - name: seed
        spec: {ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
      - name: consume
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {ECHO}
"""


def _region_joins(run: Driver) -> list[ControlStatus]:
    return [
        state.status
        for state in run.engine.to_snapshot().control_states
        if state.key.split("@")[0].endswith(":kid:spawn:join")
    ]


@pytest.mark.parametrize("restart", [False, True])
def test_an_agent_region_in_a_loop_body_releases_at_each_time(restart: bool) -> None:
    run = Driver(workflow(_LOOP, _BODY))
    run.run_one("seed")
    run.run_one("think")
    run.select("again")
    if restart:
        run.restore()
    run.run_one("think")
    run.select("finish")
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.RELEASED
    # One region join per time, each released where its agent ran.
    assert _region_joins(run) == [ControlStatus.LIVE, ControlStatus.LIVE]
    run.run_one("consume")


_CHILD = f"""
    templates:
      - name: one
        inputs: [{{name: state, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: kid
            spec: {ECHO}
          - name: think
            dependsOn: [{{node: $ingress, port: state, input: state}}]
{_AGENT}
        edges:
          - from: {{node: think}}
            to: {{node: $return, port: out}}
"""

_FAN = f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: after
        dependsOn: [collect]
        spec: {ECHO}
"""


def test_an_agent_region_in_a_child_definition_releases_in_its_child() -> None:
    run = Driver(workflow(_FAN, _CHILD))
    run.run_one("plan")
    for index, element in enumerate(("a", "b")):
        run.apply(
            run.engine.enter_definition_child(
                "fan", index, ValueRef(kind="inline", literal=element)
            )
        )
    run.apply(run.engine.seal_spawn("fan"))
    for task_id in run.ready_named("think"):
        run.run(task_id)
    assert _region_joins(run) == [ControlStatus.LIVE, ControlStatus.LIVE]
    collect = run.engine.control_state("collect")
    assert collect is not None and collect.status is ControlStatus.LIVE
    run.run_one("after")


def test_a_dead_agent_leaves_its_region_dead_and_the_loop_drains() -> None:
    gated = _BODY.replace(
        "          - name: think\n"
        "            dependsOn: [{node: $ingress, port: state, input: state}]\n",
        "          - name: gate\n"
        "            dependsOn: [{node: $ingress, port: state, input: input}]\n"
        "            region:\n"
        "              kind: branch\n"
        "              inputs: [{name: input}]\n"
        "              outputs: [{name: think}, {name: skip}]\n"
        "              selection: {input: input}\n"
        "          - name: think\n"
        "            dependsOn: [{node: gate, port: think, input: state}]\n",
    ) + (
        "          - from: {node: gate, port: skip}\n"
        "            to: {node: $egress, port: state}\n"
    )
    run = Driver(workflow(_LOOP, gated))
    run.run_one("seed")
    run.select("skip")
    assert _region_joins(run) == [ControlStatus.DEAD]
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.RELEASED
    run.run_one("consume")
