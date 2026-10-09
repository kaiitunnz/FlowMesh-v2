"""An agent's declared inputs read the value each of its edges delivers, as a leaf's
inputs do."""

from typing import Any

import pytest

from shared.schemas.result.binding import binding_text
from tests.server.task.test_runtime_control_flow import _Run, _workflow

_AGENT = """
          taskType: agent
          task: read
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""

_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _member_values(run: _Run, task_id: str) -> dict[str, list[Any]]:
    """What each member of an agent's first turn reads as on its worker, by port."""
    with run.runtime._lock:
        bindings = run.runtime._agent_inputs.agent_input_bindings(run.engine, task_id)
    return {
        binding.port: [
            (
                binding_text(member.source, run.reader.read)
                if member.source is not None
                else member.value
            )
            for member in binding.members
        ]
        for binding in bindings
    }


def _ready(run: _Run, name: str) -> str:
    return next(t for t in run.ready if run.name(t) == name)


@pytest.mark.anyio
async def test_a_root_agent_reads_the_part_of_a_result_its_input_projects() -> None:
    run = await _Run().start(_workflow(f"""
      - name: a
        spec: {_ECHO}
      - name: reader
        dependsOn: [{{node: a, input: x, project: [inner]}}]
        spec:{_AGENT}
"""))
    run.run("a", {"inner": "wanted", "outer": "unwanted"})

    assert _member_values(run, _ready(run, "reader")) == {"x": ["wanted"]}


_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec:{_AGENT.replace(chr(10) + '          ', chr(10) + '              ')}
          - name: judge
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: route
            dependsOn: [{{node: judge, input: input}}, step]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
            project: [next]
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_LOOP = """
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: refine
        dependsOn: [{node: seed, input: state}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
"""


@pytest.mark.anyio
async def test_a_body_agent_reads_a_projected_carried_value_at_a_later_time() -> None:
    run = await _Run().start(_workflow(_LOOP, _BODY))
    run.run("seed", {"value": "seed"})
    run.run("step", {"value": "s0"})
    run.run("judge", {"route": "again", "next": {"value": "wanted"}, "value": "no"})

    assert _member_values(run, _ready(run, "step")) == {
        "state": ['{"value": "wanted"}']
    }


_TWO_RETURNS = """
    templates:
      - name: two
        inputs: [{name: e, role: param}]
        returns: [{name: a}, {name: b}]
        nodes:
          - name: w
            dependsOn: [{node: $ingress, port: e, input: e}]
            spec: {taskType: echo, data: {type: list, items: [x]}}
        edges:
          - from: {node: w}
            to: {node: $return, port: a}
          - from: {node: w}
            to: {node: $return, port: b}
            project: [value]
"""


@pytest.mark.anyio
async def test_a_root_agent_reads_a_call_return_port_and_runs() -> None:
    run = await _Run().start(
        _workflow(
            f"""
      - name: src
        spec: {_ECHO}
      - name: c
        dependsOn: [src]
        region: {{kind: call, child: two, returns: [a, b]}}
      - name: reader
        dependsOn: [{{node: c, port: b, input: x}}]
        spec:{_AGENT}
""",
            _TWO_RETURNS,
        )
    )
    run.run("src", {"items": ["only"]})
    run.run("w", {"value": "wanted"})

    assert _member_values(run, _ready(run, "reader")) == {"x": ["wanted"]}


@pytest.mark.anyio
async def test_agent_children_of_a_projected_fan_out_read_their_element() -> None:
    worker_agent = _AGENT.replace(
        "boundary: [invocation, yield]",
        "boundary: [invocation, yield]\n            inputs: [facet]",
    )
    run = await _Run().start(_workflow(f"""
      - name: plan
        spec: {_ECHO}
      - name: worker_agent
        spec:{worker_agent}
      - name: fan
        dependsOn: [{{node: plan, project: [nested]}}]
        region: {{kind: spawn, child: worker_agent}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""))
    run.run("plan", {"items": ["no-0", "no-1", "no-2"], "nested": ["e0", "e1"]})

    kids = [t for t in run.ready if run.name(t) == "worker_agent"]
    assert sorted(
        value
        for kid in kids
        for port in _member_values(run, kid).values()
        for value in port
    ) == ["e0", "e1"]
