"""A file a task produced reaches its consumer as the producer's URL on every path
its reference travels."""

from typing import Any

import pytest

from server.task.v2 import CompileError
from tests.server.task.test_agent_region_inputs import _AGENT, _member_values, _ready
from tests.server.task.test_runtime_control_flow import _Run, _workflow
from tests.server.task.test_scoped_inputs import _dispatch, _worker_reads
from worker.executors.mixins.data import DataMixin
from worker.executors.utils.artifacts import maybe_resolve_artifact_ref
from worker.executors.utils.expressions import item_steps

_URL = "http://fm.example"
_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _trained(task: str, **extra: Any) -> dict[str, Any]:
    return {
        "final_lora_archive": {"path": "final_lora.tar.gz"},
        "_artifacts": {"base_dir": f"/results/{task}", "base_url": _URL},
        **extra,
    }


def _file(task: str) -> str:
    return f"{_URL}/api/v1/results/{task}/files/final_lora.tar.gz"


@pytest.mark.anyio
async def test_a_root_input_projecting_a_file_reads_its_url_whole() -> None:
    run = await _Run().start(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: use
        dependsOn: [{{node: train, input: archive, project: [final_lora_archive]}}]
        spec: {{taskType: echo, data: {{type: list, items: ["${{archive}}"]}}}}
"""))
    run.run("train", _trained("tsk-train"))

    spec, _ = _dispatch(run, "use")

    assert spec["data"]["items"] == [_file("tsk-train")]


def test_a_root_whole_read_of_an_unnamed_stage_stays_refused() -> None:
    with pytest.raises(CompileError, match="train"):
        _Run().runtime.validate(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: use
        dependsOn: [train]
        spec: {{taskType: echo, data: {{type: list, items: ["${{train}}"]}}}}
"""))


_CARRY_BODY = """
    templates:
      - name: body
        inputs: [{name: state, role: carried}]
        nodes:
          - name: step
            dependsOn: [{node: $ingress, port: state, input: state}]
            spec: {taskType: echo, data: {type: list, items: ["${state}"]}}
          - name: route
            dependsOn: [{node: step, input: input}]
            region:
              kind: branch
              inputs: [{name: input}]
              outputs: [{name: again}, {name: done}]
              selection: {input: input, field: [route]}
        edges:
          - from: {node: route, port: again}
            to: {node: $feedback, port: state}
            project: [final_lora_archive]
          - from: {node: route, port: done}
            to: {node: $egress, port: state}
"""

_CARRY = f"""
      - name: seed
        spec: {_ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          carried: [{{name: state}}]
"""


@pytest.mark.anyio
async def test_a_fed_back_file_reads_as_its_producers_url() -> None:
    run = await _Run().start(_workflow(_CARRY, _CARRY_BODY))
    run.run("seed", {"v": 0})
    run.run("step", _trained("tsk-step", route="again"))

    spec, _ = _dispatch(run, "step")

    assert spec["data"]["items"] == [_file("tsk-step")]


_CONCAT = f"""
      - name: left
        spec: {_ECHO}
      - name: right
        spec: {_ECHO}
      - name: both
        dependsOn: [{{node: left, input: l}}, {{node: right, input: r}}]
        region: {{kind: merge, combination: concat}}
      - name: use
        dependsOn: [{{node: both, input: all}}]
        spec:
          taskType: echo
          data:
            type: list
            items:
              - "${{all.0.value.final_lora_archive}}"
              - "${{all.1.value.final_lora_archive}}"
"""


@pytest.mark.anyio
async def test_each_aggregate_members_file_reads_as_its_own_producers_url() -> None:
    run = await _Run().start(_workflow(_CONCAT))
    run.run("left", _trained("tsk-left"))
    run.run("right", _trained("tsk-right"))

    spec, message = _dispatch(run, "use")

    assert spec["data"]["items"] == [_file("tsk-left"), _file("tsk-right")]
    upstream = message.task.spec.upstreamResults or {}
    expr = "all.value.final_lora_archive"
    assert [
        maybe_resolve_artifact_ref(item, upstream, "all", item_steps(expr, upstream, i))
        for i, item in enumerate(_worker_reads(message, expr))
    ] == [_file("tsk-left"), _file("tsk-right")]
    assert DataMixin()._extract_source_data_ids(message.task.spec) == [
        "tsk-left",
        "tsk-right",
    ]


@pytest.mark.anyio
async def test_an_agent_reading_a_projected_file_reads_its_url() -> None:
    run = await _Run().start(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: reader
        dependsOn: [{{node: train, input: x, project: [final_lora_archive]}}]
        spec:{_AGENT}
"""))
    run.run("train", _trained("tsk-train"))

    assert _member_values(run, _ready(run, "reader")) == {"x": [_file("tsk-train")]}
