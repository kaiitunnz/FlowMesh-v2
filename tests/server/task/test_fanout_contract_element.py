"""A fan-out child of a contract leaf names its element in its contract.

The contract a child's embodiments resolve is proven from the leaf's model and sampling
and names one element of the producer's result, so it reaches feasibility and admission
as exactly one prompt without the element itself travelling with it.
"""

from typing import Any, cast

import pytest

from shared.inference import InferenceSourceKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _worker,
)

_SPAWN_OVER_MENU = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: spawn-menu}
spec:
  graph:
    nodes:
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: gen
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {gpu_memory_utilization: 0.9}
          data: {type: list, items: ["template"]}
          resources: {hardware: {gpu: {count: 1}}}
          service: {mode: local_eligible}
      - name: fanout
        dependsOn: [planner]
        region: {kind: spawn, child: gen, authority: {invoke: []}}
      - name: collect
        dependsOn: [fanout]
        region: {kind: join, completion: all_settled}
"""


@pytest.mark.anyio
async def test_a_child_of_a_menu_leaf_names_its_element_in_its_contract() -> None:
    runtime = _live_runtime(FakeRegistry())
    _workflow_id, ids = await _register(runtime, _SPAWN_OVER_MENU)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["first", "second"]), _TS
    )
    produced = runtime.result_binding(planner)
    assert produced is not None and produced.reference is not None

    children = _pop_ready(runtime)
    assert len(children) == 2
    indices: list[int] = []
    for child in children:
        assert runtime.embodiment_menu(child) is not None
        contract = runtime.declared_contract(child)
        assert contract is not None
        source = contract.source
        assert source.kind is InferenceSourceKind.UPSTREAM
        assert source.node == planner and source.path is None
        assert source.max_items == 1 and not source.prepared_before_selection
        assert source.element is not None
        wire = contract.model_dump_json()
        assert "first" not in wire and "second" not in wire
        element = runtime.input_element(child)
        assert element is not None
        assert element.reference == produced.reference
        assert element.element == source.element
        indices.append(source.element)
    assert sorted(indices) == [0, 1]
