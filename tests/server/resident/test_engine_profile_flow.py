"""A resident leaf's engine profile reaches its family and the replica serving it."""

import asyncio
import json

from tests.server.resident.node_harness import Node
from tests.server.task.test_v2_orchestration import _register

_LEAF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: profiled}
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {max_model_len: 1024, gpu_memory_utilization: 0.4}
          service: {mode: resident}
          resources: {hardware: {gpu: {count: 1}}}
          data: {type: list, items: [hi]}
"""


def test_a_resident_leaf_profile_reaches_its_family_and_replica() -> None:
    async def run() -> None:
        node = Node(substrate="serve")
        _workflow_id, ids = await _register(node.runtime, _LEAF)
        binding = node.runtime.resolve_service_dependency(ids["a"])
        assert binding is not None
        node.control._ensure_family(binding.dependency, binding)

        family = node.control.stores.families.get(binding.dependency.service_family)
        assert family is not None and family.engine_profile is not None
        assert json.loads(family.engine_profile) == {"max_model_len": 1024}

        replica = await node.control._lifecycle.materialize(family)
        assert replica.serve_task_id is not None
        record = node.runtime.get_record(replica.serve_task_id)
        assert record is not None
        model = record.task.spec.model  # type: ignore[union-attr]
        vllm = model.vllm if model is not None else None
        assert vllm is not None and vllm["max_model_len"] == 1024
        assert "gpu_memory_utilization" not in vllm

    asyncio.run(run())
