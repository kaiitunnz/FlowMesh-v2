"""A resident leaf's serving size reaches its family, its replica, and placement."""

import asyncio
import json
import time
from typing import Any

import pytest

from server.registries.worker import Worker, gpu_available_for, hw_satisfies
from server.resident import ResidentSnapshot, ResidentStores, ServiceFamily
from server.resident.state import ClaimState
from server.schemas.resident import ResidentFamilyInfo
from server.task.v2.representations.serving_size import (
    DEFAULT_SERVING_SIZE,
    ServingSize,
)
from shared.schemas.worker import WorkerCapabilities
from shared.tasks import TaskType
from shared.tasks.specs.common import ModelSpecStrict, ModelSpecTemplate
from tests.server.registries.test_worker_registry import _worker
from tests.server.resident.node_harness import Node
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_orchestration import _register
from tests.support.waiting import until

_SIZED_LEAF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: sized}
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {max_model_len: 1024, tensor_parallel_size: 2}
          service: {mode: resident}
          resources: {hardware: {cpu: 4, memory: 8Gi, gpu: {count: 2, type: H100}}}
          data: {type: list, items: [hi]}
"""
_SIZE = ServingSize(
    cpu=4,
    memory_bytes=8 * 1024**3,
    gpu_type="h100",
    gpu_count=2,
    tensor_parallel_size=2,
)
_SERVE_BINDS = WorkerCapabilities(
    supported_task_types=frozenset({TaskType.SERVE}),
    gpu_binding_task_types=frozenset({TaskType.SERVE}),
)


async def _materialized(node: Node) -> tuple[ServiceFamily, str]:
    _workflow_id, ids = await _register(node.runtime, _SIZED_LEAF)
    binding = node.runtime.resolve_service_dependency(ids["a"])
    assert binding is not None
    node.control._ensure_family(binding.dependency, binding)
    family = node.control.stores.families.get(binding.dependency.service_family)
    assert family is not None
    replica = await node.control._lifecycle.materialize(family)
    assert replica.serve_task_id is not None
    return family, replica.serve_task_id


@pytest.mark.parametrize(("substrate", "gpu_count"), [("serve", 2), ("dev_model", 0)])
def test_a_pinned_leaf_size_reaches_its_family_and_replica(
    substrate: str, gpu_count: int
) -> None:
    async def run() -> None:
        node = Node(substrate=substrate)
        family, serve_task_id = await _materialized(node)
        assert family.serving_size == _SIZE

        record = node.runtime.get_record(serve_task_id)
        assert record is not None
        spec = record.task.spec
        hardware = spec.resources.hardware if spec.resources is not None else None
        assert hardware is not None and hardware.gpu is not None
        assert (hardware.cpu, hardware.memory) == (4, "8Gi")
        assert hardware.gpu.count == gpu_count
        assert isinstance(spec, (ModelSpecStrict, ModelSpecTemplate))
        model = spec.model
        assert model is not None and model.vllm is not None
        assert model.vllm["tensor_parallel_size"] == 2
        assert model.vllm["max_model_len"] == 1024

    asyncio.run(run())


def test_a_sized_replica_places_only_where_one_worker_holds_its_devices() -> None:
    async def run() -> None:
        node = Node(substrate="serve")
        _family, serve_task_id = await _materialized(node)
        record = node.runtime.get_record(serve_task_id)
        assert record is not None
        task = record.task

        def worker(gpus: int, name: str = "H100 80GB HBM3") -> Worker:
            return _worker(
                gpu_count=gpus,
                gpu_name=name,
                cpu_cores=8,
                sys_mem=64 * 1024**3,
                capabilities=_SERVE_BINDS,
            )

        assert hw_satisfies(worker(2), task)
        assert not hw_satisfies(worker(1), task)
        assert not hw_satisfies(worker(2, "A100-SXM4-80GB"), task)
        # A device something outside the fleet holds leaves too few free: the replica
        # waits for it rather than counting the worker out.
        held = worker(2)
        assert held.hardware is not None
        held.hardware.gpu.devices[1].gpu_available = False
        assert hw_satisfies(held, task)
        assert not gpu_available_for(held, task, relays_only=False)
        assert gpu_available_for(worker(3), task, relays_only=False)

    asyncio.run(run())


def test_registration_never_resizes_a_live_family() -> None:
    stores = ResidentStores()
    live = ServiceFamily(family="f", engine_batch_key="f", model_ref="m")
    stores.families.register(live)

    stores.families.register(
        live.model_copy(update={"serving_size": _SIZE, "warmth": "warm"})
    )

    assert stores.families.get("f") == live


def test_a_family_stored_without_a_size_restores_at_the_default() -> None:
    stores = ResidentStores()
    stores.families.register(
        ServiceFamily(family="sized", engine_batch_key="k", model_ref="m").model_copy(
            update={"serving_size": _SIZE}
        )
    )
    blob = json.loads(stores.to_snapshot().model_dump_json())
    blob["families"].append(
        {"family": "old", "engine_batch_key": "k", "model_ref": "m"}
    )

    restored = ResidentStores()
    restored.load_snapshot(ResidentSnapshot.model_validate(blob))

    old, sized = restored.families.get("old"), restored.families.get("sized")
    assert old is not None and old.serving_size == DEFAULT_SERVING_SIZE
    assert sized is not None and sized.serving_size == _SIZE


def test_the_family_listing_reports_a_demand_family_size_only() -> None:
    demand = ServiceFamily(family="f", engine_batch_key="f", model_ref="m")
    standing = demand.model_copy(update={"standing": True})

    info = ResidentFamilyInfo.project(demand.model_copy(update={"serving_size": _SIZE}))
    assert info.serving_size is not None
    assert info.serving_size.model_dump() == _SIZE.model_dump()
    assert ResidentFamilyInfo.project(standing).serving_size is None


def test_a_cold_start_no_worker_can_host_is_denied_naming_its_size() -> None:
    async def run() -> None:
        node = Node(cold_start_deadline_sec=0.3)
        node.control.bind_loop(asyncio.get_running_loop())
        errors: list[str | None] = []
        settle = node.control._settle

        def recording_settle(*args: Any, error: str | None = None) -> bool:
            errors.append(error)
            return settle(*args, error=error)

        node.control._settle = recording_settle
        _, ids = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, ids["writer"])
        directory = node.control.stores.directory
        await until(lambda: any(r.serve_task_id for r in directory.all()))
        (cold,) = directory.all()
        assert cold.serve_task_id is not None
        record = node.runtime.get_record(cold.serve_task_id)
        assert record is not None
        # The dispatcher marks a task no worker's hardware satisfies.
        record.no_eligible_since = time.time()

        (claim,) = node.control.stores.claims.all()
        await until(lambda: claim.state is ClaimState.TERMINAL, timeout=5.0)

        assert not claim.holds_credit
        assert any(
            error is not None
            and "cold_start_budget" in error
            and f"no worker can host a replica of size {DEFAULT_SERVING_SIZE.key()}"
            in error
            for error in errors
        ), errors

    asyncio.run(run())
