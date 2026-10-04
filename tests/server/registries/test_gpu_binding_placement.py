"""A worker that binds a task type's executor is placed on its free devices."""

import json

import fakeredis
import pytest

from server.registries.worker import WorkerRegistry
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import GpuInfo
from tests.server.redis_helpers import fake_redis_client
from tests.worker.factories import make_worker_hardware

_TASK = {
    "apiVersion": "flowmesh/v1",
    "kind": "Inference",
    "metadata": {"name": "infer"},
    "spec": {
        "taskType": "inference",
        "data": {"type": "list", "items": ["hi"]},
        "model": {"source": {"identifier": "org/m"}, "vllm": {"dtype": "auto"}},
        "resources": {"hardware": {"gpu": {"count": 1}}},
    },
}


def _registry(binding: frozenset[TaskType]) -> WorkerRegistry:
    """One idle worker with two GPUs, the first held by a foreign process."""
    registry = WorkerRegistry(fake_redis_client(fakeredis.FakeServer()))
    devices = [
        GpuInfo(index=i, name="A100", uuid=f"GPU-{i}", memory_total_bytes=80 * 1024**3)
        for i in range(2)
    ]
    capabilities = WorkerCapabilities(
        supported_task_types=frozenset({TaskType.INFERENCE}),
        gpu_binding_task_types=binding,
    )
    worker_id = registry.register_worker(
        "nde-1",
        "node",
        {
            "alias": "w",
            "namespace": "ns",
            "cluster": "c",
            "status": WorkerStatus.IDLE.value,
            "capabilities_json": capabilities.model_dump_json(),
            "hardware_json": make_worker_hardware(devices).model_dump_json(),
            "gpu_availability_json": json.dumps(
                {"GPU-0": {"available": False, "free_bytes": 0}}
            ),
        },
    )
    registry.update_worker_hb(worker_id, "2026-01-01T00:00:00Z", 60)
    return registry


@pytest.mark.parametrize(
    ("binding", "placed"),
    [(frozenset({TaskType.INFERENCE}), True), (frozenset(), False)],
    ids=["binds", "in-process"],
)
def test_a_held_card_withholds_the_worker_only_when_it_does_not_bind(
    binding: frozenset[TaskType], placed: bool
) -> None:
    task = TaskEnvelopeStrict.model_validate(_TASK)

    pool = _registry(binding).idle_satisfying_pool(task, False)

    assert bool(pool) is placed
