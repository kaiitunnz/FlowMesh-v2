"""Worker eligibility reads every candidate's record and heartbeat in one round
trip, and both eligibility twins agree over the same registry state."""

from typing import Any, NoReturn

import fakeredis
import pytest

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.registries.worker import WorkerRegistry
from server.schemas.worker import WorkerCordon
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.task_type import TaskType
from tests.server.redis_helpers import fake_redis_client

_ECHO = WorkerCapabilities(supported_task_types=frozenset({TaskType.ECHO}))


def _task() -> TaskEnvelopeStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "spec": {"taskType": "echo", "resources": None},
        }
    )


def _seed(
    raw: Any,
    worker_id: str,
    alias: str,
    heartbeat: str = "ttl",
    status: WorkerStatus = WorkerStatus.IDLE,
    capabilities: WorkerCapabilities = _ECHO,
) -> None:
    raw.sadd(WORKERS_SET_KEY, worker_id)
    raw.hset(
        worker_key(worker_id),
        mapping={
            "alias": alias,
            "node_id": "nde-a",
            "node_alias": "node-a",
            "status": status.value,
            "capabilities_json": capabilities.model_dump_json(),
        },
    )
    if heartbeat == "ttl":
        raw.setex(worker_hb_key(worker_id), 120, "ts")
    elif heartbeat == "persistent":
        raw.set(worker_hb_key(worker_id), "ts")


@pytest.fixture
def registry() -> WorkerRegistry:
    server = fakeredis.FakeServer()
    raw: Any = fakeredis.FakeRedis(server=server, decode_responses=True)
    _seed(raw, "wkr-1", "alpha")
    _seed(raw, "wkr-2", "beta")
    _seed(raw, "wkr-3", "gamma", heartbeat="none")
    _seed(raw, "wkr-4", "delta", heartbeat="persistent")
    _seed(raw, "wkr-5", "epsilon", capabilities=WorkerCapabilities())
    _seed(raw, "wkr-6", "zeta", status=WorkerStatus.BUSY)
    _seed(raw, "wkr-7", "eta")
    raw.sadd(WORKERS_SET_KEY, "wkr-9")
    registry = WorkerRegistry(fake_redis_client(server))
    registry.set_cordon(WorkerCordon(node_alias="node-a", alias="beta"), True)
    return registry


def _no_per_worker_read(*_args: Any) -> NoReturn:
    raise AssertionError("eligibility read a worker on its own")


@pytest.fixture
def batched(registry: WorkerRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "get_worker",
        "get_worker_async",
        "is_worker_stale",
        "is_worker_stale_async",
    ):
        monkeypatch.setattr(registry, name, _no_per_worker_read)


_READS = [
    ("wkr-1", False),
    ("wkr-2", False),
    ("wkr-3", True),
    ("wkr-4", True),
    ("wkr-5", False),
    ("wkr-6", False),
    ("wkr-7", False),
]


@pytest.mark.anyio
async def test_the_read_pairs_each_registered_worker_with_its_staleness(
    registry: WorkerRegistry,
) -> None:
    ids = registry.get_worker_ids()
    sync = [(w.id, stale) for w, stale in registry._read_workers(ids)]
    async_ = [(w.id, stale) for w, stale in await registry._read_workers_async(ids)]
    assert sync == async_ == _READS


@pytest.mark.anyio
@pytest.mark.usefixtures("batched")
async def test_both_eligibility_twins_return_the_same_workers(
    registry: WorkerRegistry,
) -> None:
    sync = [w.id for w in registry.satisfying_workers(_task())]
    async_ = [w.id for w in await registry.satisfying_workers_async(_task())]
    assert sync == async_ == ["wkr-1", "wkr-6", "wkr-7"]


@pytest.mark.usefixtures("batched")
def test_the_idle_pool_reads_in_one_batch_and_drains_a_bound_cordoned_worker(
    registry: WorkerRegistry,
) -> None:
    assert [w.id for w in registry.idle_satisfying_pool(_task(), False)] == [
        "wkr-1",
        "wkr-7",
    ]
    pool = registry.idle_satisfying_pool(_task(), False, bound_worker_id="wkr-2")
    assert [w.id for w in pool] == ["wkr-1", "wkr-2", "wkr-7"]
