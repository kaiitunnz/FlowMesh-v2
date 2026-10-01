"""Placement on a worker whose GPU another tenant holds, run against a real Redis.

The worker's held reading reaches the registry through its heartbeat, and the
dispatcher places each dispatch shape on the registry's view of it. Setting
``FLOWMESH_TEST_REDIS_URL`` points these at a Redis whose worker keys they own.
"""

import logging
import os
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import redis

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.dispatcher.base import Dispatcher
from server.registries.worker import Worker, WorkerRegistry
from server.services.monitoring import EventMonitor
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerCapabilities
from shared.tasks import TaskEnvelope, TaskEnvelopeStrict
from shared.tasks.specs import InferenceEmbodimentKind
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import GpuInfo
from tests.server.dispatch_helpers import resolved_embodiment
from tests.server.registries.test_worker_status_fence import _Rds, _Sync
from tests.server.task.test_v2_embodiment_fence import (
    LOCAL_ELIGIBLE,
    _runtime,
    _upstream_task,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.worker.factories import make_worker_hardware

_LIVE_URL = os.getenv("FLOWMESH_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not _LIVE_URL, reason="FLOWMESH_TEST_REDIS_URL is not set"
)

_WORKER = "wkr-gpu-held"
_HELD = "GPU-held"
_FREE = "GPU-free"

_ECHO = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: cpu}
spec:
  graph:
    nodes:
      - name: gen
        spec:
          taskType: echo
          data: {type: list, items: ["hello"]}
"""


class _LiveSync(_Sync):
    def ttl(self, key: str) -> int:
        return cast(int, self._client.ttl(key))


class _LiveRds(_Rds):
    def __init__(self, client: redis.Redis) -> None:
        self.sync = _LiveSync(client)


class _SpyRegistry(WorkerRegistry):
    """The registry, remembering which workers each idle-pool read offered."""

    def __init__(self, rds: Any) -> None:
        super().__init__(rds)
        self.offered: list[list[str]] = []

    def idle_satisfying_pool(
        self, task: TaskEnvelope, relays_only: bool
    ) -> list[Worker]:
        pool = super().idle_satisfying_pool(task, relays_only)
        self.offered.append([worker.id for worker in pool])
        return pool


def _live_worker(devices: list[GpuInfo]) -> Iterator[redis.Redis]:
    assert _LIVE_URL is not None
    live = redis.Redis.from_url(_LIVE_URL, decode_responses=True)
    keys = (worker_key(_WORKER), worker_hb_key(_WORKER))
    live.delete(*keys)
    live.sadd(WORKERS_SET_KEY, _WORKER)
    hardware = make_worker_hardware(devices)
    capabilities = WorkerCapabilities(
        supported_task_types=frozenset(
            {TaskType.INFERENCE, TaskType.ECHO, TaskType.SSH}
        ),
        ssh_noninteractive=True,
    )
    live.hset(
        worker_key(_WORKER),
        mapping={
            "id": _WORKER,
            "namespace": "ns",
            "cluster": "c",
            "node_id": "nde-1",
            "node_alias": "node",
            "incarnation": "1",
            "status": "IDLE",
            "hardware_json": hardware.model_dump_json(),
            "capabilities_json": capabilities.model_dump_json(),
        },
    )
    live.setex(worker_hb_key(_WORKER), 120, "ts")
    yield live
    live.delete(*keys)
    live.srem(WORKERS_SET_KEY, _WORKER)
    live.close()


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    yield from _live_worker(
        [GpuInfo(index=0, name="NVIDIA L4", uuid=_HELD, memory_total_bytes=24 << 30)]
    )


@pytest.fixture
def two_card_client() -> Iterator[redis.Redis]:
    yield from _live_worker(
        [
            GpuInfo(index=0, name="NVIDIA L4", uuid=_HELD, memory_total_bytes=24 << 30),
            GpuInfo(index=1, name="NVIDIA L4", uuid=_FREE, memory_total_bytes=24 << 30),
        ]
    )


@pytest.fixture
def cpu_client() -> Iterator[redis.Redis]:
    yield from _live_worker([])


def _report(registry: WorkerRegistry, availability: dict[str, Any]) -> None:
    """Deliver a heartbeat carrying ``availability`` through the event monitor."""
    monitor = EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("gpu-availability-placement"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=registry,
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
    )
    monitor._handle_worker_event(
        WorkerEvent(
            type="HEARTBEAT",
            worker_id=_WORKER,
            payload={"ttl_sec": 120},
            metrics={"gpu_availability": availability},
        )
    )


def _held(client: redis.Redis) -> _SpyRegistry:
    registry = _SpyRegistry(_LiveRds(client))
    _report(registry, {_HELD: {"available": False, "free_bytes": 0}})
    return registry


def _dispatcher(
    runtime: TaskRuntime, registry: WorkerRegistry, **kwargs: Any
) -> tuple[Dispatcher, list[tuple[str, dict[str, Any]]]]:
    requeued: list[tuple[str, dict[str, Any]]] = []

    class _Recording(Dispatcher):
        def requeue_task(self, task_id: str, **kw: Any) -> Any:
            requeued.append((task_id, kw))
            return super().requeue_task(task_id, **kw)

    return (
        _Recording(
            runtime=runtime,
            worker_registry=registry,
            logger=logging.getLogger("gpu-availability-placement"),
            **kwargs,
        ),
        requeued,
    )


class TestTheHeldReadingReachesTheRegistry:
    def test_a_held_device_is_withheld_and_a_cleared_one_returns(
        self, client: redis.Redis
    ) -> None:
        registry = _held(client)
        worker = registry.get_worker(_WORKER)
        assert worker is not None and worker.hardware is not None
        assert worker.hardware.gpu.devices[0].gpu_available is False

        # A heartbeat that offers no opinion leaves the reading in place.
        monitor_registry = WorkerRegistry(cast(Any, _LiveRds(client)))
        _report_without_availability(monitor_registry)
        worker = registry.get_worker(_WORKER)
        assert worker is not None and worker.hardware is not None
        assert worker.hardware.gpu.devices[0].gpu_available is False

        # An empty map is a worker whose probe failed: the reading clears.
        _report(registry, {})
        worker = registry.get_worker(_WORKER)
        assert worker is not None and worker.hardware is not None
        assert worker.hardware.gpu.devices[0].gpu_available is None

    def test_a_reading_from_an_unregistered_worker_writes_nothing(
        self, client: redis.Redis
    ) -> None:
        client.srem(WORKERS_SET_KEY, _WORKER)
        _report(
            WorkerRegistry(cast(Any, _LiveRds(client))), {_HELD: {"available": False}}
        )
        assert client.hget(worker_key(_WORKER), "gpu_availability_json") is None


def _report_without_availability(registry: WorkerRegistry) -> None:
    EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("gpu-availability-placement"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=registry,
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
    )._handle_worker_event(
        WorkerEvent(type="HEARTBEAT", worker_id=_WORKER, payload={"ttl_sec": 120})
    )


class TestPlacementOnAHeldWorker:
    @pytest.mark.anyio
    async def test_a_local_gpu_leaf_waits_rather_than_failing(
        self, client: redis.Redis
    ) -> None:
        registry = _held(client)
        runtime = _runtime(FakeRegistry())
        _wfl, ids = await _register(
            runtime, LOCAL_ELIGIBLE.replace("PRIMARY", "self_contained")
        )
        task_id = ids["gen"]
        dispatcher, requeued = _dispatcher(runtime, registry)

        assert dispatcher.dispatch_once(task_id) is False

        # The worker still could run it, so the leaf waits for the card, holding the
        # embodiment it was bound to.
        assert registry.offered == [[]]
        assert [kw["reason"] for _, kw in requeued] == ["no_idle_worker"]
        assert [w.id for w in registry.satisfying_workers(_task(runtime, task_id))] == [
            _WORKER
        ]
        resolved = resolved_embodiment(runtime, task_id)
        assert resolved is not None
        assert resolved.kind is InferenceEmbodimentKind.SELF_CONTAINED

    @pytest.mark.anyio
    async def test_cpu_work_still_lands_on_the_held_worker(
        self, client: redis.Redis
    ) -> None:
        registry = _held(client)
        runtime = _runtime(FakeRegistry())
        _wfl, ids = await _register(runtime, _ECHO)
        dispatcher, _ = _dispatcher(runtime, registry)

        dispatcher.dispatch_once(ids["gen"])

        assert registry.offered == [[_WORKER]]

    @pytest.mark.anyio
    async def test_a_resident_served_menu_dispatch_lands_on_the_held_worker(
        self, client: redis.Redis
    ) -> None:
        registry = _held(client)
        runtime = _runtime(FakeRegistry())
        _wfl, ids = await _register(
            runtime, LOCAL_ELIGIBLE.replace("PRIMARY", "resident_served")
        )
        task_id = ids["gen"]
        dispatcher, _ = _dispatcher(runtime, registry, resident_capacity_enabled=True)

        dispatcher.dispatch_once(task_id)

        resolved = resolved_embodiment(runtime, task_id)
        assert resolved is not None
        assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED
        assert registry.offered == [[_WORKER]]

    @pytest.mark.anyio
    async def test_a_pinned_resident_leaf_lands_on_the_held_worker(
        self, client: redis.Redis
    ) -> None:
        # No menu, so no resolved embodiment: only its service episode says it loads
        # no model on the worker.
        registry = _held(client)
        runtime = _runtime(FakeRegistry())
        task_id = await _upstream_task(runtime, service="{mode: resident}")
        assert runtime.embodiment_menu(task_id) is None
        assert runtime.service_episode_dispatch(task_id) is not None
        dispatcher, _ = _dispatcher(runtime, registry, resident_capacity_enabled=True)

        dispatcher.dispatch_once(task_id)

        assert registry.offered == [[_WORKER]]

    @pytest.mark.anyio
    async def test_an_input_preparation_lands_on_the_held_worker(
        self, client: redis.Redis
    ) -> None:
        registry = _held(client)
        runtime = _runtime(FakeRegistry())
        task_id = await _upstream_task(runtime, max_items=None)
        assert runtime.prepares_inputs(task_id) is True
        dispatcher, _ = _dispatcher(runtime, registry)

        dispatcher.dispatch_once(task_id)

        assert registry.offered == [[_WORKER]]


class TestPlacementBesideAHeldDevice:
    """A worker with one held card and one free one."""

    @pytest.mark.anyio
    async def test_a_model_leaf_waits_while_any_device_of_its_worker_is_held(
        self, two_card_client: redis.Redis
    ) -> None:
        # The model would see both devices, so the free one cannot carry it.
        registry = _held(two_card_client)
        runtime = _runtime(FakeRegistry())
        _wfl, ids = await _register(
            runtime, LOCAL_ELIGIBLE.replace("PRIMARY", "self_contained")
        )
        dispatcher, requeued = _dispatcher(runtime, registry)

        assert dispatcher.dispatch_once(ids["gen"]) is False

        assert registry.offered == [[]]
        assert [kw["reason"] for _, kw in requeued] == ["no_idle_worker"]

    def test_an_ssh_session_still_places_on_the_free_device(
        self, two_card_client: redis.Redis
    ) -> None:
        registry = _held(two_card_client)
        session = TaskEnvelopeStrict.model_validate(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "spec": {
                    "taskType": "ssh",
                    "interactive": False,
                    "image": "x",
                    "command": ["true"],
                    "resources": {"hardware": {"gpu": {"count": 1}}},
                },
            }
        )

        pool = registry.idle_satisfying_pool(session, False)

        assert [worker.id for worker in pool] == [_WORKER]


class TestRelayingDispatchOnACpuWorker:
    """A dispatch that runs no model needs no accelerator its leaf declares."""

    @pytest.mark.anyio
    async def test_a_pinned_resident_leaf_declaring_a_gpu_places_on_a_cpu_worker(
        self, cpu_client: redis.Redis
    ) -> None:
        registry = _SpyRegistry(cast(Any, _LiveRds(cpu_client)))
        runtime = _runtime(FakeRegistry())
        task_id = await _upstream_task(runtime, service="{mode: resident}")
        declared = _task(runtime, task_id).spec.resources
        assert declared is not None and declared.hardware is not None
        assert declared.hardware.gpu is not None and declared.hardware.gpu.count == 1
        dispatcher, requeued = _dispatcher(
            runtime, registry, resident_capacity_enabled=True
        )

        dispatcher.dispatch_once(task_id)

        assert registry.offered == [[_WORKER]]
        assert all(kw["reason"] != "no_eligible_worker" for _, kw in requeued)

    @pytest.mark.anyio
    async def test_a_resident_served_menu_dispatch_places_on_a_cpu_worker(
        self, cpu_client: redis.Redis
    ) -> None:
        registry = _SpyRegistry(cast(Any, _LiveRds(cpu_client)))
        runtime = _runtime(FakeRegistry())
        _wfl, ids = await _register(
            runtime, LOCAL_ELIGIBLE.replace("PRIMARY", "resident_served")
        )
        dispatcher, _ = _dispatcher(runtime, registry, resident_capacity_enabled=True)

        dispatcher.dispatch_once(ids["gen"])

        assert registry.offered == [[_WORKER]]

    @pytest.mark.anyio
    async def test_an_input_preparation_places_on_a_cpu_worker(
        self, cpu_client: redis.Redis
    ) -> None:
        registry = _SpyRegistry(cast(Any, _LiveRds(cpu_client)))
        runtime = _runtime(FakeRegistry())
        task_id = await _upstream_task(runtime, max_items=None)
        dispatcher, _ = _dispatcher(runtime, registry)

        dispatcher.dispatch_once(task_id)

        assert registry.offered == [[_WORKER]]


def _task(runtime: TaskRuntime, task_id: str) -> TaskEnvelope:
    record = runtime.get_record(task_id)
    assert record is not None
    return record.task
