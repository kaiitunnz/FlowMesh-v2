"""A worker reserved for a dispatch is released when the dispatch ends, run against a
real Redis.

A worker that names no dispatch in its reports, such as one on an earlier version, is
fenced while reserved, so the end of the dispatch is what frees it. Setting
``FLOWMESH_TEST_REDIS_URL`` points these at a Redis whose worker keys they own.
"""

import asyncio
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any, cast
from unittest.mock import Mock

import pytest
import redis

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.dispatcher.base import Dispatcher
from server.orchestration import WorkItemStatus
from server.registries.worker import WorkerRegistry
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus
from tests.server.dispatch_helpers import record_dispatch
from tests.server.registries.test_worker_status_fence import _Rds
from tests.server.services.test_task_event_fence import _ECHO, _event, _worker
from tests.server.task.test_agent_episode_runtime import (
    _AGENT_WF,
    _HOLDER,
    _MODEL_HELD_SCRIPT,
    _TS,
)
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
)
from tests.server.task.test_v2_orchestration import FakeRegistry
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import _runtime as _v2_runtime
from worker.executors.harness.scripted import ScriptedHarnessAdapter

_LIVE_URL = os.getenv("FLOWMESH_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not _LIVE_URL, reason="FLOWMESH_TEST_REDIS_URL is not set"
)

_WORKER = "wkr-release"


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    assert _LIVE_URL is not None
    live = redis.Redis.from_url(_LIVE_URL, decode_responses=True)
    keys = (worker_key(_WORKER), worker_hb_key(_WORKER))
    live.delete(*keys)
    live.sadd(WORKERS_SET_KEY, _WORKER)
    live.hset(worker_key(_WORKER), mapping={"status": "IDLE"})
    live.setex(worker_hb_key(_WORKER), 120, "ts")
    yield live
    live.delete(*keys)
    live.srem(WORKERS_SET_KEY, _WORKER)
    live.close()


def _dispatched(
    client: redis.Redis, publish: Callable[..., int] = lambda *_: 1
) -> tuple[TaskRuntime, EventMonitor, str, dict[str, Any]]:
    """A task dispatched to the live worker, and the monitor handling its events."""
    registry = WorkerRegistry(cast(Any, _Rds(client)))
    runtime = _runtime(_Registry(), registry)
    workflow_id, _ = asyncio.run(_register(runtime, _ECHO))
    task_id = _next(runtime)
    dispatcher_registry = Mock()
    dispatcher_registry.idle_satisfying_pool.return_value = [_worker(_WORKER)]
    dispatcher_registry.satisfying_workers.return_value = [_worker(_WORKER)]
    dispatcher_registry.reserve_worker.side_effect = registry.reserve_worker
    dispatcher_registry.release_worker.side_effect = registry.release_worker
    dispatcher_registry.publish_task.side_effect = publish
    dispatcher = Dispatcher(
        runtime,
        dispatcher_registry,
        logging.getLogger("reservation"),
        enable_task_merge=False,
    )
    monitor = _monitor(runtime, dispatcher)
    monitor._worker_registry = registry
    context = {"runtime": runtime, "workflow_id": workflow_id}
    return runtime, monitor, task_id, context


def _state(client: redis.Redis) -> dict[str, str]:
    held = cast(dict[str, str], client.hgetall(worker_key(_WORKER)))
    return {k: v for k, v in held.items() if k in ("status", "reserved_dispatch")}


def _unversioned(status: WorkerStatus) -> WorkerEvent:
    return WorkerEvent(type="STATUS", worker_id=_WORKER, status=status, payload={})


def test_an_idle_naming_no_dispatch_frees_the_worker_once_its_task_settles(
    client: redis.Redis,
) -> None:
    runtime, monitor, task_id, _ = _dispatched(client)
    cast(Any, monitor._dispatcher).dispatch_once(task_id)
    # The IDLE lands before the success commits, while the dispatch is in flight.
    monitor._handle_worker_event(_unversioned(WorkerStatus.BUSY))
    monitor._handle_worker_event(_unversioned(WorkerStatus.IDLE))
    assert _state(client)["status"] == "BUSY"

    monitor.handle_task_event(_event("TASK_SUCCEEDED", runtime, task_id, _WORKER))

    assert runtime._tasks[task_id].status == TaskStatus.DONE
    assert _state(client) == {"status": "IDLE"}


def test_a_dispatch_cancelled_while_its_publish_fails_frees_the_worker(
    client: redis.Redis,
) -> None:
    context: dict[str, Any] = {}

    def publish(*_: Any) -> int:
        context["runtime"].cancel_workflow(context["workflow_id"])
        raise RuntimeError("publish failed")

    runtime, monitor, task_id, built = _dispatched(client, publish)
    context.update(built)

    cast(Any, monitor._dispatcher).dispatch_once(task_id)

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert _state(client) == {"status": "IDLE"}


def test_a_restart_releases_a_reservation_whose_dispatch_ended(
    client: redis.Redis,
) -> None:
    runtime, monitor, task_id, _ = _dispatched(client)
    cast(Any, monitor._dispatcher).dispatch_once(task_id)
    dispatch_id = runtime._tasks[task_id].dispatch_id
    registry = WorkerRegistry(cast(Any, _Rds(client)))

    runtime.release_ended_reservations()
    assert _state(client) == {"status": "BUSY", "reserved_dispatch": dispatch_id}

    # A root restarted after the task settled holds no dispatch for it.
    _runtime(_Registry(), registry).release_ended_reservations()
    assert _state(client) == {"status": "IDLE"}


async def _suspended(
    workflows: Any, registry: WorkerRegistry
) -> tuple[TaskRuntime, EventMonitor, str, str]:
    """An agent whose step on the live worker suspended on a model boundary."""
    runtime = _v2_runtime(workflows)
    runtime._worker_registry = registry
    monitor = _monitor(runtime)
    monitor._worker_registry = registry
    runtime.set_model_settler(lambda _envelope: None)
    workflow_id, ids = await _register_v2(runtime, _AGENT_WF)
    writer = ids["writer"]
    adapter = ScriptedHarnessAdapter(_MODEL_HELD_SCRIPT, "v1")
    dispatch = runtime.agent_episode_dispatch(writer, _HOLDER)
    assert dispatch is not None
    registry.reserve_worker(_WORKER, writer, "dsp-1")
    record_dispatch(runtime, writer, _WORKER, "dsp-1")
    monitor._handle_worker_event(_unversioned(WorkerStatus.BUSY))
    step = adapter.start(writer, capsule=None, outcomes=dispatch.delivered_outcomes)
    runtime.mark_succeeded(
        writer,
        _WORKER,
        {"agent_episode": step.model_dump(mode="json")},
        _TS,
        "dsp-1",
    )
    return runtime, monitor, workflow_id, writer


def test_a_worker_whose_agent_step_suspended_is_freed(client: redis.Redis) -> None:
    registry = WorkerRegistry(cast(Any, _Rds(client)))
    runtime, monitor, workflow_id, writer = asyncio.run(
        _suspended(_Registry(), registry)
    )
    monitor._handle_worker_event(_unversioned(WorkerStatus.IDLE))

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(writer)
    assert work_item is not None and work_item.status is WorkItemStatus.BLOCKED
    assert _state(client) == {"status": "IDLE"}


@pytest.mark.parametrize("freed_by", ["sweep", "idle"])
def test_a_suspended_worker_whose_release_failed_is_freed_after_a_restart(
    client: redis.Redis, freed_by: str
) -> None:
    registry = WorkerRegistry(cast(Any, _Rds(client)))
    workflows = FakeRegistry()
    release = registry.release_worker
    registry.release_worker = Mock(  # type: ignore[method-assign]
        side_effect=ConnectionError("redis unavailable")
    )
    asyncio.run(_suspended(workflows, registry))
    registry.release_worker = release  # type: ignore[method-assign]
    assert _state(client) == {"status": "BUSY", "reserved_dispatch": "dsp-1"}

    restored = _v2_runtime(workflows)
    restored._worker_registry = registry
    restored.set_model_settler(lambda _envelope: None)
    asyncio.run(restored.rehydrate())
    if freed_by == "sweep":
        restored.release_ended_reservations()
        # The release returns the worker to the BUSY it last reported.
        assert _state(client) == {"status": "BUSY"}
    else:
        monitor = _monitor(restored)
        monitor._worker_registry = registry
        monitor._handle_worker_event(_unversioned(WorkerStatus.IDLE))
        assert _state(client) == {"status": "IDLE"}
