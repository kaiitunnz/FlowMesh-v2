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
from server.registries.worker import WorkerRegistry
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus
from tests.server.registries.test_worker_status_fence import _Rds
from tests.server.services.test_task_event_fence import _ECHO, _event, _worker
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
)

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
