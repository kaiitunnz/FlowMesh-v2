"""A dispatch its live worker keeps reporting it does not hold resolves as a lost
dispatch once the worker has disowned it for as long as a silent worker takes to be
declared dead, run against a real Redis where the registry is involved.

Setting ``FLOWMESH_TEST_REDIS_URL`` points the live tests at a Redis whose worker keys
they own; the runtime-only tests always run.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any, cast
from unittest.mock import Mock

import pytest
import redis

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.dispatcher.base import Dispatcher
from server.registries.worker import WorkerRegistry
from server.services.monitoring import EventMonitor
from server.task.models import EventEffect, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus
from tests.server.dispatch_helpers import record_dispatch
from tests.server.registries.test_worker_status_fence import _Rds
from tests.server.services.test_task_event_fence import _ECHO, _worker
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
)
from tests.server.task.test_unrequested_cancel import SSH_THEN_ECHO
from tests.server.task.test_v2_orchestration import LINEAR, FakeRegistry, _drain
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import _runtime as _runtime_v2
from tests.server.task.test_v2_orchestration import _worker as _worker_v2

_LIVE_URL = os.getenv("FLOWMESH_TEST_REDIS_URL")

_live = pytest.mark.skipif(not _LIVE_URL, reason="FLOWMESH_TEST_REDIS_URL is not set")

_WORKER = "wkr-disown"
_OTHER = "wkr-other"
_TTL_SEC = 120
_GRACE_SEC = 10
# Longer than the heartbeat TTL plus the watchdog grace.
_PAST_BOUND_SEC = _TTL_SEC + _GRACE_SEC + 5


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    assert _LIVE_URL is not None
    live = redis.Redis.from_url(_LIVE_URL, decode_responses=True)
    for worker_id in (_WORKER, _OTHER):
        live.delete(worker_key(worker_id), worker_hb_key(worker_id))
        live.sadd(WORKERS_SET_KEY, worker_id)
        live.hset(worker_key(worker_id), mapping={"status": "IDLE"})
        live.setex(worker_hb_key(worker_id), 120, "ts")
    yield live
    for worker_id in (_WORKER, _OTHER):
        live.delete(worker_key(worker_id), worker_hb_key(worker_id))
        live.srem(WORKERS_SET_KEY, worker_id)
    live.close()


def _dispatched(
    client: redis.Redis, workers: tuple[str, ...] = (_WORKER,)
) -> tuple[TaskRuntime, EventMonitor, Dispatcher, str, str]:
    """A v1 task dispatched to the live worker, with the monitor handling its events."""
    registry = WorkerRegistry(cast(Any, _Rds(client)))
    runtime = _runtime(_Registry(), registry)
    workflow_id, _ = asyncio.run(_register(runtime, _ECHO))
    task_id = _next(runtime)
    pool = Mock()
    pool.idle_satisfying_pool.return_value = [_worker(workers[0])]
    pool.satisfying_workers.return_value = [_worker(w) for w in workers]
    pool.reserve_worker.side_effect = registry.reserve_worker
    pool.release_worker.side_effect = registry.release_worker
    pool.publish_task.return_value = 1
    dispatcher = Dispatcher(
        runtime,
        pool,
        logging.getLogger("disowned"),
        enable_task_merge=False,
        no_worker_grace_sec=0,
    )
    monitor = _monitor(runtime, dispatcher)
    monitor._worker_registry = registry
    watchdog = cast(Any, monitor._watchdog)
    watchdog.death_bound_sec.side_effect = lambda ttl: ttl + _GRACE_SEC
    dispatcher.dispatch_once(task_id)
    dispatch_id = runtime._tasks[task_id].dispatch_id
    assert dispatch_id is not None
    return runtime, monitor, dispatcher, workflow_id, task_id


def _heartbeat(
    status: WorkerStatus | None, dispatch_id: str | None = "dsp-earlier"
) -> WorkerEvent:
    return WorkerEvent(
        type="HEARTBEAT",
        worker_id=_WORKER,
        status=status,
        dispatch_id=dispatch_id if status is not None else None,
        payload={"ttl_sec": _TTL_SEC},
    )


def _age(runtime: TaskRuntime, task_id: str, seconds: float) -> None:
    record = runtime._tasks[task_id]
    assert record.dispatched_ts is not None
    record.dispatched_ts -= seconds


def _state(client: redis.Redis, worker_id: str = _WORKER) -> dict[str, str]:
    held = cast(dict[str, str], client.hgetall(worker_key(worker_id)))
    return {k: v for k, v in held.items() if k in ("status", "reserved_dispatch")}


@_live
def test_a_disowned_dispatch_returns_after_the_bound_and_frees_its_worker(
    client: redis.Redis,
) -> None:
    runtime, monitor, _, _, task_id = _dispatched(client)
    dispatch_id = runtime._tasks[task_id].dispatch_id

    _age(runtime, task_id, _TTL_SEC)
    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))
    assert runtime._tasks[task_id].status == TaskStatus.DISPATCHED
    assert _state(client) == {"status": "BUSY", "reserved_dispatch": str(dispatch_id)}

    _age(runtime, task_id, _PAST_BOUND_SEC - _TTL_SEC)
    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.failed_workers == [_WORKER]
    assert _state(client) == {"status": "IDLE"}


@_live
def test_a_first_dispatch_disowned_with_no_earlier_one_resolves(
    client: redis.Redis,
) -> None:
    runtime, monitor, _, _, task_id = _dispatched(client)
    _age(runtime, task_id, _PAST_BOUND_SEC)

    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE, None))

    assert runtime._tasks[task_id].status == TaskStatus.PENDING


@_live
@pytest.mark.parametrize(
    "report",
    [
        _heartbeat(None),
        WorkerEvent(
            type="STATUS", worker_id=_WORKER, status=WorkerStatus.IDLE, payload={}
        ),
        "busy",
    ],
    ids=["unversioned_heartbeat", "unversioned_status", "busy"],
)
def test_a_report_that_is_no_disowning_heartbeat_never_resolves(
    client: redis.Redis, report: Any
) -> None:
    runtime, monitor, _, _, task_id = _dispatched(client)
    _age(runtime, task_id, _PAST_BOUND_SEC)
    if report == "busy":
        report = _heartbeat(WorkerStatus.BUSY, runtime._tasks[task_id].dispatch_id)

    monitor._handle_worker_event(report)

    assert runtime._tasks[task_id].status == TaskStatus.DISPATCHED


@_live
def test_a_late_start_of_a_resolved_dispatch_is_fenced(client: redis.Redis) -> None:
    runtime, monitor, _, _, task_id = _dispatched(client)
    dispatch_id = runtime._tasks[task_id].dispatch_id
    _age(runtime, task_id, _PAST_BOUND_SEC)
    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))

    effect = runtime.mark_started(task_id, _WORKER, {}, "ts", dispatch_id)

    assert effect is EventEffect.STALE
    assert runtime._tasks[task_id].status == TaskStatus.PENDING


@_live
def test_a_disowned_dispatch_being_cancelled_settles_cancelled(
    client: redis.Redis,
) -> None:
    runtime, monitor, _, workflow_id, task_id = _dispatched(client)
    runtime.cancel_workflow(workflow_id)
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING
    _age(runtime, task_id, _PAST_BOUND_SEC)

    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert runtime.workflow_settlement(workflow_id).settled
    assert _state(client) == {"status": "IDLE"}


@_live
def test_the_retry_goes_to_another_worker(client: redis.Redis) -> None:
    runtime, monitor, dispatcher, _, task_id = _dispatched(client, (_WORKER, _OTHER))
    _age(runtime, task_id, _PAST_BOUND_SEC)
    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))
    pool = cast(Any, dispatcher._worker_registry)
    pool.idle_satisfying_pool.return_value = [_worker(_WORKER), _worker(_OTHER)]

    assert _next(runtime) == task_id
    dispatcher.dispatch_once(task_id)

    assert runtime._tasks[task_id].assigned_worker == _OTHER


@_live
def test_the_only_eligible_worker_disowning_its_task_fails_it(
    client: redis.Redis,
) -> None:
    runtime, monitor, dispatcher, workflow_id, task_id = _dispatched(client)
    _age(runtime, task_id, _PAST_BOUND_SEC)
    monitor._handle_worker_event(_heartbeat(WorkerStatus.IDLE))

    assert _next(runtime) == task_id
    dispatcher.dispatch_once(task_id)

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.FAILED
    assert record.error is not None and "does not hold dispatch" in record.error
    assert runtime.workflow_settlement(workflow_id).settled


def _v2_disowned(workflow: str, node: str) -> tuple[TaskRuntime, str, dict[str, str]]:
    runtime = _runtime_v2(FakeRegistry())
    workflow_id, ids = asyncio.run(_register_v2(runtime, workflow))
    task_id = ids[node]
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker_v2()), "dsp-1")
    return runtime, workflow_id, ids


def _resolve(runtime: TaskRuntime, task_id: str) -> Callable[[], Any]:
    return lambda: runtime.resolve_disowned_dispatch(task_id, "dsp-1", "wkr-1", 0)


def test_a_disowned_replayable_v2_task_reruns_under_its_invocation() -> None:
    runtime, workflow_id, ids = _v2_disowned(LINEAR, "a")
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(ids["a"])
    assert work_item is not None
    invocation = work_item.invocation_id

    outcome = _resolve(runtime, ids["a"])()

    assert outcome is not None and outcome.effect is EventEffect.RETURNED
    record = runtime.get_record(ids["a"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert record.failed_workers == ["wkr-1"]
    assert work_item.invocation_id == invocation
    assert _drain(runtime) == [ids["a"], ids["b"], ids["c"]]
    assert runtime.workflow_settlement(workflow_id).settled


def test_a_disowned_external_effect_fails_with_its_dependent() -> None:
    runtime, workflow_id, ids = _v2_disowned(SSH_THEN_ECHO, "session")

    outcome = _resolve(runtime, ids["session"])()

    assert outcome is not None and outcome.effect is EventEffect.FAILED
    assert [task for task, _ in outcome.impacted] == [ids["after"]]
    session = runtime.get_record(ids["session"])
    after = runtime.get_record(ids["after"])
    assert session is not None and session.status == TaskStatus.FAILED
    assert after is not None and after.status == TaskStatus.FAILED
    assert runtime.workflow_settlement(workflow_id).settled


def test_a_dispatch_an_event_applied_to_never_resolves() -> None:
    runtime, _, ids = _v2_disowned(LINEAR, "a")
    runtime.mark_started(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z", "dsp-1")

    assert _resolve(runtime, ids["a"])() is None
