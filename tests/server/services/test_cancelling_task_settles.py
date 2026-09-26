"""A cancelling task settles CANCELLED whatever ends its dispatch."""

import logging
from unittest import mock

import pytest

from server.dispatcher.base import Dispatcher
from server.services.monitoring import EventMonitor
from server.services.watchdog import WorkerWatchdog
from server.task.models import DispatchEnd, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_task_merge import (
    _WORKER,
    _InterruptRecorder,
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
    _siblings,
)
from tests.server.task.test_v2_orchestration import _TS


def _failed(task_id: str, retryable: bool | None) -> TaskEvent:
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=_WORKER.id,
        error="worker_heartbeat_expired" if retryable is None else "boom",
        retryable=retryable,
        ts=_TS,
    )


async def _cancelling(registry: _Registry) -> tuple[TaskRuntime, str]:
    runtime = _runtime(registry, _InterruptRecorder())
    workflow_id, _ = await _register(runtime, _siblings(names=["a"]))
    task_id = _next(runtime)
    record_dispatch(runtime, task_id, _WORKER)
    runtime.cancel_workflow(workflow_id)
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING
    return runtime, task_id


def _assert_cancelled(
    registry: _Registry, runtime: TaskRuntime, metrics: mock.MagicMock, task_id: str
) -> None:
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert registry.durable_status(task_id) == TaskStatus.CANCELLED
    recorded = [call.args[0].type for call in metrics.record_task_event.call_args_list]
    assert recorded == ["TASK_CANCELLED"]
    metrics.finalize_task_cancellation.assert_called_once_with(task_id)
    metrics.finalize_task_failure.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "retryable", [True, None, False], ids=["retryable", "synthetic", "non-retryable"]
)
async def test_a_failure_of_a_cancelling_task_settles_it_cancelled(
    retryable: bool | None,
) -> None:
    registry = _Registry()
    runtime, task_id = await _cancelling(registry)
    monitor = _monitor(runtime)
    metrics = mock.MagicMock()
    monitor._metrics = metrics

    monitor._handle_task_event(_failed(task_id, retryable))

    _assert_cancelled(registry, runtime, metrics, task_id)
    assert runtime.workflow_settlement(runtime._tasks[task_id].workflow_id).settled


@pytest.mark.anyio
async def test_an_unpublished_synthetic_failure_settles_a_cancelling_task() -> None:
    registry = _Registry()
    runtime, task_id = await _cancelling(registry)
    redis = mock.MagicMock()
    redis.xadd_telemetry.side_effect = ConnectionError("telemetry redis down")
    watchdog = WorkerWatchdog(
        redis,
        mock.MagicMock(),
        runtime,
        logging.getLogger("cancelling"),
        enabled=True,
        check_interval=1,
        grace_seconds=0,
    )
    metrics = mock.MagicMock()
    monitor = EventMonitor(
        redis_client=mock.MagicMock(),
        logger=logging.getLogger("cancelling"),
        runtime=runtime,
        dispatcher=mock.MagicMock(),
        worker_registry=mock.MagicMock(),
        node_registry=mock.MagicMock(),
        metrics_recorder=metrics,
        watchdog=watchdog,
    )
    watchdog.set_failure_fallback(monitor._handle_task_event)

    watchdog._handle_worker_expired(_WORKER.id)

    _assert_cancelled(registry, runtime, metrics, task_id)


@pytest.mark.anyio
async def test_a_returned_cancelling_task_settles_rather_than_waiting() -> None:
    registry = _Registry()
    runtime, task_id = await _cancelling(registry)

    end = runtime.return_dispatch(task_id, None, increment_retry=True, front=False)

    assert end is DispatchEnd.CANCELLED

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert task_id not in runtime._ready_index


@pytest.mark.anyio
@pytest.mark.parametrize("retryable", [True, None], ids=["retryable", "synthetic"])
async def test_a_retried_failure_of_a_cancelling_merged_parent_runs_its_children_alone(
    retryable: bool | None,
) -> None:
    registry = _Registry()
    runtime = _runtime(registry, _InterruptRecorder())
    first, a = await _register(runtime, _siblings(names=["a1"]))
    _, b = await _register(runtime, _siblings(names=["b1", "b2"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"], b["b2"]]
    record_dispatch(runtime, parent, _WORKER)
    runtime.cancel_workflow(first)

    _monitor(runtime)._handle_task_event(_failed(parent, retryable))

    assert runtime._tasks[parent].status == TaskStatus.CANCELLED
    for child in b.values():
        record = runtime._tasks[child]
        assert record.status == TaskStatus.PENDING
        assert record.merge_key is None
        assert child in runtime._ready_index


@pytest.mark.anyio
async def test_a_task_cancelled_while_the_dispatcher_holds_it_stays_cancelled() -> None:
    registry = _Registry()
    runtime = _runtime(registry, _InterruptRecorder())
    workflow_id, _ = await _register(runtime, _siblings(names=["a"]))
    task_id = _next(runtime)
    worker_registry = mock.MagicMock()

    def cancel_then_find_no_worker(_task: object) -> list[object]:
        runtime.cancel_workflow(workflow_id)
        return []

    worker_registry.idle_satisfying_pool.side_effect = cancel_then_find_no_worker
    worker_registry.satisfying_workers.return_value = [_WORKER]
    Dispatcher(runtime, worker_registry, logging.getLogger("cancelling")).dispatch_once(
        task_id
    )

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert runtime._tasks[task_id].attempts == 0
    assert task_id not in runtime._ready_index
    assert registry.durable_status(task_id) == TaskStatus.CANCELLED
