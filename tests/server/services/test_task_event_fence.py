"""A worker's task event changes its task only while its dispatch holds the task."""

import json
import logging
from typing import Any
from unittest import mock

import pytest

from server.dispatcher.base import Dispatcher
from server.orchestration import WorkItemStatus
from server.registries.worker import Worker
from server.services.monitoring import EventMonitor
from server.services.watchdog import WorkerWatchdog
from server.task.models import DispatchEnd, EventEffect, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent, parse_event
from shared.tasks.worker_message import WorkerStatus, WorkerTaskMessage
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import result_payload
from tests.server.task.test_task_merge import (
    _InterruptRecorder,
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
    _siblings,
)
from tests.server.task.test_v2_orchestration import _TS

_ECHO = """
apiVersion: mloc/v1
kind: Workflow
metadata: {name: fence}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo}
"""

_ECHO_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: fence-v2}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""

_EVENT_TYPES = [
    "TASK_STARTED",
    "TASK_UPDATE",
    "TASK_SUCCEEDED",
    "TASK_FAILED",
    "TASK_CANCELLED",
]


def _worker(worker_id: str) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )


def _event(
    event_type: str,
    runtime: TaskRuntime,
    task_id: str,
    worker_id: str,
    dispatch_id: str | None = None,
) -> TaskEvent:
    fields: dict[str, Any] = {}
    if dispatch_id is not None:
        fields["dispatch_id"] = dispatch_id
    payload: dict[str, Any] = {}
    match event_type:
        case "TASK_UPDATE":
            payload = {"progress": worker_id}
        case "TASK_SUCCEEDED":
            payload = result_payload(runtime._results, task_id, {"value": worker_id})
        case "TASK_FAILED":
            fields.update(error=f"failed on {worker_id}", retryable=True)
    return TaskEvent(
        type=event_type,
        task_id=task_id,
        worker_id=worker_id,
        payload=payload,
        ts=_TS,
        **fields,
    )


async def _solo(runtime: TaskRuntime) -> tuple[str, str]:
    workflow_id, _ = await _register(runtime, _ECHO)
    return workflow_id, _next(runtime)


def _assert_held_by(
    runtime: TaskRuntime, task_id: str, worker_id: str, attempts: int = 1
) -> None:
    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.DISPATCHED
    assert record.assigned_worker == worker_id
    assert record.attempts == attempts
    assert record.latest_update is None
    assert runtime.result_binding(task_id) is None


async def _requeued_to_another_worker(
    runtime: TaskRuntime, monitor: EventMonitor, dispatch_id: str | None
) -> str:
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", dispatch_id)
    monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-2", dispatch_id and "dsp-2")
    return task_id


@pytest.mark.anyio
@pytest.mark.parametrize("dispatch_id", [None, "dsp-1"], ids=["tokenless", "token"])
@pytest.mark.parametrize("event_type", _EVENT_TYPES)
async def test_an_event_from_a_superseded_worker_changes_nothing(
    event_type: str, dispatch_id: str | None
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    task_id = await _requeued_to_another_worker(runtime, monitor, dispatch_id)
    metrics = mock.MagicMock()
    monitor._metrics = metrics

    monitor._handle_task_event(
        _event(event_type, runtime, task_id, "wkr-1", dispatch_id)
    )

    _assert_held_by(runtime, task_id, "wkr-2")
    assert runtime._tasks[task_id].failed_workers == []
    assert runtime._tasks[task_id].last_error is None
    assert metrics.record_task_event.call_count == 0


@pytest.mark.anyio
async def test_a_late_start_after_its_worker_left_does_not_re_dispatch_the_task() -> (
    None
):
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1")

    monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))
    monitor._handle_task_event(_event("TASK_STARTED", runtime, task_id, "wkr-1"))

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.assigned_worker is None
    assert task_id in runtime._ready_index


@pytest.mark.anyio
@pytest.mark.parametrize("event_type", _EVENT_TYPES)
async def test_an_earlier_dispatch_to_the_same_worker_changes_nothing(
    event_type: str,
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    monitor._handle_task_event(
        _event("TASK_FAILED", runtime, task_id, "wkr-1", "dsp-1")
    )
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-1", "dsp-2")

    monitor._handle_task_event(_event(event_type, runtime, task_id, "wkr-1", "dsp-1"))

    _assert_held_by(runtime, task_id, "wkr-1")
    assert runtime._tasks[task_id].dispatch_id == "dsp-2"


@pytest.mark.anyio
async def test_an_event_naming_no_dispatch_is_matched_by_its_worker() -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")

    monitor._handle_task_event(_event("TASK_SUCCEEDED", runtime, task_id, "wkr-2"))
    assert runtime._tasks[task_id].status == TaskStatus.DISPATCHED
    monitor._handle_task_event(_event("TASK_SUCCEEDED", runtime, task_id, "wkr-1"))
    assert runtime._tasks[task_id].status == TaskStatus.DONE


def _watchdog(runtime: TaskRuntime) -> tuple[WorkerWatchdog, mock.MagicMock]:
    redis = mock.MagicMock()
    watchdog = WorkerWatchdog(
        redis,
        mock.MagicMock(),
        runtime,
        logging.getLogger("fence"),
        enabled=True,
        check_interval=1,
        grace_seconds=0,
    )
    return watchdog, redis


def _published(redis: mock.MagicMock) -> list[TaskEvent]:
    events = []
    for call in redis.xadd_telemetry.call_args_list:
        event = parse_event(json.loads(call.args[1]["payload"]))
        assert isinstance(event, TaskEvent)
        events.append(event)
    return events


@pytest.mark.anyio
@pytest.mark.parametrize("dispatch_id", [None, "dsp-1"], ids=["tokenless", "token"])
async def test_a_loss_the_watchdog_and_the_worker_both_report_spends_one_attempt(
    dispatch_id: str | None,
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", dispatch_id)
    watchdog, redis = _watchdog(runtime)

    watchdog._handle_worker_expired("wkr-1")
    (synthetic,) = _published(redis)
    assert synthetic.dispatch_id == dispatch_id
    monitor._handle_task_event(synthetic)
    monitor._handle_task_event(
        _event("TASK_FAILED", runtime, task_id, "wkr-1", dispatch_id)
    )

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 1
    assert record.last_error == "worker_heartbeat_expired"


@pytest.mark.anyio
async def test_a_loss_reported_by_unregister_and_the_worker_spends_one_attempt() -> (
    None
):
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1")

    monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))
    monitor._handle_task_event(_event("TASK_FAILED", runtime, task_id, "wkr-1"))

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 1
    assert record.failed_workers == []


@pytest.mark.anyio
async def test_an_unregister_after_the_task_moved_on_leaves_its_new_dispatch() -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1")
    monitor._handle_task_event(_event("TASK_FAILED", runtime, task_id, "wkr-1"))
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-2")

    end = runtime.return_dispatch(task_id, "wkr-1", increment_retry=True, front=True)

    assert end is DispatchEnd.STALE
    _assert_held_by(runtime, task_id, "wkr-2")


@pytest.mark.anyio
@pytest.mark.parametrize("dispatch_id", [None, "dsp-1"], ids=["tokenless", "token"])
async def test_a_failure_replayed_after_a_restart_spends_one_attempt(
    dispatch_id: str | None,
) -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", dispatch_id)
    failure = _event("TASK_FAILED", runtime, task_id, "wkr-1", dispatch_id)
    _monitor(runtime)._handle_task_event(failure)

    restored = _runtime(registry)
    await restored.rehydrate()
    _monitor(restored)._handle_task_event(failure)

    record = restored._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 1
    assert task_id in restored._ready_index


@pytest.mark.anyio
async def test_a_failure_handled_again_after_its_commit_failed_is_committed_once() -> (
    None
):
    registry = _Registry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    failure = _event("TASK_FAILED", runtime, task_id, "wkr-1", "dsp-1")

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        monitor._handle_task_event(failure)
    monitor._handle_task_event(failure)

    record = runtime._tasks[task_id]
    assert record.attempts == 1
    assert task_id in runtime._ready_index
    assert registry.durable_status(task_id) == TaskStatus.PENDING


def _fast_worker_dispatcher(
    runtime: TaskRuntime, monitor: EventMonitor, *event_types: str
) -> tuple[Dispatcher, mock.Mock]:
    """A dispatcher whose worker reports on the task before the dispatch is
    recorded."""
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker("wkr-1")]
    registry.satisfying_workers.return_value = [_worker("wkr-1")]

    def publish(_worker: Worker, message: WorkerTaskMessage) -> int:
        for event_type in event_types:
            monitor._handle_task_event(
                _event(
                    event_type,
                    runtime,
                    message.task_id,
                    "wkr-1",
                    message.dispatch_id,
                )
            )
        return 1

    registry.publish_task.side_effect = publish
    return (
        Dispatcher(
            runtime, registry, logging.getLogger("fence"), enable_task_merge=False
        ),
        registry,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event_types", "status"),
    [
        (("TASK_STARTED",), TaskStatus.DISPATCHED),
        (("TASK_STARTED", "TASK_SUCCEEDED"), TaskStatus.DONE),
    ],
    ids=["started", "succeeded"],
)
async def test_a_report_before_its_dispatch_is_recorded_applies(
    event_types: tuple[str, ...], status: str
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    dispatcher, _ = _fast_worker_dispatcher(runtime, monitor, *event_types)

    dispatcher.dispatch_once(task_id)

    record = runtime._tasks[task_id]
    assert record.status == status
    assert record.assigned_worker == "wkr-1"


@pytest.mark.anyio
async def test_a_failure_before_its_dispatch_is_recorded_returns_the_task() -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    dispatcher, _ = _fast_worker_dispatcher(
        runtime, monitor, "TASK_STARTED", "TASK_FAILED"
    )

    dispatcher.dispatch_once(task_id)

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 1
    assert task_id in runtime._ready_index


class _Crash(Exception):
    pass


@pytest.mark.anyio
async def test_a_dispatch_lost_before_it_was_recorded_runs_again() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(runtime, _monitor(runtime))

    with mock.patch.object(runtime, "mark_dispatched", side_effect=_Crash):
        with pytest.raises(_Crash):
            dispatcher.dispatch_once(task_id)
    message: WorkerTaskMessage = worker_registry.publish_task.call_args.args[1]
    restored = _runtime(registry)
    await restored.rehydrate()
    _monitor(restored)._handle_task_event(
        _event(
            "TASK_STARTED",
            restored,
            task_id,
            "wkr-1",
            message.dispatch_id,
        )
    )

    record = restored._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert task_id in restored._ready_index


@pytest.mark.anyio
async def test_a_success_reported_before_its_dispatch_was_recorded_replays() -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(
        runtime, monitor, "TASK_SUCCEEDED"
    )
    dispatcher.dispatch_once(task_id)
    message: WorkerTaskMessage = worker_registry.publish_task.call_args.args[1]

    replay = runtime.mark_succeeded(
        task_id,
        "wkr-1",
        _event("TASK_SUCCEEDED", runtime, task_id, "wkr-1").payload,
        _TS,
        message.dispatch_id,
    )

    assert (replay.effect, replay.status) == (EventEffect.SETTLED, TaskStatus.DONE)
    assert runtime._tasks[task_id].dispatch_id == message.dispatch_id


@pytest.mark.anyio
@pytest.mark.parametrize("loss", ["unregistered", "expired"])
async def test_a_worker_lost_before_its_dispatch_was_recorded_runs_the_task_elsewhere(
    loss: str,
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(runtime, monitor)

    def lose_the_worker(_worker: Worker, _message: WorkerTaskMessage) -> int:
        if loss == "unregistered":
            monitor._handle_worker_event(
                WorkerEvent(type="UNREGISTER", worker_id="wkr-1")
            )
        else:
            watchdog, redis = _watchdog(runtime)
            watchdog._handle_worker_expired("wkr-1")
            for event in _published(redis):
                monitor._handle_task_event(event)
        return 1

    worker_registry.publish_task.side_effect = lose_the_worker
    dispatcher.dispatch_once(task_id)

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-2")
    monitor._handle_task_event(_event("TASK_SUCCEEDED", runtime, task_id, "wkr-2"))
    assert record.status == TaskStatus.DONE
    assert record.assigned_worker == "wkr-2"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "event_types",
    [("TASK_STARTED", "TASK_SUCCEEDED"), ("TASK_STARTED", "TASK_FAILED")],
    ids=["succeeded", "failed"],
)
async def test_a_dispatch_its_worker_ended_first_leaves_the_worker_idle(
    event_types: tuple[str, ...],
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(
        runtime, monitor, *event_types
    )

    dispatcher.dispatch_once(task_id)

    writes = [call.args for call in worker_registry.update_worker_status.call_args_list]
    assert ("wkr-1", WorkerStatus.BUSY) not in writes


@pytest.mark.anyio
async def test_a_start_before_a_retry_is_recorded_starts_the_retry() -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    workflow_id, _ = await _register(runtime, _ECHO_V2)
    task_id = _next(runtime)
    record_dispatch(runtime, task_id, "wkr-2", "dsp-1")
    monitor._handle_task_event(
        _event("TASK_FAILED", runtime, task_id, "wkr-2", "dsp-1")
    )
    assert _next(runtime) == task_id
    dispatcher, _ = _fast_worker_dispatcher(runtime, monitor, "TASK_STARTED")

    dispatcher.dispatch_once(task_id)

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(task_id)
    assert work_item is not None
    attempts = [
        attempt.status.value
        for attempt in engine.to_snapshot().attempts
        if attempt.work_item_id == work_item.work_item_id
    ]
    assert attempts == ["failed", "running"]


@pytest.mark.anyio
@pytest.mark.parametrize("retryable", [True, False])
async def test_a_merged_batch_failing_before_it_is_recorded_runs_each_task_alone(
    retryable: bool,
) -> None:
    runtime = _runtime(_Registry())
    monitor = _monitor(runtime)
    _, ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    worker_registry = mock.Mock()
    worker_registry.idle_satisfying_pool.return_value = [_worker("wkr-1")]
    worker_registry.satisfying_workers.return_value = [_worker("wkr-1")]

    def publish(_worker: Worker, message: WorkerTaskMessage) -> int:
        assert message.merged_children
        failure = _event("TASK_FAILED", runtime, parent, "wkr-1", message.dispatch_id)
        monitor._handle_task_event(failure.model_copy(update={"retryable": retryable}))
        return 1

    worker_registry.publish_task.side_effect = publish
    Dispatcher(runtime, worker_registry, logging.getLogger("fence")).dispatch_once(
        parent
    )

    for task_id in ids.values():
        record = runtime._tasks[task_id]
        assert record.status == TaskStatus.PENDING
        assert record.attempts == 0
        assert record.failed_workers == []
        assert record.merge_key is None


@pytest.mark.anyio
async def test_a_cancel_while_a_dispatch_is_published_interrupts_its_worker() -> None:
    interrupts = _InterruptRecorder()
    runtime = _runtime(_Registry(), interrupts)
    workflow_id, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(runtime, _monitor(runtime))
    worker_registry.publish_task.side_effect = lambda *_: (
        runtime.cancel_workflow(workflow_id) and 1
    )

    dispatcher.dispatch_once(task_id)

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING
    assert interrupts.interrupted == [task_id]


@pytest.mark.anyio
async def test_a_dispatch_a_cancel_recorded_but_never_delivered_settles_cancelled() -> (
    None
):
    runtime = _runtime(_Registry(), _InterruptRecorder())
    workflow_id, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(runtime, _monitor(runtime))

    def cancel_then_fail(*_: Any) -> int:
        runtime.cancel_workflow(workflow_id)
        raise ConnectionError("publish failed")

    worker_registry.publish_task.side_effect = cancel_then_fail

    dispatcher.dispatch_once(task_id)

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert task_id not in runtime._ready_index


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("cancel_first", "terminal", "settled"),
    [
        (False, "TASK_SUCCEEDED", TaskStatus.DONE),
        (False, "TASK_FAILED", TaskStatus.FAILED),
        (True, "TASK_CANCELLED", TaskStatus.CANCELLED),
        (True, "TASK_SUCCEEDED", TaskStatus.CANCELLED),
    ],
)
async def test_a_terminal_landing_while_its_worker_unregisters_stays_settled(
    cancel_first: bool, terminal: str, settled: str
) -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    monitor = _monitor(runtime)
    workflow_id, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    if cancel_first:
        runtime.cancel_workflow(workflow_id)
    event = _event(terminal, runtime, task_id, "wkr-1", "dsp-1")
    event = event.model_copy(update={"retryable": False})
    listed = runtime.recover_tasks_for_worker

    def listed_then_settled(worker_id: str) -> list[str]:
        tasks = listed(worker_id)
        monitor._handle_task_event(event)
        return tasks

    with mock.patch.object(runtime, "recover_tasks_for_worker", listed_then_settled):
        monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))

    record = runtime._tasks[task_id]
    assert record.status == settled
    assert record.attempts == 0
    assert task_id not in runtime._ready_index


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("settle", "report"),
    [
        ("TASK_SUCCEEDED", "TASK_FAILED"),
        ("TASK_CANCELLED", "TASK_CANCELLED"),
        ("TASK_CANCELLED", "TASK_SUCCEEDED"),
        ("TASK_CANCELLED", "TASK_FAILED"),
    ],
)
async def test_a_late_report_on_a_settled_task_records_nothing(
    settle: str, report: str
) -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    monitor = _monitor(runtime)
    workflow_id, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    if settle == "TASK_CANCELLED":
        runtime.cancel_workflow(workflow_id)
    monitor._handle_task_event(_event(settle, runtime, task_id, "wkr-1", "dsp-1"))
    settled = runtime._tasks[task_id].status
    metrics = mock.MagicMock()
    monitor._metrics = metrics

    monitor._handle_task_event(_event(report, runtime, task_id, "wkr-1", "dsp-1"))

    record = runtime._tasks[task_id]
    assert record.status == settled
    assert record.failed_workers == []
    assert record.last_error is None
    assert metrics.record_task_event.call_count == 0
    assert metrics.finalize_task_failure.call_count == 0
    assert metrics.finalize_task_cancellation.call_count == 0


@pytest.mark.anyio
async def test_a_v2_failure_handled_again_after_its_commit_failed_retries_once() -> (
    None
):
    registry = _Registry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    workflow_id, _ = await _register(runtime, _ECHO_V2)
    task_id = _next(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    failure = _event("TASK_FAILED", runtime, task_id, "wkr-1", "dsp-1")

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        monitor._handle_task_event(failure)
    monitor._handle_task_event(failure)

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(task_id)
    assert work_item is not None and work_item.status is WorkItemStatus.READY
    assert registry.ledger_blobs[workflow_id] == engine.to_snapshot().model_dump_json()
    assert runtime._tasks[task_id].attempts == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("cancel_first", "report", "retryable", "settled"),
    [
        (False, "TASK_SUCCEEDED", None, TaskStatus.DONE),
        (False, "TASK_FAILED", False, TaskStatus.FAILED),
        (False, "TASK_FAILED", True, TaskStatus.PENDING),
        (True, "TASK_CANCELLED", None, TaskStatus.CANCELLED),
        (True, "TASK_FAILED", True, TaskStatus.CANCELLED),
    ],
)
async def test_a_report_handled_again_after_its_commit_failed_counts_once(
    cancel_first: bool, report: str, retryable: bool | None, settled: str
) -> None:
    registry = _Registry()
    runtime = _runtime(registry, _InterruptRecorder())
    monitor = _monitor(runtime)
    monitor._metrics = mock.MagicMock()
    workflow_id, task_id = await _solo(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    if cancel_first:
        runtime.cancel_workflow(workflow_id)
    event = _event(report, runtime, task_id, "wkr-1", "dsp-1")
    event = event.model_copy(update={"retryable": retryable})

    with (
        mock.patch.object(monitor, "_unregister_port_forward") as unregister,
        mock.patch.object(monitor, "_close_task_log_stream") as close_log,
    ):
        registry.fail_next = True
        with pytest.raises(ConnectionError):
            monitor._handle_task_event(event)
        monitor._handle_task_event(event)
        monitor._handle_task_event(event)

    assert runtime._tasks[task_id].status == settled
    assert registry.durable_status(task_id) == settled
    assert monitor._metrics.record_task_event.call_count == 1
    assert unregister.call_count == 1
    assert close_log.call_count == (0 if settled == TaskStatus.PENDING else 1)


@pytest.mark.anyio
async def test_a_v2_success_handled_again_after_its_commit_failed_settles_it() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    workflow_id, _ = await _register(runtime, _ECHO_V2)
    task_id = _next(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    success = _event("TASK_SUCCEEDED", runtime, task_id, "wkr-1", "dsp-1")

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        monitor._handle_task_event(success)
    monitor._handle_task_event(success)

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(task_id)
    assert work_item is not None and work_item.status is WorkItemStatus.SETTLED
    assert registry.ledger_blobs[workflow_id] == engine.to_snapshot().model_dump_json()
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_cancel_before_the_publish_begins_publishes_nothing() -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    workflow_id, task_id = await _solo(runtime)
    dispatcher, worker_registry = _fast_worker_dispatcher(runtime, _monitor(runtime))
    traceparent = runtime.dispatch_traceparent

    def cancel_while_building(tid: str) -> str | None:
        runtime.cancel_workflow(workflow_id)
        return traceparent(tid)

    with mock.patch.object(runtime, "dispatch_traceparent", cancel_while_building):
        dispatcher.dispatch_once(task_id)

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert worker_registry.publish_task.call_count == 0
    assert task_id not in runtime._publishing
