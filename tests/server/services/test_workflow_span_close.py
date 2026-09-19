"""The workflow span closes whenever the workflow's last task goes terminal.

The span is emitted by the completion finalizer, which returns early unless every task
has already left the remaining-task set. A workflow whose final tasks settle as a
cascade settles several tasks under one event, so the span depends on the finalizer
seeing the drained set on that call -- there is no later event to try again on.
"""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from shared.schemas.event import TaskEvent
from shared.utils.time import ts_to_iso
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _runtime,
    _worker,
)

_TS = "2026-09-16T00:00:00Z"

_CHAIN = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: span-close}
spec:
  graph:
    nodes:
      - name: head
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: tail
        dependsOn: [head]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""

_TERMINAL = (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)


class _RedisMirroringTaskState:
    """A Redis stand-in whose remaining-task set mirrors the runtime's own records.

    Production removes a task from that set when it reaches a terminal state, so
    deriving the set from the runtime keeps this test honest about *when* it drains
    rather than hard-coding an answer.
    """

    def __init__(self, runtime: Any, task_ids: list[str]) -> None:
        self._runtime = runtime
        self._task_ids = task_ids
        self.keys: dict[str, str] = {}

    def exists(self, key: str) -> int:
        return 1 if key in self.keys else 0

    def set_value(self, key: str, value: str) -> None:
        self.keys[key] = value

    def set_members(self, key: str) -> set[str]:
        if not key.endswith(":tasks"):
            return set()
        return {
            task_id
            for task_id in self._task_ids
            if (record := self._runtime.get_record(task_id)) is not None
            and record.status not in _TERMINAL
        }

    def __getattr__(self, name: str) -> Any:
        return MagicMock()


class _RecordingWorkflowSpanEmitter:
    def __init__(self) -> None:
        self.emitted: list[str] = []
        self.extents: list[tuple[str, str]] = []

    def emit(self, workflow_id: str, submitted_at: str, ended_at: str) -> None:
        self.emitted.append(workflow_id)
        self.extents.append((submitted_at, ended_at))


def _monitor(runtime: Any, redis: Any, emitter: Any) -> EventMonitor:
    return EventMonitor(
        redis_client=cast(Any, redis),
        logger=logging.getLogger("test.monitoring.workflow_span"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        workflow_span_emitter=cast(Any, emitter),
    )


@pytest.mark.anyio
async def test_a_cascade_failure_still_closes_the_workflow_span() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _CHAIN)
    head, tail = ids["head"], ids["tail"]

    registry.submitted_at = _TS
    redis = _RedisMirroringTaskState(runtime, [head, tail])
    redis.keys[f"workflow:{workflow_id}"] = "1"
    emitter = _RecordingWorkflowSpanEmitter()
    monitor = _monitor(runtime, redis, emitter)

    runtime.mark_dispatched(head, cast(Any, _worker()))
    # The head's failure cascades to the tail: both settle under this one event, and
    # no further task event follows it.
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_FAILED",
            task_id=head,
            worker_id="wkr-1",
            error="boom",
            retryable=False,
            ts=_TS,
        )
    )

    head_record, tail_record = runtime.get_record(head), runtime.get_record(tail)
    assert head_record is not None and head_record.status in _TERMINAL
    assert tail_record is not None and tail_record.status in _TERMINAL
    assert emitter.emitted == [workflow_id]


@pytest.mark.anyio
async def test_an_unreadable_submission_time_does_not_strand_the_log_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The span lookup must not be able to take the log-stream close down with it.

    Reading the workflow's submission time is a Redis read like the finalizer's other
    three, but it sits after their shared guard and before the close, so a failure
    there leaves the workflow's log stream open forever and the close silently never
    happens.
    """
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _CHAIN)
    head, tail = ids["head"], ids["tail"]

    def _raises(_workflow_id: str) -> str | None:
        raise RuntimeError("workflow record unreadable")

    monkeypatch.setattr(runtime, "workflow_submitted_at", _raises)

    redis = _RedisMirroringTaskState(runtime, [head, tail])
    redis.keys[f"workflow:{workflow_id}"] = "1"
    monitor = _monitor(runtime, redis, _RecordingWorkflowSpanEmitter())

    runtime.mark_dispatched(head, cast(Any, _worker()))
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_FAILED",
            task_id=head,
            worker_id="wkr-1",
            error="boom",
            retryable=False,
            ts=_TS,
        )
    )

    assert f"workflow:{workflow_id}:logs:closed" in redis.keys


@pytest.mark.anyio
async def test_an_already_terminal_task_event_still_closes_the_workflow() -> None:
    """A task settled before its event is handled still closes the workflow.

    The dispatcher settles a no-eligible-worker task in the runtime itself, so an
    event for it arrives describing a task that is already terminal. The finalizer
    handles that correctly -- but note the dispatcher does not in fact deliver such
    an event today, so this guards the finalizer rather than describing that path.
    """
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _CHAIN)
    head, tail = ids["head"], ids["tail"]

    redis = _RedisMirroringTaskState(runtime, [head, tail])
    redis.keys[f"workflow:{workflow_id}"] = "1"
    emitter = _RecordingWorkflowSpanEmitter()
    monitor = _monitor(runtime, redis, emitter)

    # Dispatcher.fail_task settles in the runtime first; this then delivers the
    # event it builds, which is the step that does not happen in production.
    runtime.mark_failed(
        head,
        None,
        {"reason": "no_eligible_worker"},
        _TS,
        error="No worker satisfies the task requirements",
    )
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_FAILED",
            task_id=head,
            error="No worker satisfies the task requirements",
            retryable=False,
            payload={"reason": "no_eligible_worker"},
            ts=_TS,
        )
    )

    assert redis.set_members(f"workflow:{workflow_id}:tasks") == set()
    assert f"workflow:{workflow_id}:logs:closed" in redis.keys
    assert emitter.emitted == [workflow_id]


@pytest.mark.anyio
async def test_the_workflow_span_ends_at_the_last_tasks_recorded_finish() -> None:
    """The end is read from the ledger, not the clock.

    The finalizer runs once per workflow but runs again after a restart, so an end taken
    from wall clock gives the re-emitted span a different duration than the first. Every
    other synthesized span reads both its ends from durable records; this one must too.
    """
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _CHAIN)
    head, tail = ids["head"], ids["tail"]

    registry.submitted_at = _TS
    redis = _RedisMirroringTaskState(runtime, [head, tail])
    redis.keys[f"workflow:{workflow_id}"] = "1"
    emitter = _RecordingWorkflowSpanEmitter()
    monitor = _monitor(runtime, redis, emitter)

    runtime.mark_dispatched(head, cast(Any, _worker()))
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_FAILED",
            task_id=head,
            worker_id="wkr-1",
            error="boom",
            retryable=False,
            ts=_TS,
        )
    )

    finishes = [
        record.finished_ts
        for task_id in (head, tail)
        if (record := runtime.get_record(task_id)) is not None
        and record.finished_ts is not None
    ]
    assert len(finishes) == 2
    assert emitter.extents == [(_TS, ts_to_iso(max(finishes)))]
