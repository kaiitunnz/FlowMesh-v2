"""The metrics snapshot counts the tasks that finished, and keeps nothing per task once
a task's workflow is freed."""

import logging
from pathlib import Path

from server.services.metrics import MetricsRecorder
from shared.schemas.event import TaskEvent

_TS = "2026-06-01T00:00:00Z"


def _event(event_type: str, task_id: str) -> TaskEvent:
    return TaskEvent(type=event_type, task_id=task_id, payload={}, ts=_TS)


def test_each_finished_task_counts_once(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path, logging.getLogger("metrics-count"))
    for task_id in ("tsk-1", "tsk-2", "tsk-3"):
        recorder.record_task_event(_event("TASK_SUBMITTED", task_id))
    recorder.record_task_event(_event("TASK_SUCCEEDED", "tsk-1"))
    recorder.record_task_event(_event("TASK_SUCCEEDED", "tsk-1"))
    recorder.record_task_event(_event("TASK_FAILED", "tsk-2"))
    recorder.finalize_task_failure("tsk-2")
    recorder.finalize_task_failure("tsk-2")

    assert recorder.snapshot()["completed_tasks"] == 2


def test_a_freed_task_keeps_its_count_and_nothing_else(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path, logging.getLogger("metrics-forget"))
    recorder.record_task_event(_event("TASK_SUBMITTED", "tsk-1"))
    recorder.record_task_event(_event("TASK_SUCCEEDED", "tsk-1"))

    recorder.forget_tasks(["tsk-1", "tsk-unknown"])

    assert recorder._task_meta == {}
    assert recorder.snapshot()["completed_tasks"] == 1
