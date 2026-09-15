"""One agent episode counts one task completion, whatever its step count.

An episode reports a worker success per run-to-yield step, so the completion metric
follows the step that settles the task rather than every dispatch it took to get there.
"""

import logging
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor
from shared.harness import HarnessResult, HarnessResultKind
from shared.schemas.event import TaskEvent

_TS = "2026-09-16T00:00:00Z"


def _monitor(metrics: MagicMock) -> EventMonitor:
    runtime = MagicMock()
    runtime.get_merged_children.return_value = []
    runtime.mark_succeeded.return_value = []
    runtime.get_record.return_value = None
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.episode_completion"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=metrics,
        watchdog=MagicMock(),
    )


def _succeeded(step: HarnessResult) -> TaskEvent:
    return TaskEvent(
        type="TASK_SUCCEEDED",
        task_id="tsk-1",
        worker_id="wkr-1",
        payload={"agent_episode": step.model_dump(mode="json")},
        ts=_TS,
    )


def test_multi_step_episode_counts_one_completion() -> None:
    metrics = MagicMock()
    monitor = _monitor(metrics)
    steps = [
        HarnessResult(kind=HarnessResultKind.YIELD),
        HarnessResult(kind=HarnessResultKind.YIELD),
        HarnessResult(kind=HarnessResultKind.COMPLETION, value="done"),
    ]
    for step in steps:
        monitor._handle_task_event(_succeeded(step))
    assert metrics.record_task_event.call_count == 1


def test_an_ordinary_success_counts_one_completion() -> None:
    metrics = MagicMock()
    monitor = _monitor(metrics)
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_SUCCEEDED",
            task_id="tsk-2",
            worker_id="wkr-1",
            payload={"result": "ok"},
            ts=_TS,
        )
    )
    assert metrics.record_task_event.call_count == 1
