"""What a report's handler does after it settles a task finds its workflow resident."""

import asyncio
from typing import Any

import pytest

from shared.schemas.event import TaskEvent
from tests.server.runtime_helpers import reclaimable
from tests.server.services.test_monitor_commit_fence import (
    _dispatched,
    _monitor,
    _Store,
    _with_usage,
)
from tests.server.services.test_task_event_fence import _ECHO_V2, _event
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_settled_history import _registry, _settled
from tests.server.task.test_v2_orchestration import LINEAR, _register


def test_a_workflow_settled_inside_a_tail_stays_resident_until_it_ends() -> None:
    runtime = _runtime(_Store())
    workflow_id, _ = asyncio.run(_register(runtime, LINEAR))
    requested: list[str] = []
    setattr(runtime.settled_history, "request", requested.append)

    with runtime.event_tail():
        with runtime.event_tail():
            runtime.cancel_workflow(workflow_id)
        assert runtime.workflow_settlement(workflow_id).settled
        assert not reclaimable(runtime, workflow_id)

    assert reclaimable(runtime, workflow_id)
    assert workflow_id in requested


def test_a_tail_holds_the_workflow_of_the_task_it_names() -> None:
    runtime = _runtime(_Store())
    workflow_id, _ = asyncio.run(_register(runtime, LINEAR))
    runtime.cancel_workflow(workflow_id)
    task_id = runtime._tasks.ids_of(workflow_id)[0]

    with runtime.event_tail(task_id, "tsk-unknown"):
        assert not reclaimable(runtime, workflow_id)
    assert reclaimable(runtime, workflow_id)


def test_a_report_applied_directly_finishes_its_side_effects_before_freeing() -> None:
    _, runtime, workflow_id, task_id = _dispatched(_ECHO_V2)
    monitor, telemetry, _ = _monitor(runtime)
    seen: list[bool] = []
    close = monitor._close_task_log_stream

    def closing(closed_task_id: str) -> None:
        seen.append(reclaimable(runtime, workflow_id))
        close(closed_task_id)

    setattr(monitor, "_close_task_log_stream", closing)
    success = _with_usage(_event("TASK_SUCCEEDED", runtime, task_id, "wkr-1", "dsp-1"))

    monitor.handle_task_event(success)

    assert seen == [False]
    assert len(telemetry.sealed) == 1
    assert reclaimable(runtime, workflow_id)


def test_a_late_report_for_a_freed_task_is_acknowledged_from_the_store() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    task_id = runtime._tasks.ids_of(workflow_id)[0]
    runtime.settled_history.drain()
    assert runtime.get_record(task_id) is None
    monitor, telemetry, _ = _monitor(runtime)
    late = TaskEvent(
        type="TASK_SUCCEEDED", task_id=task_id, worker_id="wkr-1", payload={}
    )

    monitor.handle_task_event(late)

    assert runtime.task_freed(task_id)
    assert not runtime.task_freed("tsk-unknown")
    assert telemetry.sealed == []
    setattr(registry, "history_members", _unavailable)
    with pytest.raises(ConnectionError):
        monitor.handle_task_event(late)


def _unavailable(*args: Any) -> Any:
    raise ConnectionError("store unavailable")
