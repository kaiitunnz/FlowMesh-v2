"""Reports naming a task the runtime does not hold."""

import pytest

from server.task.models import EventEffect, TaskStatus
from tests.server.task.test_runtime_rehydrate import (
    GRAPH,
    FakeWorkflowRegistry,
    _register,
    _runtime,
)

_TS = "2026-06-01T00:00:00Z"


@pytest.mark.anyio
async def test_a_report_for_an_unknown_task_changes_nothing() -> None:
    runtime = _runtime(FakeWorkflowRegistry())
    _, ids = await _register(runtime, GRAPH)
    before = {task_id: runtime.describe_task(task_id) for task_id in ids.values()}

    outcome = runtime.mark_succeeded("tsk-unknown", "wkr-1", {}, _TS)
    assert outcome.effect is EventEffect.STALE
    assert runtime.mark_failed("tsk-unknown", "wkr-1", {"error": "boom"}, _TS) == (
        [],
        [],
    )

    assert runtime.get_record("tsk-unknown") is None
    assert runtime.task_statuses() == {
        task_id: TaskStatus.PENDING for task_id in ids.values()
    }
    assert {task_id: runtime.describe_task(task_id) for task_id in ids.values()} == (
        before
    )
    assert "tsk-unknown" not in runtime._completed
    assert "tsk-unknown" not in runtime._failed
