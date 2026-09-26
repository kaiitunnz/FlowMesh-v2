"""A task whose worker could not reach its inputs runs again at no cost.

Its inputs are in the store, so the report spends no attempt and does not count
against the worker; it still has to come from the dispatch holding the task.
"""

from typing import Any

import pytest

from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, TaskFailureKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
)
from tests.server.task.test_v2_orchestration import _TS

_ECHO = """
apiVersion: mloc/v1
kind: Workflow
metadata: {name: unavailable}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo}
"""


def _failure(
    task_id: str, worker_id: str, dispatch_id: str, **fields: Any
) -> TaskEvent:
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=worker_id,
        dispatch_id=dispatch_id,
        error="task cannot reach input",
        retryable=True,
        ts=_TS,
        **fields,
    )


async def _dispatched(runtime: TaskRuntime) -> str:
    await _register(runtime, _ECHO)
    task_id = _next(runtime)
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    return task_id


@pytest.mark.anyio
async def test_an_unreachable_input_returns_the_task_without_spending_an_attempt() -> (
    None
):
    runtime = _runtime(_Registry())
    task_id = await _dispatched(runtime)

    _monitor(runtime).handle_task_event(
        _failure(
            task_id, "wkr-1", "dsp-1", failure_kind=TaskFailureKind.INPUT_UNAVAILABLE
        )
    )

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.failed_workers == []
    assert _next(runtime) == task_id


@pytest.mark.anyio
async def test_an_ordinary_retryable_failure_still_spends_an_attempt() -> None:
    runtime = _runtime(_Registry())
    task_id = await _dispatched(runtime)

    _monitor(runtime).handle_task_event(_failure(task_id, "wkr-1", "dsp-1"))

    record = runtime._tasks[task_id]
    assert record.attempts == 1
    assert record.failed_workers == ["wkr-1"]


@pytest.mark.anyio
async def test_a_report_from_another_dispatch_changes_nothing() -> None:
    runtime = _runtime(_Registry())
    task_id = await _dispatched(runtime)

    _monitor(runtime).handle_task_event(
        _failure(
            task_id, "wkr-1", "dsp-old", failure_kind=TaskFailureKind.INPUT_UNAVAILABLE
        )
    )

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.DISPATCHED
    assert record.dispatch_id == "dsp-1"


@pytest.mark.anyio
async def test_a_cancelling_task_settles_cancelled() -> None:
    runtime = _runtime(_Registry())
    task_id = await _dispatched(runtime)
    runtime.cancel_workflow(runtime._tasks[task_id].workflow_id)
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING

    _monitor(runtime).handle_task_event(
        _failure(
            task_id, "wkr-1", "dsp-1", failure_kind=TaskFailureKind.INPUT_UNAVAILABLE
        )
    )

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
