"""A task whose worker could not reach its inputs runs again at no cost.

Its inputs are in the store, so the report spends no attempt and does not count
against the worker; it still has to come from the dispatch holding the task.
"""

from typing import Any, cast

import pytest

from server.orchestration.state import AttemptStatus, WorkItemStatus
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
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _pop_ready,
)
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import (
    _worker,
)

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


_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: v2-unavailable}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""


@pytest.mark.anyio
async def test_a_v2_return_closes_its_attempt_without_charging_it() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register_v2(runtime, _V2)
    task_id = ids["a"]
    assert _pop_ready(runtime) == [task_id]
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-1")), "dsp-1")
    runtime.mark_started(task_id, "wkr-1", {}, _TS, dispatch_id="dsp-1")

    runtime.fail_dispatch(
        task_id,
        "wkr-1",
        {},
        _TS,
        "dsp-1",
        error="task cannot reach input",
        retryable=True,
        failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
    )

    engine = runtime._engines[workflow_id]
    work_item = engine.work_item(task_id)
    assert work_item is not None
    assert work_item.status is WorkItemStatus.READY
    assert [engine._attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED
    ]
    assert runtime._tasks[task_id].attempts == 0

    assert _pop_ready(runtime) == [task_id]
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-2")), "dsp-2")
    runtime.mark_started(task_id, "wkr-2", {}, _TS, dispatch_id="dsp-2")
    runtime.mark_succeeded(task_id, "wkr-2", {}, _TS, dispatch_id="dsp-2")
    assert [engine._attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED,
        AttemptStatus.SUCCEEDED,
    ]
