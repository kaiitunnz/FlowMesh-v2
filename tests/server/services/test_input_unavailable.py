"""A task whose worker could not reach its inputs runs again at no cost.

The report spends no attempt and has to come from the dispatch holding the task. When
it names the inputs, control reads them itself to decide whose fault it was.
"""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import AttemptStatus, WorkItemStatus
from server.task.models import TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.results import ResultUnavailable, ResultUnreadable
from server.task.runtime import TaskRuntime
from shared.content import ContentReference, reference_for
from shared.schemas.event import TaskEvent, TaskFailureKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
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
    _NoopSecretVault,
    _pop_ready,
)
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import (
    _worker,
    _WorkerRegistryStub,
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


_INPUT = reference_for("org", b"upstream", media_type="application/json")


class _Probe:
    """Control's own read of a stored object, answering as told."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.reads: list[ContentReference] = []

    def __call__(self, reference: ContentReference) -> None:
        self.reads.append(reference)
        if self.error is not None:
            raise self.error


def _attributing_runtime() -> tuple[TaskRuntime, StoreRedriveScheduler, _Probe]:
    schedulers: list[StoreRedriveScheduler] = []

    def _scheduler(fire: Any, logger: logging.Logger) -> StoreRedriveScheduler:
        scheduler = StoreRedriveScheduler(
            fire, logger, base_delay_sec=0, run_thread=False
        )
        schedulers.append(scheduler)
        return scheduler

    reader = make_result_reader()
    probe = _Probe()
    reader.verify = probe  # type: ignore[method-assign]
    runtime = TaskRuntime(
        cast(Any, _Registry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("input-unavailable"),
        secret_vault=cast(Any, _NoopSecretVault()),
        redrive=_scheduler,
    )
    return runtime, schedulers[0], probe


def _ready(runtime: TaskRuntime) -> str | None:
    with runtime._cv:
        return runtime._pop_ready_locked()


def _report_unavailable(runtime: TaskRuntime, task_id: str) -> None:
    _monitor(runtime).handle_task_event(
        _failure(
            task_id,
            "wkr-1",
            "dsp-1",
            failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
            unavailable_inputs=[_INPUT],
        )
    )


@pytest.mark.anyio
async def test_a_task_waits_while_control_reads_what_its_worker_could_not() -> None:
    runtime, scheduler, probe = _attributing_runtime()
    task_id = await _dispatched(runtime)

    _report_unavailable(runtime, task_id)

    assert runtime._tasks[task_id].status == TaskStatus.PENDING
    assert _ready(runtime) is None
    scheduler.run_due()
    assert probe.reads == [_INPUT]


@pytest.mark.anyio
async def test_input_control_reads_fine_runs_again_away_from_its_worker() -> None:
    runtime, scheduler, _probe = _attributing_runtime()
    task_id = await _dispatched(runtime)

    _report_unavailable(runtime, task_id)
    scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.failed_workers == ["wkr-1"]
    assert _ready(runtime) == task_id


@pytest.mark.anyio
async def test_input_missing_at_control_fails_the_task_typed() -> None:
    runtime, scheduler, probe = _attributing_runtime()
    task_id = await _dispatched(runtime)
    probe.error = ResultUnreadable("no content")

    _report_unavailable(runtime, task_id)
    scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.FAILED
    assert (record.error or "").startswith("input_unreadable:")
    assert record.attempts == 0


@pytest.mark.anyio
async def test_a_store_control_cannot_reach_either_holds_the_task_blaming_no_one() -> (
    None
):
    runtime, scheduler, probe = _attributing_runtime()
    task_id = await _dispatched(runtime)
    probe.error = ResultUnavailable("store down")

    _report_unavailable(runtime, task_id)
    scheduler.run_due()

    assert runtime._tasks[task_id].status == TaskStatus.PENDING
    assert _ready(runtime) is None
    assert scheduler.pending(runtime._tasks[task_id].workflow_id)

    probe.error = None
    scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.failed_workers == []
    assert record.attempts == 0
    assert _ready(runtime) == task_id


@pytest.mark.anyio
async def test_a_task_cancelled_while_held_is_left_settled() -> None:
    runtime, scheduler, _probe = _attributing_runtime()
    task_id = await _dispatched(runtime)
    _report_unavailable(runtime, task_id)

    runtime.cancel_workflow(runtime._tasks[task_id].workflow_id)
    scheduler.run_due()

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert _ready(runtime) is None
