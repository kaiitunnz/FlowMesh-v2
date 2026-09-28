"""A cancel its worker reports without the server having requested one returns the
task to the queue, as a lost dispatch does."""

import threading
from typing import Any, cast

import pytest

from server.orchestration.state import WorkItemStatus
from server.task.models import EventEffect, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import (
    _TS,
    LINEAR,
    FakeRegistry,
    _drain,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _runtime,
    _worker,
)
from tests.server.task.test_v2_region_failure import _HEAD, _JOINS, _spawn_join


def _cancelled_by_worker(
    runtime: TaskRuntime, task_id: str, dispatch_id: str = "dsp-1"
) -> EventEffect:
    record_dispatch(runtime, task_id, cast(Any, _worker()), dispatch_id)
    return runtime.mark_cancelled(task_id, "wkr-1", {}, _TS, dispatch_id).effect


def _next(runtime: TaskRuntime) -> str | None:
    if runtime.ready_queue_length() == 0:
        return None
    return runtime.next_ready(threading.Event(), timeout=0.01)


@pytest.mark.anyio
async def test_a_workflow_whose_task_its_worker_gave_up_reruns_it_and_settles() -> None:
    runtime = _runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, LINEAR)
    assert _next(runtime) == ids["a"]

    assert _cancelled_by_worker(runtime, ids["a"]) is EventEffect.RETURNED

    record = runtime.get_record(ids["a"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert record.attempts == 0 and record.failed_workers == []
    work_item = runtime.orchestration_engine(workflow_id).work_item(ids["a"])  # type: ignore[union-attr]
    assert work_item is not None and work_item.status is not WorkItemStatus.CANCELLED
    assert _drain(runtime) == [ids["a"], ids["b"], ids["c"]]
    assert all(
        runtime.get_record(task_id).status == TaskStatus.DONE  # type: ignore[union-attr]
        for task_id in ids.values()
    )
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_spawned_child_its_worker_gave_up_reruns_alone() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(
        runtime, _HEAD + _spawn_join(_JOINS["all_settled"])
    )
    record_dispatch(runtime, ids["a"], cast(Any, _worker()))
    runtime.mark_succeeded(
        ids["a"], "wkr-1", _planned(runtime, ids["a"], ["x", "y"]), _TS
    )
    kids = _pop_ready(runtime)
    assert len(kids) == 2
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    scope = engine.scope_for("fan")

    assert _cancelled_by_worker(runtime, kids[0]) is EventEffect.RETURNED

    assert engine.scope_for("fan") == scope
    for kid in kids:
        work_item = engine.work_item(kid)
        assert work_item is not None
        assert work_item.status is not WorkItemStatus.CANCELLED
    sibling = runtime.get_record(kids[1])
    assert sibling is not None and sibling.status == TaskStatus.PENDING
    assert _pop_ready(runtime) == [kids[0]]

    for kid in kids:
        record_dispatch(runtime, kid, cast(Any, _worker()), f"dsp-{kid}")
        runtime.mark_succeeded(kid, "wkr-1", {}, _TS, f"dsp-{kid}")
    _drain(runtime)
    record = runtime.get_record(ids["after"])
    assert record is not None and record.status == TaskStatus.DONE
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
@pytest.mark.parametrize("unregister_first", [False, True])
async def test_a_drained_worker_returns_its_task_whichever_report_lands_first(
    unregister_first: bool,
) -> None:
    runtime = _runtime(FakeRegistry())
    monitor = _monitor(runtime)
    workflow_id, ids = await _register(runtime, LINEAR)
    task_id = ids["a"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")
    cancelled = TaskEvent(
        type="TASK_CANCELLED",
        task_id=task_id,
        worker_id="wkr-1",
        dispatch_id="dsp-1",
        ts=_TS,
    )
    unregistered = WorkerEvent(type="UNREGISTER", worker_id="wkr-1")

    for event in (
        (unregistered, cancelled) if unregister_first else (cancelled, unregistered)
    ):
        if isinstance(event, TaskEvent):
            monitor.handle_task_event(event)
        else:
            monitor._handle_worker_event(event)

    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.PENDING
    assert _next(runtime) == task_id
    assert _next(runtime) is None
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-2")), "dsp-2")
    runtime.mark_succeeded(task_id, "wkr-2", {}, _TS, "dsp-2")
    _drain(runtime, "wkr-2")
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_requested_cancel_still_settles_cancelled() -> None:
    runtime = _runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, LINEAR)
    assert _next(runtime) == ids["a"]
    record_dispatch(runtime, ids["a"], cast(Any, _worker()), "dsp-1")
    runtime.cancel_workflow(workflow_id)

    outcome = runtime.mark_cancelled(ids["a"], "wkr-1", {}, _TS, "dsp-1")

    assert outcome.effect is EventEffect.APPLIED
    assert all(
        runtime.get_record(task_id).status == TaskStatus.CANCELLED  # type: ignore[union-attr]
        for task_id in ids.values()
    )
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_returned_task_reruns_after_a_restart() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    assert _next(runtime) == ids["a"]
    assert _cancelled_by_worker(runtime, ids["a"]) is EventEffect.RETURNED

    restored = _runtime(registry)
    await restored.rehydrate()

    record = restored.get_record(ids["a"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert _drain(restored) == [ids["a"], ids["b"], ids["c"]]
    assert restored.workflow_settlement(workflow_id).settled
