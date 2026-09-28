"""A task its worker gives up without a requested cancel, as a draining worker does,
returns without spending an attempt, or fails as on its worker's loss when it is a v2
task that cannot safely re-run."""

import threading
from typing import Any, cast
from unittest.mock import patch

import pytest

from server.orchestration.state import WorkItemStatus
from server.task.models import EventEffect, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent, parse_event
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

_USAGE = {
    "started_at": _TS,
    "finished_at": _TS,
    "runtime_sec": 1.5,
    "hardware": {"gpu": {"driver_version": None, "cuda_version": None, "devices": []}},
    "cost_per_hour": 2.0,
    "total_cost": 0.5,
}

V1_CHAIN = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: v1-chain
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo}
"""


def _cancelled_by_worker(
    runtime: TaskRuntime, task_id: str, dispatch_id: str = "dsp-1"
) -> EventEffect:
    record_dispatch(runtime, task_id, cast(Any, _worker()), dispatch_id)
    return runtime.mark_cancelled(task_id, "wkr-1", {}, _TS, dispatch_id).effect


def _on_last_attempt(runtime: TaskRuntime, task_id: str) -> None:
    record = runtime.get_record(task_id)
    assert record is not None
    record.max_attempts = 1


def _given_up_reports(
    task_id: str, unregister_first: bool, worker_id: str = "wkr-1"
) -> tuple[TaskEvent | WorkerEvent, ...]:
    """A draining worker's two reports of a task it gave up, in arrival order."""
    cancelled = TaskEvent(
        type="TASK_CANCELLED",
        task_id=task_id,
        worker_id=worker_id,
        dispatch_id="dsp-1" if worker_id == "wkr-1" else f"dsp-{worker_id}",
        ts=_TS,
    )
    left = WorkerEvent(type="UNREGISTER", worker_id=worker_id, graceful=True)
    return (left, cancelled) if unregister_first else (cancelled, left)


def _deliver(monitor: Any, events: tuple[Any, ...]) -> None:
    for event in events:
        if isinstance(event, TaskEvent):
            monitor.handle_task_event(event)
        else:
            monitor._handle_worker_event(event)


def _statuses(runtime: TaskRuntime, task_ids: Any) -> set[str]:
    statuses = set()
    for task_id in task_ids:
        record = runtime.get_record(task_id)
        assert record is not None
        statuses.add(record.status)
    return statuses


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
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(ids["a"])
    assert work_item is not None and work_item.status is not WorkItemStatus.CANCELLED
    assert _drain(runtime) == [ids["a"], ids["b"], ids["c"]]
    assert _statuses(runtime, ids.values()) == {TaskStatus.DONE}
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
@pytest.mark.parametrize("unregister_first", [False, True])
async def test_a_spawned_child_its_worker_gave_up_reruns_alone(
    unregister_first: bool,
) -> None:
    runtime = _live_runtime(FakeRegistry())
    monitor = _monitor(runtime)
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

    record_dispatch(runtime, kids[0], cast(Any, _worker("wkr-9")), "dsp-wkr-9")
    _deliver(monitor, _given_up_reports(kids[0], unregister_first, "wkr-9"))

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
@pytest.mark.parametrize("workflow", [LINEAR, V1_CHAIN], ids=["v2", "v1"])
async def test_a_drained_worker_returns_its_last_attempt_whichever_report_lands_first(
    unregister_first: bool, workflow: str
) -> None:
    runtime = _runtime(FakeRegistry())
    monitor = _monitor(runtime)
    workflow_id, ids = await _register(runtime, workflow)
    task_id = ids["a"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")
    _on_last_attempt(runtime, task_id)

    _deliver(monitor, _given_up_reports(task_id, unregister_first))

    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert _next(runtime) == task_id
    assert _next(runtime) is None
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-2")), "dsp-2")
    runtime.mark_succeeded(task_id, "wkr-2", {}, _TS, "dsp-2")
    _drain(runtime, "wkr-2")
    assert _statuses(runtime, ids.values()) == {TaskStatus.DONE}
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
    assert _statuses(runtime, ids.values()) == {TaskStatus.CANCELLED}
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "unregistered",
    [
        WorkerEvent(type="UNREGISTER", worker_id="wkr-1"),
        parse_event({"type": "UNREGISTER", "worker_id": "wkr-1", "payload": {}}),
    ],
    ids=["supervisor", "older_worker"],
)
async def test_a_worker_lost_without_a_graceful_unregister_spends_an_attempt(
    unregistered: Any,
) -> None:
    runtime = _runtime(FakeRegistry())
    monitor = _monitor(runtime)
    workflow_id, ids = await _register(runtime, V1_CHAIN)
    task_id = ids["a"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")
    _on_last_attempt(runtime, task_id)

    monitor._handle_worker_event(unregistered)

    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.FAILED
    assert runtime.workflow_settlement(workflow_id).settled


SSH_THEN_ECHO = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: ssh-then-echo}
spec:
  graph:
    nodes:
      - name: session
        spec: {taskType: ssh}
      - name: after
        dependsOn: [session]
        spec: {taskType: echo, data: {type: list, items: [after]}}
"""


@pytest.mark.anyio
@pytest.mark.parametrize("unregister_first", [False, True])
async def test_a_drained_external_effect_fails_whichever_report_lands_first(
    unregister_first: bool,
) -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    workflow_id, ids = await _register(runtime, SSH_THEN_ECHO)
    task_id = ids["session"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")

    _deliver(monitor, _given_up_reports(task_id, unregister_first))

    for rt in (runtime, _runtime(registry)):
        if rt is not runtime:
            await rt.rehydrate()
        session = rt.get_record(task_id)
        after = rt.get_record(ids["after"])
        assert session is not None and session.status == TaskStatus.FAILED
        assert session.error == "ambiguity-terminal effect"
        assert after is not None and after.status == TaskStatus.FAILED
        assert after.error == f"Dependency {task_id} failed"
        assert rt.ready_queue_length() == 0
        assert rt.workflow_settlement(workflow_id).settled


def _dispatched_unsaved(
    registry: FakeRegistry, runtime: TaskRuntime, task: str
) -> None:
    """Dispatch a task, then lose the ledger save that recorded the dispatch."""
    saved = dict(registry.ledger_blobs)
    record_dispatch(runtime, task, cast(Any, _worker()), "dsp-1")
    registry.ledger_blobs.clear()
    registry.ledger_blobs.update(saved)


@pytest.mark.anyio
@pytest.mark.parametrize("lost", ["given_up", "crashed"])
@pytest.mark.parametrize(
    ("workflow", "node", "rerun"),
    [(LINEAR, "a", True), (SSH_THEN_ECHO, "session", False)],
    ids=["replayable", "external_effect"],
)
async def test_a_dispatch_its_ledger_never_saved_resolves_after_a_restart(
    lost: str, workflow: str, node: str, rerun: bool
) -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, workflow)
    task_id = ids[node]
    assert _next(runtime) == task_id
    _dispatched_unsaved(registry, runtime, task_id)
    restored = _runtime(registry)
    await restored.rehydrate()

    if lost == "given_up":
        _monitor(restored).handle_task_event(
            TaskEvent(
                type="TASK_CANCELLED",
                task_id=task_id,
                worker_id="wkr-1",
                dispatch_id="dsp-1",
                ts=_TS,
            )
        )
    else:
        _monitor(restored).record_worker_losses(
            "wkr-1", restored.recover_tasks_for_worker("wkr-1").resolved
        )

    record = restored.get_record(task_id)
    assert record is not None
    if rerun:
        assert record.status == TaskStatus.PENDING
        assert _drain(restored) == [ids["a"], ids["b"], ids["c"]]
    else:
        assert record.status == TaskStatus.FAILED
    assert restored.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_given_up_task_that_fails_is_billed_for_its_dispatch() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, SSH_THEN_ECHO)
    task_id = ids["session"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")

    outcome = runtime.mark_cancelled(task_id, "wkr-1", _USAGE, _TS, "dsp-1")

    assert outcome.effect is EventEffect.FAILED
    [(billed, usage)] = outcome.usages
    assert (billed, usage.status, usage.total_cost) == (task_id, "FAILED", 0.5)
    restored = _runtime(registry)
    await restored.rehydrate()
    record = restored.get_record(task_id)
    assert record is not None and record.usages == [usage]


@pytest.mark.anyio
async def test_the_monitor_emits_the_usage_of_a_given_up_task_that_fails() -> None:
    runtime = _runtime(FakeRegistry())
    monitor = _monitor(runtime)
    _, ids = await _register(runtime, SSH_THEN_ECHO)
    task_id = ids["session"]
    assert _next(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")
    emitted: list[Any] = []

    with patch.object(monitor, "_schedule_emit_usage", side_effect=emitted.extend):
        monitor.handle_task_event(
            TaskEvent(
                type="TASK_CANCELLED",
                task_id=task_id,
                worker_id="wkr-1",
                dispatch_id="dsp-1",
                ts=_TS,
                payload=_USAGE,
            )
        )

    assert [(billed, usage.status) for billed, usage in emitted] == [
        (task_id, "FAILED")
    ]
