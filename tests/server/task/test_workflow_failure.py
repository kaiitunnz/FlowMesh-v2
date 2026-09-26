"""A workflow control fails releases everything its work held, as a cancel does."""

import asyncio
from typing import Any, cast

import pytest

from server.orchestration.state import AttemptStatus, WorkItemStatus
from server.task.models import TaskStatus
from server.task.results import ResultUnreadable
from server.task.runtime import TaskRuntime, _InputCheck
from shared.content import reference_for
from shared.schemas.event import TaskFailureKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_agent_episode_runtime import _held_boundary
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _runtime,
    _worker,
    _WorkerRegistryStub,
)

_PARALLEL = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: parallel}
spec:
  graph:
    nodes:
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: side
        spec: {taskType: echo, data: {type: list, items: [s]}}
      - name: reviewer
        spec: {taskType: echo, data: {type: list, items: [tmpl]}}
      - name: fanout
        dependsOn: [planner]
        region: {kind: spawn, child: reviewer}
"""


def _fail(runtime: TaskRuntime, workflow_id: str) -> None:
    with runtime._cv:
        runtime._fail_workflow_locked(workflow_id, "fan-out producer unreadable")


def _fail_by_unreadable_fanout(runtime: TaskRuntime, planner: str) -> None:
    """Settle a spawn producer whose stored collection control cannot read."""
    record_dispatch(runtime, planner, cast(Any, _worker()), "dsp-p")
    payload = _planned(runtime, planner, ["a"])

    def _corrupt(binding: Any) -> Any:
        raise ResultUnreadable("corrupt")

    read = runtime._results.read
    runtime._results.read = _corrupt  # type: ignore[method-assign]
    runtime.mark_succeeded(planner, "wkr-1", payload, _TS, dispatch_id="dsp-p")
    runtime._results.read = read  # type: ignore[method-assign]


def test_a_held_resident_credit_is_released_once() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        releases: list[tuple[str, bool]] = []
        runtime.set_resident_terminal_hook(
            lambda invocation, failed: releases.append((invocation, failed))
        )
        workflow_id, writer, engine, env = await _held_boundary(runtime)

        _fail(runtime, workflow_id)

        assert engine.work_item(writer).status is WorkItemStatus.SETTLED
        assert len(releases) == 1 and releases[0][1] is True
        late = runtime.settle_episode_invocation(
            env.task_id, env.call_correlation, "resident:late"
        )
        assert late is False
        assert len(releases) == 1

    asyncio.run(run())


def test_the_agents_private_state_write_is_released() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, writer, engine, _env = await _held_boundary(runtime)
        assert engine.grant_private_state(writer, "wkr-1", 1) is not None

        _fail(runtime, workflow_id)

        assert engine.grant_private_state(writer, "wkr-1", 1) is None

    asyncio.run(run())


def test_the_agents_pending_operations_are_reaped() -> None:
    frames: list[Any] = []

    class _Workers(_WorkerRegistryStub):
        def publish_mediated_op(self, worker: Any, message: Any) -> int:
            frames.append(message)
            return 1

    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        runtime._worker_registry = cast(Any, _Workers())
        workflow_id, writer, _engine, env = await _held_boundary(runtime)
        runtime._pending_ops["mop-held"] = (writer, env.call_correlation, "wkr-1")

        _fail(runtime, workflow_id)

        assert runtime._pending_ops == {}
        assert [(f.frame_kind, f.payload["agent_task_id"]) for f in frames] == [
            ("reap", writer)
        ]

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_returned_attempt_keeps_its_outcome() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _PARALLEL)
    _pop_ready(runtime)
    side = ids["side"]
    record_dispatch(runtime, side, cast(Any, _worker()), "dsp-s")
    runtime.mark_started(side, "wkr-1", {}, _TS, dispatch_id="dsp-s")
    runtime.return_dispatch(side, "wkr-1", increment_retry=False, front=False)

    _fail_by_unreadable_fanout(runtime, ids["planner"])

    engine = runtime._engines[workflow_id]
    work_item = engine.work_item(side)
    assert work_item is not None
    assert [engine._attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED
    ]
    assert runtime._tasks[side].status == TaskStatus.FAILED


@pytest.mark.anyio
async def test_a_task_still_running_is_interrupted() -> None:
    interrupts: list[Any] = []

    class _Workers(_WorkerRegistryStub):
        def publish_interrupt(self, *args: Any) -> int:
            interrupts.append(args[1])
            return 1

    runtime = _live_runtime(FakeRegistry())
    runtime._worker_registry = cast(Any, _Workers())
    _, ids = await _register(runtime, _PARALLEL)
    _pop_ready(runtime)
    side = ids["side"]
    record_dispatch(runtime, side, cast(Any, _worker("wkr-2")), "dsp-s")
    runtime.mark_started(side, "wkr-2", {}, _TS, dispatch_id="dsp-s")

    _fail_by_unreadable_fanout(runtime, ids["planner"])

    assert [(i.task_id, i.worker_id) for i in interrupts] == [(side, "wkr-2")]
    assert runtime._tasks[side].status == TaskStatus.FAILED
    late = runtime.fail_dispatch(
        side,
        "wkr-2",
        {},
        _TS,
        "dsp-s",
        error="interrupted",
        retryable=True,
        failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
    )
    assert runtime._tasks[side].status == TaskStatus.FAILED
    assert late.attempts == 0


@pytest.mark.anyio
async def test_a_held_input_check_is_dropped() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _PARALLEL)
    reference = reference_for("org", b"input", media_type="application/json")
    runtime._input_checks[ids["side"]] = _InputCheck("wkr-1", "dsp-s", (reference,))

    _fail(runtime, workflow_id)

    assert runtime._input_checks == {}
