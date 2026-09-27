"""A child its region's residual policy cancels is cancelled as a task, not only in the
ledger."""

import asyncio
from typing import Any, cast

import pytest

from server.orchestration.state import InvocationState, LedgerSnapshot
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from server.task.v2.representations.operators import JoinRegion
from shared.harness import BoundaryEventKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_agent_episode_runtime import _SCRIPT, _step
from tests.server.task.test_resident_origin_loss import (
    _capture_resident_boundary,
    _originating,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _worker,
)
from tests.server.task.test_v2_region_failure import _HEAD, _engine
from tests.server.task.test_worker_originated_boundary import _runtime
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_ANY_CANCEL = """
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [planner]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: any, residual: cancel}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


class _Interrupts:
    def __init__(self, runtime: TaskRuntime) -> None:
        self.sent: list[tuple[str, str]] = []
        registry = cast(Any, runtime._worker_registry)
        registry.publish_interrupt = lambda worker, message: self.sent.append(
            (message.task_id, worker.id)
        )


async def _fanned_out(
    registry: FakeRegistry, count: int
) -> tuple[TaskRuntime, str, dict[str, str], list[str]]:
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(runtime, _HEAD + _ANY_CANCEL)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    items = [f"h{index}" for index in range(count)]
    runtime.mark_succeeded(planner, "wkr-1", _planned(runtime, planner, items), _TS)
    return runtime, workflow_id, ids, _pop_ready(runtime)


def _win(runtime: TaskRuntime, winner: str) -> None:
    record_dispatch(runtime, winner, cast(Any, _worker()))
    runtime.mark_succeeded(winner, "wkr-1", {}, _TS)


def _finish(runtime: TaskRuntime, task_id: str) -> None:
    record_dispatch(runtime, task_id, cast(Any, _worker()))
    runtime.mark_succeeded(task_id, "wkr-1", {}, _TS)


def _status(runtime: TaskRuntime, task_id: str) -> str | None:
    record = runtime.get_record(task_id)
    return record.status if record is not None else None


@pytest.mark.anyio
async def test_a_pending_residual_child_is_cancelled_and_never_dispatched() -> None:
    registry = FakeRegistry()
    runtime, workflow_id, ids, (winner, loser) = await _fanned_out(registry, 2)
    with runtime._cv:
        # The loser waits in the ready queue for a worker.
        runtime._enqueue_ready_locked(loser)

    _win(runtime, winner)

    assert _pop_ready(runtime) == [ids["after"]]
    assert _status(runtime, loser) == TaskStatus.CANCELLED
    persisted = registry.load_task_states(loser)[0]
    assert persisted is not None and persisted.record.status == TaskStatus.CANCELLED
    _finish(runtime, ids["after"])
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_running_residual_child_is_interrupted_and_settles_cancelled() -> None:
    runtime, workflow_id, ids, (winner, loser) = await _fanned_out(FakeRegistry(), 2)
    interrupts = _Interrupts(runtime)
    record_dispatch(runtime, loser, cast(Any, _worker("wkr-2")))

    _win(runtime, winner)

    assert _status(runtime, loser) == TaskStatus.CANCELLING
    assert interrupts.sent == [(loser, "wkr-2")]
    _finish(runtime, _pop_ready(runtime)[0])
    assert not runtime.workflow_settlement(workflow_id).settled
    record = runtime.get_record(loser)
    assert record is not None
    runtime.mark_succeeded(loser, "wkr-2", {}, _TS, record.dispatch_id)
    assert _status(runtime, loser) == TaskStatus.CANCELLED
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.parametrize("state", ["pending", "running"])
def test_a_restart_cancels_a_residual_child_a_crash_left_behind(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime, workflow_id, ids, (winner, loser) = await _fanned_out(registry, 2)
        if state == "running":
            record_dispatch(runtime, loser, cast(Any, _worker("wkr-2")))
        # A stored workflow whose residual cancel reached only the ledger.
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime, "_cancel_residual_locked", lambda *_: False, raising=False
            )
            _win(runtime, winner)

        restored = _live_runtime(registry, "restored", reader=runtime._results)
        interrupts = _Interrupts(restored)
        assert await restored.rehydrate() == 1

        if state == "running":
            assert _status(restored, loser) == TaskStatus.CANCELLING
            assert interrupts.sent == [(loser, "wkr-2")]
            record = restored.get_record(loser)
            assert record is not None
            restored.mark_succeeded(loser, "wkr-2", {}, _TS, record.dispatch_id)
        assert _status(restored, loser) == TaskStatus.CANCELLED
        assert _pop_ready(restored) == [ids["after"]]
        _finish(restored, ids["after"])
        assert restored.workflow_settlement(workflow_id).settled

    asyncio.run(run())


def test_a_restart_interrupts_a_task_whose_cancel_interrupt_was_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime, workflow_id, _, (running, _) = await _fanned_out(registry, 2)
        record_dispatch(runtime, running, cast(Any, _worker("wkr-2")))
        # The root crashes after the cancel is durable, before it interrupts the worker.
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_release_terminated_work", lambda *_: None)
            runtime.cancel_workflow(workflow_id)

        restored = _live_runtime(registry, "restored", reader=runtime._results)
        interrupts = _Interrupts(restored)
        assert await restored.rehydrate() == 1

        assert _status(restored, running) == TaskStatus.CANCELLING
        assert interrupts.sent == [(running, "wkr-2")]

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_restart_runs_on_past_a_residual_cancel() -> None:
    registry = FakeRegistry()
    runtime, workflow_id, ids, (winner, _) = await _fanned_out(registry, 2)
    _win(runtime, winner)

    restored = _live_runtime(registry, "restored", reader=runtime._results)
    assert await restored.rehydrate() == 1

    # A residual cancel is no workflow cancel: what follows the join still runs.
    assert _pop_ready(restored) == [ids["after"]]
    _finish(restored, ids["after"])
    assert _status(restored, ids["after"]) == TaskStatus.DONE
    assert restored.workflow_settlement(workflow_id).settled


_RESIDENT_REVIEWER = """
      - name: lead
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: reviewer, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding: {mode: resident, service_model_ref: Qwen/Qwen3-4B}
"""


def test_an_agents_cancel_residual_releases_a_cancelled_childs_credit() -> None:
    async def run() -> None:
        runtime = _runtime()
        registry = cast(FakeRegistry, runtime._workflow_registry)
        originated = _originating(runtime)
        workflow_id, ids = await _register(runtime, _HEAD + _RESIDENT_REVIEWER)
        engine = _engine(runtime, workflow_id)
        join_op = f"{ids['lead']}:reviewer:spawn:join"
        join = engine._operators[join_op]
        assert isinstance(join, JoinRegion)
        engine._operators[join_op] = join.model_copy(
            update={"residual_policy": "cancel"}
        )

        def durable(invocation_id: str) -> InvocationState:
            stored = LedgerSnapshot.model_validate_json(
                registry.ledger_blobs[workflow_id]
            )
            return next(
                i.state for i in stored.invocations if i.invocation_id == invocation_id
            )

        releases: list[tuple[str, InvocationState]] = []
        runtime.set_resident_terminal_hook(
            lambda inv, _failed: releases.append((inv, durable(inv)))
        )
        lead = ids["lead"]
        # The lead spawns one reviewer, then completes with its region still open.
        spawn, _, complete = _SCRIPT
        adapter = ScriptedHarnessAdapter([spawn, complete], "v1")
        assert _pop_ready(runtime) == [lead]
        _step(runtime, adapter, lead)
        child = next(t for t in _pop_ready(runtime) if t != lead)
        _capture_resident_boundary(runtime, child)
        (env,) = originated

        _step(runtime, adapter, lead)

        assert _status(runtime, child) == TaskStatus.CANCELLED
        assert releases == [(env.invocation_id, InvocationState.TERMINAL)]
        assert runtime.workflow_settlement(workflow_id).settled

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_residual_cancel_settles_its_workflow_as_finished_work() -> None:
    registry = FakeRegistry()
    runtime, workflow_id, _, (winner, loser) = await _fanned_out(registry, 2)
    cancelled: list[str] = []
    commit = registry.commit_transition

    def spy(workflow_id: str, **kwargs: Any) -> None:
        cancelled.extend(kwargs.get("cancelled", ()))
        commit(workflow_id, **kwargs)

    registry.commit_transition = spy  # type: ignore[method-assign]
    _win(runtime, winner)

    # The loser leaves the remaining set without reading as a cancelled workflow.
    assert _status(runtime, loser) == TaskStatus.CANCELLED
    assert loser not in registry.remaining_of(workflow_id)
    assert cancelled == []


_NESTED_REVIEWERS = """
      - name: lead
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: reviewer, authority: {invoke: [model], delegate: [model]}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: sub, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: sub
        spec: {taskType: echo, data: {type: list, items: [s]}}
"""


def _spawner(region: str, value: str) -> ScriptedHarnessAdapter:
    return ScriptedHarnessAdapter(
        [
            ScriptedStep(
                op="boundary", kind=BoundaryEventKind.SPAWN, call="c0", region=region
            ),
            ScriptedStep(op="complete", value=value),
        ],
        "v1",
    )


@pytest.mark.parametrize("state", ["pending", "running"])
def test_a_residual_cancel_reaches_a_cancelled_agents_own_children(state: str) -> None:
    async def run() -> None:
        runtime = _live_runtime(FakeRegistry())
        interrupts = _Interrupts(runtime)
        workflow_id, ids = await _register(runtime, _HEAD + _NESTED_REVIEWERS)
        engine = _engine(runtime, workflow_id)
        join_op = f"{ids['lead']}:reviewer:spawn:join"
        join = engine._operators[join_op]
        assert isinstance(join, JoinRegion)
        engine._operators[join_op] = join.model_copy(
            update={"residual_policy": "cancel"}
        )
        lead, lead_adapter = ids["lead"], _spawner("reviewer", "done")
        assert _pop_ready(runtime) == [lead]
        _step(runtime, lead_adapter, lead)
        reviewer = next(t for t in _pop_ready(runtime) if t != lead)
        _step(runtime, _spawner("sub", "reviewed"), reviewer, worker="wkr-2")
        sub = next(t for t in _pop_ready(runtime) if t != reviewer)
        if state == "running":
            record_dispatch(runtime, sub, cast(Any, _worker("wkr-3")))
        else:
            with runtime._cv:
                runtime._enqueue_ready_locked(sub)

        # The lead completes while its reviewer and the reviewer's own child still run.
        _step(runtime, lead_adapter, lead)

        assert _status(runtime, reviewer) == TaskStatus.CANCELLED
        sub_wi = engine.work_item(sub)
        assert sub_wi is not None
        sub_scope = engine._scopes[engine._activations[sub_wi.activation_id].scope_id]
        assert (
            sub_scope.grant_id is not None
            and engine._grants[sub_scope.grant_id].revoked
        )
        if state == "running":
            assert _status(runtime, sub) == TaskStatus.CANCELLING
            assert interrupts.sent == [(sub, "wkr-3")]
            record = runtime.get_record(sub)
            assert record is not None
            runtime.mark_succeeded(sub, "wkr-3", {}, _TS, record.dispatch_id)
        assert _pop_ready(runtime) == []
        assert _status(runtime, sub) == TaskStatus.CANCELLED
        assert runtime.workflow_settlement(workflow_id).settled

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_committed_cancelling_task_stays_in_the_dispatched_set() -> None:
    registry = FakeRegistry()
    runtime, _, _, (_, loser) = await _fanned_out(registry, 2)
    record_dispatch(runtime, loser, cast(Any, _worker("wkr-2")))
    dispatched: list[str] = []
    commit = registry.commit_transition

    def spy(workflow_id: str, **kwargs: Any) -> None:
        dispatched.extend(kwargs.get("dispatched", ()))
        commit(workflow_id, **kwargs)

    registry.commit_transition = spy  # type: ignore[method-assign]
    with runtime._cv:
        runtime._tasks[loser].status = TaskStatus.CANCELLING
        runtime._commit_locked(loser)

    assert dispatched == [loser]
