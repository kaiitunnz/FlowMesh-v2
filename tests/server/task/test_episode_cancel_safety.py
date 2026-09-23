"""A cancelled agent episode is never re-admitted by a step success that races it.

A cancel moves an in-flight episode to CANCELLING and interrupts its worker, but the
worker may already have finished the step it was running and report it as a success.
That success settles the cancellation, not returning the episode to the queue,
and the per-step usage row it carries is tagged with the status the task is actually in.
"""

import asyncio
from typing import Any, cast

from server.orchestration import WorkItemStatus
from server.orchestration.tool_dispatch import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeTurnGroup,
)
from server.task.models import DispatchEnd, TaskStatus
from shared.harness import BoundaryEventKind, HarnessCapsule, HarnessResult
from shared.private_state import OwnerFence
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _runtime,
    _worker,
)
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_HOLDER = OwnerFence(worker_id="wkr-1", incarnation=1)
_TS = "2026-09-16T00:00:00Z"

_USAGE_PAYLOAD = {
    "started_at": _TS,
    "finished_at": _TS,
    "runtime_sec": 1.5,
    "hardware": {"gpu": {"driver_version": None, "cuda_version": None, "devices": []}},
    "cost_per_hour": 2.0,
    "total_cost": 0.5,
}

_AGENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: cancel-safety}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: reviewer
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec: {taskType: echo, data: {type: list, items: [placeholder]}}
"""

_SCRIPT = [
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN, call="c0", region="reviewer"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN_SEAL, call="c1", region="reviewer"
    ),
    ScriptedStep(op="complete", value="reviewed"),
]


def _run_step(runtime, adapter, task_id: str, worker: str = "wkr-1") -> HarnessResult:
    """Run one dispatch on the worker without reporting it to the server."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id, _HOLDER)
    assert engine is not None and dispatch is not None
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    record_dispatch(runtime, task_id, worker)
    runtime.mark_started(task_id, worker, {}, _TS)
    return adapter.start(task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes)


def test_a_step_success_racing_a_cancel_does_not_re_admit_the_episode() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        # The worker finishes a non-terminal step; the cancel lands before it reports.
        result = _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)
        assert runtime._tasks[writer].status == TaskStatus.CANCELLING

        success = runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )
        assert success is not None
        usages = success.usages

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.CANCELLED
        assert writer not in runtime._ready_index
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.CANCELLED
        # Every task has left the workflow's remaining set, so the workflow reaches a
        # terminal status rather than hanging on a task no dispatch will ever settle.
        assert registry.remaining_of(workflow_id) == set()
        assert [usage.status for _, usage in usages] == [TaskStatus.DISPATCHED]

    asyncio.run(run())


def test_cancelling_an_episode_suspended_on_a_boundary_settles_it() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )

        # No settler is installed, so the model boundary suspends the lane: the worker
        # holds no dispatch for this task and will report no terminal for it.
        result = _run_step(runtime, adapter, writer)
        runtime.mark_succeeded(
            writer, "wkr-1", {"agent_episode": result.model_dump(mode="json")}, _TS
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        assert engine.suspended_boundary_tasks() == [writer]

        runtime.cancel_workflow(workflow_id)

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.CANCELLED
        assert registry.remaining_of(workflow_id) == set()
        # A boundary outcome arriving afterwards settles nothing and revives nothing.
        assert runtime.settle_episode_invocation(writer, "m0", "late") is False
        assert runtime._tasks[writer].status == TaskStatus.CANCELLED

    asyncio.run(run())


def test_cancelling_an_episode_running_a_step_waits_for_its_worker() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        # The episode is running a step, so it holds no unsettled boundary: its worker
        # gets the interrupt and owns the terminal.
        _run_step(runtime, adapter, writer)
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        assert engine.suspended_boundary_tasks() == []

        runtime.cancel_workflow(workflow_id)

        assert runtime._tasks[writer].status == TaskStatus.CANCELLING

    asyncio.run(run())


def test_a_live_episode_step_bills_its_dispatch_as_in_flight() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        _, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        result = _run_step(runtime, adapter, writer)
        success = runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )
        assert success is not None
        usages = success.usages

        assert runtime._tasks[writer].status == TaskStatus.PENDING
        assert [usage.status for _, usage in usages] == [TaskStatus.DISPATCHED]

    asyncio.run(run())


def test_a_late_start_does_not_erase_a_cancellation() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        # The worker's start event lands after the cancel has already moved the task to
        # CANCELLING, so it arrives on a task the cancel is waiting to settle.
        result = _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)
        runtime.mark_started(writer, "wkr-1", {}, _TS)
        assert runtime._tasks[writer].status == TaskStatus.CANCELLING

        runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )

        assert runtime._tasks[writer].status == TaskStatus.CANCELLED
        assert writer not in runtime._ready_index
        assert registry.remaining_of(workflow_id) == set()

    asyncio.run(run())


def test_a_racing_dispatch_does_not_erase_a_cancellation() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        result = _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)
        runtime.mark_dispatched(writer, cast(Any, _worker("wkr-2")))
        assert runtime._tasks[writer].status == TaskStatus.CANCELLING
        assert runtime._tasks[writer].assigned_worker != "wkr-2"

        runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )

        assert runtime._tasks[writer].status == TaskStatus.CANCELLED
        assert writer not in runtime._ready_index
        assert registry.remaining_of(workflow_id) == set()

    asyncio.run(run())


def test_a_return_settles_a_cancelling_episode() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)
        end = runtime.return_dispatch(writer, None, increment_retry=False, front=True)

        assert end is DispatchEnd.CANCELLED
        assert runtime._tasks[writer].status == TaskStatus.CANCELLED
        assert writer not in runtime._ready_index

    asyncio.run(run())


def test_a_completion_racing_a_cancel_settles_cancelled() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [ScriptedStep(op="complete", value="done")], "v1"
        )

        # The worker ran the episode's last step; the cancel lands before it reports.
        result = _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)
        assert runtime._tasks[writer].status == TaskStatus.CANCELLING

        success = runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )
        assert success is not None
        usages = success.usages

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.CANCELLED
        assert record.error == "cancelled"
        assert [usage.status for _, usage in usages] == [TaskStatus.CANCELLED]
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.CANCELLED
        assert registry.remaining_of(workflow_id) == set()

    asyncio.run(run())


def test_a_failure_racing_a_cancel_settles_cancelled() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(_SCRIPT, "v1")

        _run_step(runtime, adapter, writer)
        runtime.cancel_workflow(workflow_id)

        impacted, usages = runtime.mark_failed(
            writer, "wkr-1", dict(_USAGE_PAYLOAD), _TS, error="worker blew up"
        )

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.CANCELLED
        assert record.error == "cancelled"
        assert impacted == []
        assert [usage.status for _, usage in usages] == [TaskStatus.CANCELLED]
        assert registry.remaining_of(workflow_id) == set()

    asyncio.run(run())


def test_a_completion_with_a_facade_group_racing_a_cancel_settles_cancelled() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [ScriptedStep(op="complete", value="done")], "v1"
        )

        # The worker finished the episode's last step and captured a facade group; the
        # cancel lands before it reports. The group must not route its members.
        result = _run_step(runtime, adapter, writer)
        group = FacadeTurnGroup(
            group_id=f"{writer}:0",
            activation_id=writer,
            turn_id="0",
            members=(
                FacadeCallMember(
                    ordinal=0,
                    kind=BoundaryEventKind.INVOCATION,
                    completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
                    call_correlation=f"{writer}:0:0",
                    harness_call_id="call0",
                    tool_name="web_search",
                    interface_or_region="search/v1",
                    request_payload='{"query": "q0"}',
                ),
            ),
        )
        runtime.originate_facade_turn_group(writer, group)
        runtime.cancel_workflow(workflow_id)
        assert runtime._tasks[writer].status == TaskStatus.CANCELLING

        success = runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": result.model_dump(mode="json"), **_USAGE_PAYLOAD},
            _TS,
        )
        assert success is not None
        usages = success.usages

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.CANCELLED
        assert record.pending_facade_group is None
        assert [usage.status for _, usage in usages] == [TaskStatus.CANCELLED]
        assert registry.remaining_of(workflow_id) == set()

    asyncio.run(run())
