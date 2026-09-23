"""One agent episode counts one task completion, whatever its step count.

An episode reports a worker success per run-to-yield step, and a turn completion that
carries or resumes a captured facade group reroutes that group rather than ending the
task, so the completion metric follows the success that settles the task.
"""

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

from server.orchestration.tool_dispatch import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeTurnGroup,
)
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from shared.harness import BoundaryEventKind, HarnessResult, HarnessResultKind
from shared.schemas.event import TaskEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime

_TS = "2026-09-16T00:00:00Z"

_AGENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: completion-metric}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _monitor(runtime, metrics: MagicMock) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.episode_completion"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=metrics,
        watchdog=MagicMock(),
    )


def _succeeded(task_id: str, step: HarnessResult, **extra) -> TaskEvent:
    payload = {"agent_episode": step.model_dump(mode="json"), **extra}
    return TaskEvent(
        type="TASK_SUCCEEDED",
        task_id=task_id,
        worker_id="wkr-1",
        payload=payload,
        ts=_TS,
    )


def _search_group(activation: str) -> FacadeTurnGroup:
    member = FacadeCallMember(
        ordinal=0,
        kind=BoundaryEventKind.INVOCATION,
        completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
        call_correlation=f"{activation}:0",
        harness_call_id="call0",
        tool_name="web_search",
        interface_or_region="search/v1",
        request_payload='{"query": "q"}',
    )
    return FacadeTurnGroup(
        group_id=f"{activation}:0",
        activation_id=activation,
        turn_id="0",
        members=(member,),
    )


def _routed_group(runtime: Any, task_id: str) -> bool:
    engine = runtime.orchestration_engine(runtime.get_record(task_id).workflow_id)
    work_item = engine.work_item(task_id)
    return work_item is not None and work_item.pending_outcome_group is not None


def _episode_task() -> tuple:
    runtime = _runtime(FakeRegistry())
    _, ids = asyncio.run(_register(runtime, _AGENT_WF))
    record_dispatch(runtime, ids["writer"])
    return runtime, ids["writer"]


def test_a_multi_step_episode_counts_one_completion() -> None:
    runtime, task_id = _episode_task()
    metrics = MagicMock()
    monitor = _monitor(runtime, metrics)
    steps = [
        HarnessResult(kind=HarnessResultKind.YIELD),
        HarnessResult(kind=HarnessResultKind.YIELD),
        HarnessResult(kind=HarnessResultKind.COMPLETION, value="done"),
    ]
    for step in steps:
        monitor._handle_task_event(_succeeded(task_id, step))
        record_dispatch(runtime, task_id)
    assert metrics.record_task_event.call_count == 1


def test_a_turn_completion_carrying_a_facade_group_counts_no_completion() -> None:
    runtime, task_id = _episode_task()
    metrics = MagicMock()
    monitor = _monitor(runtime, metrics)
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value=None)
    event = _succeeded(
        task_id,
        completion,
        agent_episode_facade_group=_search_group(task_id).model_dump(mode="json"),
    )

    monitor._handle_task_event(event)

    # The turn reroutes its group and the episode runs on, so nothing settled here.
    assert metrics.record_task_event.call_count == 0
    assert runtime.get_record(task_id).status != TaskStatus.DONE
    assert _routed_group(runtime, task_id)


def test_a_turn_completion_resuming_a_stashed_facade_group_counts_no_completion() -> (
    None
):
    runtime, task_id = _episode_task()
    runtime.originate_facade_turn_group(task_id, _search_group(task_id))
    metrics = MagicMock()
    monitor = _monitor(runtime, metrics)
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value=None)

    # The gateway stashed this group, so it rides no payload: only the runtime knows.
    monitor._handle_task_event(_succeeded(task_id, completion))

    assert metrics.record_task_event.call_count == 0
    assert runtime.get_record(task_id).status != TaskStatus.DONE
    assert _routed_group(runtime, task_id)


def test_a_rerouted_facade_turn_bills_its_dispatch_as_in_flight() -> None:
    runtime, task_id = _episode_task()
    runtime.originate_facade_turn_group(task_id, _search_group(task_id))
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value=None)

    success = runtime.mark_succeeded(
        task_id,
        "wkr-1",
        {
            "agent_episode": completion.model_dump(mode="json"),
            "started_at": _TS,
            "finished_at": _TS,
            "runtime_sec": 1.5,
            "hardware": {
                "gpu": {"driver_version": None, "cuda_version": None, "devices": []}
            },
            "cost_per_hour": 2.0,
            "total_cost": 0.5,
        },
        _TS,
    )
    assert success is not None
    usages = success.usages

    assert [usage.status for _, usage in usages] == [TaskStatus.DISPATCHED]


def test_an_ordinary_success_counts_one_completion() -> None:
    runtime, _ = _episode_task()
    metrics = MagicMock()
    monitor = _monitor(runtime, metrics)
    monitor._handle_task_event(
        TaskEvent(
            type="TASK_SUCCEEDED",
            task_id="tsk-unknown",
            worker_id="wkr-1",
            payload={"result": "ok"},
            ts=_TS,
        )
    )
    assert metrics.record_task_event.call_count == 1
