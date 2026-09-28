"""A resident inference or embedding leaf suspended on a call its worker holds ends DONE
when that worker drains after the call's outcome commits, and fails when the worker
leaves first."""

import asyncio
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from server.task.models import TaskStatus
from shared.harness import HarnessResultKind
from shared.schemas.event import WorkerEvent
from shared.tasks.task_type import TaskType
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import _TS, _register
from tests.server.task.test_worker_originated_boundary import _runtime
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors.episode_support import EpisodeStepResult
from worker.executors.inference.resolution import resolve_task_contract
from worker.executors.service_leaf_executor import ServiceLeafExecutor
from worker.resident import ResidentRequestStore

_LEAF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: leaf}}
spec:
  graph:
    nodes:
      - name: gen
        spec:
          taskType: {task_type}
          model: {{source: {{identifier: Qwen/Qwen3-4B}}}}
          data: {data}
          service: {{mode: resident}}
"""
_SHAPES = {
    "inference": ('{type: list, items: ["hello"]}', TaskType.INFERENCE),
    "embedding": ('{type: list, items: ["a", "b"]}', TaskType.EMBEDDING),
}
_ANSWER = '{"answer": "resident completion"}'


class _Leaf:
    """A resident leaf driven through its real worker step on a runtime whose resident
    origination and credit release are recorded."""

    def __init__(self, shape: str) -> None:
        self.shape = shape
        self.runtime = _runtime()
        self.originated: list[Any] = []
        self.releases: list[bool] = []
        self.runtime._resident_originate = self._originate
        self.runtime.set_resident_terminal_hook(
            lambda _invocation, failed: self.releases.append(failed)
        )

    def _originate(self, env: Any) -> bool:
        self.originated.append(env)
        return True

    async def register(self) -> str:
        data, task_type = _SHAPES[self.shape]
        _, ids = await _register(
            self.runtime, _LEAF.format(task_type=task_type.value, data=data)
        )
        self.task = ids["gen"]
        return self.task

    def step(self, worker: str) -> EpisodeStepResult:
        """Dispatch the leaf to ``worker`` and run its executor step there."""
        runtime, task = self.runtime, self.task
        dispatch = runtime.service_episode_dispatch(task)
        engine = runtime.orchestration_engine(runtime._tasks[task].workflow_id)
        assert dispatch is not None and engine is not None
        engine.on_dispatched(task, worker)
        record = runtime._tasks[task]
        record.assigned_worker = worker
        record.status = TaskStatus.DISPATCHED
        lifecycle = mock.MagicMock()
        lifecycle.resident_requests = ResidentRequestStore()
        executor = ServiceLeafExecutor(make_worker_config(), lifecycle=lifecycle)
        message = make_worker_task_message(
            record.task.model_dump(mode="json", exclude_none=True)["spec"],
            task_type=_SHAPES[self.shape][1],
            task_id=task,
            service_episode=dispatch.model_dump(mode="json"),
        )
        if (contract := runtime.declared_contract(task)) is not None:
            message.declared_contract = contract
            resolved = resolve_task_contract(message)
            assert resolved is not None
            message.resolved_contract = resolved.request
        result = executor.run(message, Path("/nonexistent"))
        assert isinstance(result, EpisodeStepResult)
        runtime.mark_succeeded(
            task,
            worker,
            {"agent_episode": result.harness_result.model_dump(mode="json")},
            _TS,
        )
        return result

    def drain(self, worker: str) -> None:
        _monitor(self.runtime)._handle_worker_event(
            WorkerEvent(type="UNREGISTER", worker_id=worker, graceful=True)
        )


@pytest.mark.parametrize("shape", ["inference", "embedding"])
def test_a_leaf_whose_drained_worker_finished_its_call_ends_done(shape: str) -> None:
    async def run() -> None:
        leaf = _Leaf(shape)
        task = await leaf.register()
        first = leaf.step("wkr-1")
        assert first.harness_result.kind is HarnessResultKind.BOUNDARY
        (env,) = leaf.originated

        # The draining worker's outcome commits before its UNREGISTER.
        assert leaf.runtime.settle_episode_invocation(
            task, env.call_correlation, _ANSWER
        )
        leaf.drain("wkr-1")
        record = leaf.runtime.get_record(task)
        assert record is not None and record.status is TaskStatus.PENDING

        second = leaf.step("wkr-2")
        record = leaf.runtime.get_record(task)
        assert record is not None and record.status is TaskStatus.DONE
        assert second.value == _ANSWER
        assert leaf.releases == [False]

    asyncio.run(run())


@pytest.mark.parametrize("shape", ["inference", "embedding"])
def test_a_leaf_whose_worker_left_before_its_outcome_fails(shape: str) -> None:
    async def run() -> None:
        leaf = _Leaf(shape)
        task = await leaf.register()
        leaf.step("wkr-1")
        (env,) = leaf.originated

        leaf.drain("wkr-1")
        late = leaf.runtime.settle_episode_invocation(
            task, env.call_correlation, _ANSWER
        )

        record = leaf.runtime.get_record(task)
        assert record is not None and record.status is TaskStatus.FAILED
        assert "ambiguity-terminal" in (record.error or "")
        assert not late

    asyncio.run(run())
