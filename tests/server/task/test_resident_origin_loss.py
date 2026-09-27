"""Losing the origin worker of a resident boundary releases the credit it holds."""

import asyncio
from typing import Any

from server.orchestration.state import InvocationState
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import HarnessCapsule
from tests.server.task.test_v2_orchestration import _TS, _register
from tests.server.task.test_worker_originated_boundary import (
    _HOLDER,
    _MODEL_SCRIPT,
    _runtime,
)
from worker.executors.harness.scripted import ScriptedHarnessAdapter
from worker.resident import capture_resident_request
from worker.resident.request_store import ResidentRequestStore

_RESIDENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: resident-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding: {mode: resident, service_model_ref: Qwen/Qwen3-4B}
"""


def _capture_resident_boundary(runtime: TaskRuntime, task_id: str) -> None:
    """Run the agent's step as its worker does: the resident request stays on the
    worker and only its digest reaches control."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id, _HOLDER)
    assert engine is not None and dispatch is not None
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    engine.on_dispatched(task_id, "wkr-1")
    record = runtime._tasks[task_id]
    record.assigned_worker = "wkr-1"
    record.status = TaskStatus.DISPATCHED
    result = ScriptedHarnessAdapter(_MODEL_SCRIPT, "v1").start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    result = capture_resident_request(ResidentRequestStore(), task_id, result)
    runtime.mark_succeeded(
        task_id, "wkr-1", {"agent_episode": result.model_dump(mode="json")}, _TS
    )


def test_losing_the_origin_worker_releases_the_resident_credit_once_durable() -> None:
    async def run() -> None:
        runtime = _runtime()
        originated: list[Any] = []
        runtime._resident_originate = originated.append  # type: ignore[assignment]
        releases: list[tuple[str, bool]] = []
        runtime.set_resident_terminal_hook(
            lambda inv, failed: releases.append((inv, failed))
        )
        workflow_id, ids = await _register(runtime, _RESIDENT_WF)
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer)
        (env,) = originated
        assert env.request_digest is not None

        runtime.recover_tasks_for_worker("wkr-1")

        record = runtime.get_record(writer)
        assert record is not None and record.status == TaskStatus.FAILED
        assert releases == [(env.invocation_id, True)]
        snapshot = runtime._workflow_registry.ledger_blobs[workflow_id]  # type: ignore[attr-defined]
        durable = next(
            i.state
            for i in type(runtime.orchestration_engine(workflow_id).to_snapshot())  # type: ignore[union-attr]
            .model_validate_json(snapshot)
            .invocations
            if i.invocation_id == env.invocation_id
        )
        assert durable is InvocationState.TERMINAL
        assert (
            runtime.resident_invocation_completed(workflow_id, env.invocation_id)
            is False
        )

    asyncio.run(run())
