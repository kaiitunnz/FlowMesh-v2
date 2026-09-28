"""Losing the origin worker of a resident boundary releases the credit it holds."""

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import InvocationState, LedgerSnapshot
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import HarnessCapsule
from shared.private_state import PrivateStateSealReport
from shared.schemas.event import WorkerEvent
from tests.server.result_store import make_result_reader
from tests.server.task.test_private_state_ledger import _manifest
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _NoopSecretVault,
    _register,
)
from tests.server.task.test_worker_originated_boundary import (
    _HOLDER,
    _MODEL_SCRIPT,
    _runtime,
    _WorkerStub,
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


def _originating(runtime: TaskRuntime) -> list[Any]:
    """Accept every resident origination, keeping each envelope."""
    originated: list[Any] = []

    def originate(env: Any) -> bool:
        originated.append(env)
        return True

    runtime._resident_originate = originate
    return originated


def _capture_resident_boundary(
    runtime: TaskRuntime, task_id: str, seal_in: Path | None = None
) -> None:
    """Run the agent's step as its worker does: the resident request stays on the
    worker and only its digest reaches control, with the private state the worker
    sealed under ``seal_in`` when given."""
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
    payload: dict[str, Any] = {"agent_episode": result.model_dump(mode="json")}
    if seal_in is not None and (attachment := dispatch.private_state_attachment):
        manifest = _manifest(
            seal_in, attachment.reference_id, attachment.generation + 1
        )
        payload["agent_episode_private_state"] = PrivateStateSealReport(
            manifest=manifest, write_epoch=attachment.write_epoch
        ).model_dump(mode="json")
    runtime.mark_succeeded(task_id, "wkr-1", payload, _TS)


def test_losing_the_origin_worker_releases_the_resident_credit_once_durable() -> None:
    async def run() -> None:
        runtime = _runtime()
        originated = _originating(runtime)
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
        registry = cast(FakeRegistry, runtime._workflow_registry)
        stored = LedgerSnapshot.model_validate_json(registry.ledger_blobs[workflow_id])
        durable = next(
            i.state for i in stored.invocations if i.invocation_id == env.invocation_id
        )
        assert durable is InvocationState.TERMINAL
        assert (
            runtime.resident_invocation_completed(workflow_id, env.invocation_id)
            is False
        )

    asyncio.run(run())


def test_a_failed_save_holds_the_credit_until_the_next_save_succeeds() -> None:
    async def run() -> None:
        runtime = _runtime()
        originated = _originating(runtime)
        releases: list[str] = []
        runtime.set_resident_terminal_hook(lambda inv, _failed: releases.append(inv))
        workflow_id, ids = await _register(runtime, _RESIDENT_WF)
        _capture_resident_boundary(runtime, ids["writer"])
        (env,) = originated
        registry = cast(FakeRegistry, runtime._workflow_registry)
        save = registry.save_ledger_snapshot

        def down(workflow_id: str, snapshot: LedgerSnapshot) -> None:
            raise RuntimeError("control redis unavailable")

        registry.save_ledger_snapshot = down  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            runtime.recover_tasks_for_worker("wkr-1")
        assert releases == []

        registry.save_ledger_snapshot = save  # type: ignore[method-assign]
        with runtime._cv:
            runtime._save_ledger_locked(workflow_id)
        runtime._release_pending_terminations()
        assert releases == [env.invocation_id]

    asyncio.run(run())


def test_an_agent_whose_drained_worker_finished_its_resident_call_keeps_it() -> None:
    async def run() -> None:
        runtime = _runtime()
        originated = _originating(runtime)
        releases: list[tuple[str, bool]] = []
        # The fenced terminal releases the credit and reaps the origin's request.
        runtime.set_resident_terminal_hook(
            lambda inv, failed: releases.append((inv, failed))
        )
        _, ids = await _register(runtime, _RESIDENT_WF)
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer)
        (env,) = originated

        assert runtime.settle_episode_invocation(
            writer, env.call_correlation, "a completion"
        )
        assert releases == [(env.invocation_id, False)]
        _monitor(runtime)._handle_worker_event(
            WorkerEvent(type="UNREGISTER", worker_id="wkr-1", graceful=True)
        )

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.PENDING
        engine = runtime.orchestration_engine(record.workflow_id)
        assert engine is not None
        _, outcomes = engine.episode_context(writer)
        assert [o.value for o in outcomes] == ["a completion"]
        assert releases == [(env.invocation_id, False)]

    asyncio.run(run())


def test_a_resident_call_whose_settle_a_crash_cut_short_originates_again(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        originated = _originating(runtime)
        _, ids = await _register(runtime, _RESIDENT_WF)
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer, seal_in=tmp_path)
        (env,) = originated
        registry = cast(FakeRegistry, runtime._workflow_registry)
        save = registry.save_ledger_snapshot

        def crash(*_: Any, **__: Any) -> None:
            raise ConnectionError("crash before the ledger save")

        registry.save_ledger_snapshot = crash  # type: ignore[method-assign]
        with pytest.raises(ConnectionError):
            runtime.settle_episode_invocation(writer, env.call_correlation, "done")
        registry.save_ledger_snapshot = save  # type: ignore[method-assign]

        restored = TaskRuntime(
            cast(Any, registry),
            cast(Any, _WorkerStub()),
            OrchestrationConfig(),
            make_result_reader(),
            logging.getLogger("resident-test"),
            secret_vault=cast(Any, _NoopSecretVault()),
        )
        reoriginated = _originating(restored)
        await restored.rehydrate()

        assert [e.call_correlation for e in reoriginated] == [env.call_correlation]
        # The resident delivery reaches the origin worker through the task's record.
        record = restored.get_record(writer)
        assert record is not None and record.status is TaskStatus.DISPATCHED
        assert record.assigned_worker == "wkr-1"
        assert restored.settle_episode_invocation(writer, env.call_correlation, "done")
        dispatch = restored.agent_episode_dispatch(writer, _HOLDER)
        assert dispatch is not None
        assert [o.value for o in dispatch.delivered_outcomes] == ["done"]

    asyncio.run(run())


def test_a_resident_call_control_cannot_originate_reaps_its_request(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _RESIDENT_WF)
        writer = ids["writer"]

        _capture_resident_boundary(runtime, writer, seal_in=tmp_path)

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.FAILED
        frames = cast(Any, runtime._worker_registry).frames
        assert [(target, payload) for target, kind, payload in frames] == [
            ("wkr-1", {"task_id": writer, "call_correlation": "m0"})
        ]
        assert [kind for _, kind, _ in frames] == ["resident_reap"]

    asyncio.run(run())
