"""Losing the origin worker of a resident boundary releases the credit it holds."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import InvocationState, LedgerSnapshot
from server.resident import ClaimState, ClaimTerminalReason
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import HarnessCapsule
from shared.private_state import PrivateStateSealReport
from shared.resident.reports import ResidentBootstrapAck, ResidentBootstrapOutcome
from shared.schemas.event import WorkerEvent
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.resident.test_service import _build
from tests.server.result_store import make_result_reader
from tests.server.runtime_helpers import manual_durability_retry
from tests.server.task.test_private_state_ledger import _manifest
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
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
    result = capture_resident_request(ResidentRequestStore(), task_id, result, None)
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

        runtime.recover_tasks_for_worker("wkr-1", spend_attempt=True)

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

        def down(
            workflow_id: str, snapshot: LedgerSnapshot, control: Any = None
        ) -> None:
            raise ConnectionError("control redis unavailable")

        registry.save_ledger_snapshot = down  # type: ignore[method-assign]
        runtime.recover_tasks_for_worker("wkr-1", spend_attempt=True)
        assert releases == []

        registry.save_ledger_snapshot = save  # type: ignore[method-assign]
        runtime._retry_durability(workflow_id)
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
        assert runtime.settle_episode_invocation(writer, env.call_correlation, "done")
        runtime.shutdown()
        registry.save_ledger_snapshot = save  # type: ignore[method-assign]

        restored = TaskRuntime(
            cast(Any, registry),
            cast(Any, _WorkerStub()),
            OrchestrationConfig(),
            make_result_reader(),
            logging.getLogger("resident-test"),
            credential_vault=InMemoryCredentialVault(),
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


def _wire_resident_service(
    runtime: TaskRuntime,
) -> tuple[Any, Any, Any, asyncio.AbstractEventLoop, list[Any]]:
    svc, stores, _, delivery = _build()
    svc._settle = runtime.settle_episode_invocation
    svc._redispatch = runtime.redispatch_episode_invocation
    svc._boundary_settleable = runtime.boundary_settleable
    assert svc._delivery is not None
    svc._delivery = replace(
        svc._delivery,
        origin_worker_of_task=lambda task_id: (
            record.assigned_worker
            if (record := runtime.get_record(task_id)) is not None
            else None
        ),
    )
    runtime.set_resident_terminal_hook(svc.on_invocation_terminal)
    loop = asyncio.new_event_loop()
    svc.bind_loop(loop)
    originated: list[Any] = []

    def originate(env: Any) -> bool:
        originated.append(env)
        return svc.originate(env)

    runtime._resident_originate = originate
    return svc, stores, delivery, loop, originated


def test_a_cancel_that_beats_its_resident_origination_holds_no_credit(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    svc, stores, delivery, loop, originated = _wire_resident_service(runtime)
    try:
        workflow_id, ids = loop.run_until_complete(_register(runtime, _RESIDENT_WF))
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer, seal_in=tmp_path)
        (env,) = originated
        # The cancel lands before the loop runs the origination it raced.
        runtime.cancel_workflow(workflow_id)
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()

    record = runtime.get_record(writer)
    assert record is not None and record.status is TaskStatus.CANCELLED
    assert all(
        claim.state is ClaimState.TERMINAL
        for claim in stores.claims.by_invocation(env.invocation_id)
    )
    assert not any(
        stores.credit_ledger.held(replica.replica_id)
        for replica in stores.directory.all()
    )
    assert [
        (worker, payload)
        for worker, kind, payload in delivery.relays
        if kind == "resident_reap"
    ] == [("wkr-1", {"task_id": writer, "call_correlation": env.call_correlation})]


def test_a_cancel_at_a_re_drive_s_check_reaps_after_the_credit_release(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    svc, stores, delivery, loop, originated = _wire_resident_service(runtime)
    held_at_reap: list[int] = []
    reclaim = svc._reclaim_adapter_slot

    def reclaim_after_release(attempt: Any) -> None:
        held_at_reap.append(stores.credit_ledger.held(attempt.replica_id))
        reclaim(attempt)

    svc._reclaim_adapter_slot = reclaim_after_release
    try:
        workflow_id, ids = loop.run_until_complete(_register(runtime, _RESIDENT_WF))
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer, seal_in=tmp_path)
        (env,) = originated
        loop.run_until_complete(asyncio.sleep(0.05))
        attempt = svc._attempts[env.invocation_id]
        loop.run_until_complete(
            svc._on_ack(
                ResidentBootstrapAck(
                    task_id=writer,
                    call_correlation=env.call_correlation,
                    invocation_id=env.invocation_id,
                    session_id=attempt.session_id,
                    outcome=ResidentBootstrapOutcome.ACKED,
                )
            )
        )
        check = runtime.boundary_settleable

        def cancel_then_check(task_id: str, call_correlation: str) -> bool:
            # The cancel commits before the re-drive's check reads the boundary, and
            # its release reaches the loop only after the check.
            runtime.cancel_workflow(workflow_id)
            return check(task_id, call_correlation)

        svc._boundary_settleable = cancel_then_check
        assert runtime.redispatch_episode_invocation(writer, env.call_correlation)
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()

    assert all(
        claim.state is ClaimState.TERMINAL
        for claim in stores.claims.by_invocation(env.invocation_id)
    )
    assert held_at_reap == [0]
    assert [kind for _, kind, _ in delivery.relays].count("resident_reap") == 1


def test_a_cancel_during_the_relay_bootstrap_reaps_the_attempt_it_records(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    svc, stores, delivery, loop, originated = _wire_resident_service(runtime)
    sessions = svc._delivery.sessions
    update = sessions.update
    cancelled: dict[str, Any] = {}

    async def cancel_during_the_session_write(session_id: str, **fields: Any) -> None:
        if "at" not in cancelled:
            cancelled["at"] = session_id
            runtime.cancel_workflow(cancelled["workflow_id"])
            # The session write yields to the loop, which runs the queued release.
            await asyncio.sleep(0)
        await update(session_id, **fields)

    sessions.update = cancel_during_the_session_write
    try:
        workflow_id, ids = loop.run_until_complete(_register(runtime, _RESIDENT_WF))
        cancelled["workflow_id"] = workflow_id
        _capture_resident_boundary(runtime, ids["writer"], seal_in=tmp_path)
        (env,) = originated
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()

    assert "at" in cancelled
    assert all(
        claim.state is ClaimState.TERMINAL
        for claim in stores.claims.by_invocation(env.invocation_id)
    )
    assert env.invocation_id not in svc._attempts
    kinds = [kind for _, kind, _ in delivery.relays]
    assert "resident_handoff" not in kinds
    assert "resident_sidecar_reap" in kinds


def test_a_cancel_during_a_cold_start_ends_the_origination(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = _runtime()
    svc, stores, delivery, loop, originated = _wire_resident_service(runtime)
    materialize = svc._lifecycle.materialize
    cancelled: dict[str, Any] = {}

    async def cancel_during_the_cold_start(definition: Any) -> Any:
        if "during" not in cancelled:
            cancelled["during"] = True
            runtime.cancel_workflow(cancelled["workflow_id"])
            await asyncio.sleep(0)
        return await materialize(definition)

    svc._lifecycle.materialize = cancel_during_the_cold_start
    try:
        workflow_id, ids = loop.run_until_complete(_register(runtime, _RESIDENT_WF))
        cancelled["workflow_id"] = workflow_id
        _capture_resident_boundary(runtime, ids["writer"], seal_in=tmp_path)
        (env,) = originated
        with caplog.at_level(logging.WARNING):
            loop.run_until_complete(asyncio.sleep(0.3))
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()

    assert cancelled.get("during")
    assert pending == []
    assert all(
        claim.state is ClaimState.TERMINAL
        for claim in stores.claims.by_invocation(env.invocation_id)
    )
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
    assert "resident_handoff" not in [kind for _, kind, _ in delivery.relays]


def test_a_credit_release_the_admission_store_refused_is_finished_once(
    tmp_path: Path,
) -> None:
    runtime = TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("resident-test"),
        credential_vault=InMemoryCredentialVault(),
        durability_retry=manual_durability_retry,
    )
    svc, stores, _, loop, originated = _wire_resident_service(runtime)
    admission = svc._admission
    persist = admission._persist
    told: list[str] = []
    on_release = admission._on_release

    def released(replica_id: str) -> None:
        told.append(replica_id)
        on_release(replica_id)

    def refused() -> None:
        raise ConnectionError("control redis unavailable")

    admission._on_release = released
    try:
        workflow_id, ids = loop.run_until_complete(_register(runtime, _RESIDENT_WF))
        writer = ids["writer"]
        _capture_resident_boundary(runtime, writer, seal_in=tmp_path)
        loop.run_until_complete(asyncio.sleep(0.05))
        (env,) = originated
        (claim,) = stores.claims.by_invocation(env.invocation_id)
        assert claim.holds_credit

        admission._persist = refused
        assert runtime.settle_episode_invocation(writer, env.call_correlation, "done")
        loop.run_until_complete(asyncio.sleep(0.05))
        assert told == []
        assert runtime._durability.pending(workflow_id)
        admission._persist = persist

        assert runtime._durability.run_due() == [workflow_id]
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()

    assert claim.state is ClaimState.TERMINAL
    assert claim.terminal_reason is ClaimTerminalReason.COMPLETED
    assert told == [claim.replica_id]
    assert not stores.credit_ledger.held(claim.replica_id)
