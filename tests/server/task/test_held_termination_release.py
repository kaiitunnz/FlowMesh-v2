"""A failed workflow releases what its work holds only once its terminal ledger is
durable.

A worker report whose durable writes fail completes in memory and is redelivered; a
workflow it failed along the way keeps its resident credits until a later write makes
the held ledger durable, so a crash in between never leaves a credit released against
a ledger that still shows the invocation open.
"""

import asyncio
import logging
import threading
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import InvocationState
from server.task.results import ResultUnreadable
from server.task.runtime import TaskRuntime, TransitionNotDurable
from server.task.runtime.after_commit import AfterCommit, CreditRelease
from shared.tools.contract import MediatedOperationOutcome
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader, result_payload
from tests.server.runtime_helpers import manual_durability_retry
from tests.server.task.test_agent_episode_runtime import _MODEL_HELD_SCRIPT, _step
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _worker,
    _WorkerRegistryStub,
)
from worker.executors.harness.scripted import ScriptedHarnessAdapter

_TS = "2026-09-28T00:00:00Z"

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: held-termination}
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
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [planner]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled}
"""


def _runtime(registry: FakeRegistry) -> TaskRuntime:
    """A runtime whose durability retry runs only when a test drives it."""
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("held-termination"),
        credential_vault=InMemoryCredentialVault(),
        durability_retry=manual_durability_retry,
    )


class _Scenario:
    """An agent holds a model boundary while its workflow's spawn producer succeeds
    with a collection that cannot be read, which fails the workflow."""

    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.runtime: TaskRuntime = _runtime(self.registry)
        captured: list[Any] = []
        self.runtime.set_model_settler(captured.append)
        # Each release records the invocation's durable state at the moment it fires.
        self.releases: list[tuple[str, Any]] = []
        self.runtime.set_resident_terminal_hook(self._release)
        self.workflow_id, self.ids = asyncio.run(_register(self.runtime, _WF))
        _step(
            self.runtime,
            ScriptedHarnessAdapter(_MODEL_HELD_SCRIPT, "v1"),
            self.ids["writer"],
        )
        (env,) = captured
        self.invocation_id: str = env.invocation_id
        planner = self.ids["planner"]
        record_dispatch(self.runtime, planner, _worker(), "dsp-p")
        self.payload = result_payload(
            self.runtime._results, planner, {"ok": True, "items": ["a"]}, "org"
        )

        def unreadable(_binding: Any) -> Any:
            raise ResultUnreadable("corrupt")

        self.runtime._results.read = unreadable  # type: ignore[method-assign,assignment]
        self._commit = self.registry.commit_transition
        self.writes = 0

    def _release(self, invocation_id: str, _failed: bool) -> None:
        self.releases.append((invocation_id, self.durable_state(invocation_id)))

    def durable_state(self, invocation_id: str) -> InvocationState:
        snapshot = self.registry.ledger(self.workflow_id)
        assert snapshot is not None
        return next(
            i.state for i in snapshot.invocations if i.invocation_id == invocation_id
        )

    def fail_writes_after(self, allowed: int) -> None:
        def commit(*args: Any, **kwargs: Any) -> None:
            self.writes += 1
            if self.writes > allowed:
                raise ConnectionError("redis down")
            self._commit(*args, **kwargs)

        self.registry.commit_transition = commit  # type: ignore[method-assign]

    def heal_writes(self) -> None:
        self.registry.commit_transition = self._commit  # type: ignore[method-assign]

    def report_success(self) -> Exception | None:
        try:
            with self.runtime.acknowledging():
                self.runtime.mark_succeeded(
                    self.ids["planner"], "wkr-1", self.payload, _TS, "dsp-p"
                )
        except TransitionNotDurable as exc:
            return exc
        return None


def test_a_held_terminal_ledger_releases_nothing() -> None:
    scenario = _Scenario()
    # The producer's success commits; the workflow failure's commit does not.
    scenario.fail_writes_after(1)
    assert scenario.report_success() is not None

    assert scenario.releases == []
    assert scenario.durable_state(scenario.invocation_id) is InvocationState.ISSUED


def test_the_replayed_report_releases_once_after_the_ledger_is_durable() -> None:
    scenario = _Scenario()
    scenario.fail_writes_after(1)
    assert scenario.report_success() is not None
    scenario.heal_writes()

    assert scenario.report_success() is None
    assert scenario.report_success() is None
    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_the_durability_retry_releases_once_with_no_replay() -> None:
    scenario = _Scenario()
    scenario.fail_writes_after(1)
    assert scenario.report_success() is not None
    assert scenario.runtime._durability.run_due() == [scenario.workflow_id]
    assert scenario.releases == []
    assert scenario.runtime._durability.pending(scenario.workflow_id)
    scenario.heal_writes()

    assert scenario.runtime._durability.run_due() == [scenario.workflow_id]

    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]
    assert not scenario.runtime._durability.pending(scenario.workflow_id)


def test_an_unheld_failure_releases_once_after_the_ledger_is_durable() -> None:
    scenario = _Scenario()
    assert scenario.report_success() is None
    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_a_later_report_of_the_same_dispatch_keeps_the_release_held() -> None:
    scenario = _Scenario()
    planner = scenario.ids["planner"]
    scenario.fail_writes_after(1)
    assert scenario.report_success() is not None
    with pytest.raises(TransitionNotDurable), scenario.runtime.acknowledging():
        scenario.runtime.fail_dispatch(planner, "wkr-1", {}, _TS, "dsp-p", error="late")
    assert scenario.releases == []
    scenario.heal_writes()

    scenario.runtime.fail_dispatch(planner, "wkr-1", {}, _TS, "dsp-p", error="late")

    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_a_failed_credit_release_is_retried_without_repeating_its_siblings() -> None:
    scenario = _Scenario()
    runtime = scenario.runtime
    attempts: list[str] = []

    def flaky(invocation_id: str, failed: bool) -> None:
        attempts.append(invocation_id)
        if len(attempts) == 1:
            raise ConnectionError("resident control unavailable")
        scenario._release(invocation_id, failed)

    runtime.set_resident_terminal_hook(flaky)
    assert scenario.report_success() is None
    assert scenario.releases == []
    assert runtime._durability.pending(scenario.workflow_id)

    assert runtime._durability.run_due() == [scenario.workflow_id]

    assert attempts == [scenario.invocation_id] * 2
    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]
    assert not runtime._durability.pending(scenario.workflow_id)


def _lock_probe(runtime: TaskRuntime) -> list[bool]:
    """Owe one action; records whether its delivery held the lock."""
    owned: list[bool] = []

    def deliver(_workflow_id: str | None, _action: AfterCommit) -> None:
        # A reentrant acquire succeeds on the owning thread, so probe from another.
        def probe() -> None:
            free = runtime._lock.acquire(blocking=False)
            if free:
                runtime._lock.release()
            owned.append(not free)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()

    runtime._deliver = deliver  # type: ignore[method-assign,assignment]
    runtime._actions.queue_locked(CreditRelease("inv-x", failed=True))
    return owned


def test_a_mediated_outcome_delivers_off_the_lock() -> None:
    runtime = _runtime(FakeRegistry())
    owned = _lock_probe(runtime)

    runtime.settle_mediated_operation(
        MediatedOperationOutcome(
            permit_id="mop-x",
            agent_task_id="tsk-x",
            call_correlation="c0",
            invocation_id="inv-x",
            idempotency_key=None,
            error="boom",
        )
    )

    assert owned == [False]


def test_a_redispatched_boundary_delivers_off_the_lock() -> None:
    runtime = _runtime(FakeRegistry())
    owned = _lock_probe(runtime)

    runtime.redispatch_episode_invocation("tsk-x", "c0")

    assert owned == [False]


def test_a_stopped_resident_control_fails_its_boundary_off_the_lock() -> None:
    scenario = _Scenario()
    runtime = scenario.runtime
    settled: list[str | None] = []

    def settle(*_args: Any, error: str | None = None, **_kwargs: Any) -> bool:
        settled.append(error)
        return True

    runtime._resident_originate = lambda _env: False
    runtime._settle_episode_invocation = settle  # type: ignore[method-assign]
    owned = _lock_probe(runtime)
    with runtime._cv:
        env = runtime._engines[scenario.workflow_id].pending_tool_dispatches()[0]

    runtime._originate_resident(env)

    assert settled == ["resident-capacity control is not running"]
    assert owned == [False, False]
