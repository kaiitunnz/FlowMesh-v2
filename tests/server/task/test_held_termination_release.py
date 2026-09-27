"""A failed workflow releases what its work holds only once its terminal ledger is
durable.

A worker report whose durable writes fail completes in memory and is redelivered; a
workflow it failed along the way keeps its resident credits until the replay makes the
held ledger durable, so a crash in between never leaves a credit released against a
ledger that still shows the invocation open.
"""

import asyncio
from typing import Any

import pytest

from server.orchestration.state import InvocationState, LedgerSnapshot
from server.task.models import EventEffect, SettleOutcome
from server.task.results import ResultUnreadable
from server.task.runtime import TaskRuntime, _HeldWrites, _Termination, _Unacknowledged
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import result_payload
from tests.server.task.test_agent_episode_runtime import _MODEL_HELD_SCRIPT, _step
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _runtime,
    _worker,
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
        self.runtime.set_resident_terminal_hook(
            lambda inv, _failed: self.releases.append((inv, self.durable_state(inv)))
        )
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

    def durable_state(self, invocation_id: str) -> InvocationState:
        snapshot = LedgerSnapshot.model_validate_json(
            self.registry.ledger_blobs[self.workflow_id]
        )
        return next(
            i.state for i in snapshot.invocations if i.invocation_id == invocation_id
        )

    def fail_writes_after(self, allowed: int) -> None:
        def commit(*args: Any, **kwargs: Any) -> None:
            self.writes += 1
            if self.writes > allowed:
                raise RuntimeError("redis down")
            self._commit(*args, **kwargs)

        self.registry.commit_transition = commit  # type: ignore[method-assign]

    def heal_writes(self) -> None:
        self.registry.commit_transition = self._commit  # type: ignore[method-assign]

    def report_success(self) -> Exception | None:
        try:
            self.runtime.mark_succeeded(
                self.ids["planner"], "wkr-1", self.payload, _TS, "dsp-p"
            )
        except RuntimeError as exc:
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
    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_an_unheld_failure_releases_once_after_the_ledger_is_durable() -> None:
    scenario = _Scenario()
    assert scenario.report_success() is None
    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_a_replaced_stash_keeps_the_release_it_carries() -> None:
    scenario = _Scenario()
    planner = scenario.ids["planner"]
    scenario.fail_writes_after(1)
    assert scenario.report_success() is not None
    # A failure report of the same dispatch replaces the stash while writes still fail.
    with pytest.raises(RuntimeError):
        scenario.runtime.fail_dispatch(planner, "wkr-1", {}, _TS, "dsp-p", error="late")
    scenario.heal_writes()

    scenario.runtime.fail_dispatch(planner, "wkr-1", {}, _TS, "dsp-p", error="late")

    assert scenario.releases == [(scenario.invocation_id, InvocationState.TERMINAL)]


def test_a_recommit_under_a_held_report_keeps_its_release_held() -> None:
    scenario = _Scenario()
    runtime = scenario.runtime
    termination = _Termination([], [], [], resident_invocation_ids=["inv-x"])
    current = runtime._report_writes.held = _HeldWrites(error=RuntimeError("down"))
    try:
        with runtime._cv:
            runtime._recommit_locked(_HeldWrites(terminations=[termination]))
    finally:
        runtime._report_writes.held = None

    assert runtime._pending_terminations == []
    assert current.terminations == [termination]


def test_a_replayed_cancel_report_releases_what_its_stash_held() -> None:
    scenario = _Scenario()
    runtime = scenario.runtime
    released: list[str] = []
    runtime.set_resident_terminal_hook(lambda inv, _failed: released.append(inv))
    planner = scenario.ids["planner"]
    termination = _Termination([], [], [], resident_invocation_ids=["inv-x"])
    runtime._unacknowledged[planner] = _Unacknowledged(
        "TASK_CANCELLED",
        "wkr-1",
        "dsp-p",
        _HeldWrites(terminations=[termination]),
        SettleOutcome(EventEffect.SETTLED, "cancelled", [], []),
    )

    runtime.mark_cancelled(planner, "wkr-1", {}, _TS, "dsp-p")

    assert released == ["inv-x"]
    assert runtime._pending_terminations == []
