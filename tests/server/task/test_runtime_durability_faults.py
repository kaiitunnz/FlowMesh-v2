"""A runtime transition whose durable writes fail delivers nothing early, and heals.

Every durable write from the k-th on fails, either refused or applied before its error,
for each k a transition makes. Each action the runtime delivers is checked against the
durable state its kind needs committed first, and once writes heal the transition
completes as a clean run does: the claim settles once at the claim FSM, and the durable
state matches memory. A crash at the same cut restores without an early release or an
orphan issue.
"""

import asyncio
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.orchestration.state import InvocationState, LedgerSnapshot
from server.registries.workflow import PersistedTask
from server.resident import ClaimState, ClaimTerminalReason
from server.task.models import TERMINAL_TASK_STATUSES, PublishGate, TaskStatus
from server.task.runtime import TaskRuntime
from server.task.runtime.after_commit import (
    AfterCommit,
    AuthorizeTurn,
    CreditRelease,
    Interrupt,
    Issue,
    Purge,
    Reap,
    Revoke,
    Settled,
)
from tests.server.dispatch_helpers import record_dispatch
from tests.server.runtime_helpers import durable_invocation
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
    _wire_resident_service,
)
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    FakeRegistry,
    _planned,
    _register,
)
from tests.support.waiting import pop_ready


class _FaultyRegistry(FakeRegistry):
    """Fails every durable write from ``fail_from`` on while ``down``; with
    ``applied`` each failing write lands before its error."""

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0
        self.fail_from: int | None = None
        self.applied = False

    def _write(self, apply: Callable[[], None]) -> None:
        self.writes += 1
        if self.fail_from is not None and self.writes >= self.fail_from:
            if self.applied:
                apply()
            raise ConnectionError(f"durable write {self.writes} refused")
        apply()

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        self._write(
            lambda: super(_FaultyRegistry, self).commit_transition(
                workflow_id, **kwargs
            )
        )

    def commit_dynamic_tasks(self, workflow_id: str, *args: Any, **kwargs: Any) -> None:
        self._write(
            lambda: super(_FaultyRegistry, self).commit_dynamic_tasks(
                workflow_id, *args, **kwargs
            )
        )

    def save_ledger_snapshot(
        self, workflow_id: str, snapshot: LedgerSnapshot, control: Any = None
    ) -> None:
        self._write(
            lambda: super(_FaultyRegistry, self).save_ledger_snapshot(
                workflow_id, snapshot
            )
        )

    def heal(self) -> None:
        self.fail_from = None

    def ledger(self, workflow_id: str) -> LedgerSnapshot | None:
        blob = self.ledger_blobs.get(workflow_id)
        return LedgerSnapshot.model_validate_json(blob) if blob else None

    def record(self, task_id: str) -> Any:
        blob = self.task_blobs.get(task_id)
        return PersistedTask.model_validate_json(blob).record if blob else None


@dataclass
class _Observer:
    """Checks each delivered action against the durable state its kind needs."""

    registry: _FaultyRegistry
    workflow_id: str
    delivered: list[AfterCommit] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def _invocation(self, invocation_id: str | None) -> InvocationState | None:
        ledger = self.registry.ledger(self.workflow_id)
        if ledger is None or invocation_id is None:
            return None
        return next(
            (i.state for i in ledger.invocations if i.invocation_id == invocation_id),
            None,
        )

    def _call_settled(self, task_id: str, call: str) -> bool:
        record = self.registry.record(task_id)
        if record is not None and record.status in TERMINAL_TASK_STATUSES:
            return True
        ledger = self.registry.ledger(self.workflow_id)
        invocations = {
            event.invocation_id
            for event in (ledger.boundary_events if ledger else [])
            if event.call_correlation == call and event.invocation_id
        }
        return bool(invocations) and all(
            self._invocation(i) is InvocationState.TERMINAL for i in invocations
        )

    def _settled(self, task_id: str, dispatch_id: str | None) -> bool:
        record = self.registry.record(task_id)
        return record is not None and (
            record.status in (*TERMINAL_TASK_STATUSES, TaskStatus.CANCELLING)
            or record.dispatch_id != dispatch_id
        )

    def check(self, action: AfterCommit) -> None:
        match action:
            case CreditRelease(invocation_id=invocation_id):
                ok = self._invocation(invocation_id) is InvocationState.TERMINAL
            case Issue(invocation_id=invocation_id):
                ok = self._invocation(invocation_id) is not None
            case Reap(task_id=task_id, call_correlation=call):
                ok = self._call_settled(task_id, call)
            case Interrupt(message=message) | Revoke(message=message):
                ok = self._settled(message.task_id, message.dispatch_id)
            case Purge(workflow_id=workflow_id, settled_only=settled_only):
                records = [
                    self.registry.record(task_id)
                    for task_id in self.registry.workflow_task_ids[workflow_id]
                ]
                done = (
                    TERMINAL_TASK_STATUSES
                    if settled_only
                    else (*TERMINAL_TASK_STATUSES, TaskStatus.CANCELLING)
                )
                ok = all(r is not None and r.status in done for r in records)
            case AuthorizeTurn(proposal=proposal):
                record = self.registry.record(proposal.agent_task_id)
                ok = record is not None and record.dispatch_id == proposal.dispatch_id
            case Settled():
                ok = True
        if not ok:
            self.violations.append(repr(action))
        self.delivered.append(action)

    def effects(self) -> Counter[str]:
        """What reached workers and resident control, by kind; a completion notice
        and a credential purge are idempotent, so their repeats do not count."""
        return Counter(
            type(action).__name__
            for action in self.delivered
            if not isinstance(action, (Settled, Purge))
        )


def _observe(runtime: TaskRuntime, observer: _Observer) -> None:
    deliver = runtime._deliver

    def checked(workflow_id: str | None, action: AfterCommit) -> None:
        observer.check(action)
        deliver(workflow_id, action)

    setattr(runtime, "_deliver", checked)


class _Resident:
    """An agent whose resident model boundary resident admission accepted, over the
    real claim FSM, with every replica release recorded."""

    def __init__(self, tmp_path: Path) -> None:
        self.registry = _FaultyRegistry()
        self.runtime = _runtime(self.registry)
        (
            self.svc,
            self.stores,
            self.delivery,
            self.loop,
            self.originated,
        ) = _wire_resident_service(self.runtime)
        self.released: list[str] = []
        admission = self.svc._admission
        on_release = admission._on_release

        def released(replica_id: str) -> None:
            self.released.append(replica_id)
            on_release(replica_id)

        admission._on_release = released
        self.svc._resolve_dependency = self.runtime.resolve_service_dependency
        self.workflow_id, ids = self.loop.run_until_complete(
            _register(self.runtime, _RESIDENT_WF)
        )
        self.writer = ids["writer"]
        self.observer = _Observer(self.registry, self.workflow_id)
        _observe(self.runtime, self.observer)
        self.tmp_path = tmp_path

    def capture(self) -> None:
        _capture_resident_boundary(self.runtime, self.writer, seal_in=self.tmp_path)
        self.settle_loop()

    def settle_loop(self) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.05))

    @property
    def env(self) -> Any:
        return self.originated[0]

    def claim(self) -> Any:
        (claim,) = self.stores.claims.by_invocation(self.env.invocation_id)
        return claim

    def credit_held(self) -> bool:
        return any(
            self.stores.credit_ledger.held(replica.replica_id)
            for replica in self.stores.directory.all()
        )

    def heal(self) -> None:
        self.registry.heal()
        for _ in range(10):
            if not self.runtime._durability.pending(self.workflow_id):
                break
            self.runtime._durability.run_due()
            self.settle_loop()
        assert not self.runtime._durability.pending(self.workflow_id)

    def assert_durable_matches_memory(self) -> None:
        for record in self.runtime.task_records():
            durable = self.registry.record(record.task_id)
            assert durable is not None and durable.status == record.status
        engine = self.runtime.orchestration_engine(self.workflow_id)
        assert engine is not None
        assert self.registry.ledger(self.workflow_id) == engine.to_snapshot()

    def crash_and_restore(self) -> None:
        """Lose the runtime at the current cut and restart it from the durable state,
        with resident control kept, as a root restart does."""
        self.runtime.shutdown()
        self.registry.heal()
        self.runtime = _runtime(self.registry)
        _observe(self.runtime, self.observer)
        runtime = self.runtime
        self.svc._settle = runtime.settle_episode_invocation
        self.svc._redispatch = runtime.redispatch_episode_invocation
        self.svc._boundary_settleable = runtime.boundary_settleable
        self.svc._resolve_dependency = runtime.resolve_service_dependency
        runtime.set_resident_terminal_hook(self.svc.on_invocation_terminal)

        def originate(env: Any) -> bool:
            self.originated.append(env)
            return bool(self.svc.originate(env))

        runtime._resident_originate = originate
        assert self.loop.run_until_complete(runtime.rehydrate()) == 1
        self.svc.reconcile_workflow_terminals(runtime.resident_invocation_completed)
        self.settle_loop()
        self.heal()

    def close(self) -> None:
        self.runtime.shutdown()
        self.loop.close()


@contextmanager
def _resident(tmp_path: Path) -> Iterator[_Resident]:
    resident = _Resident(tmp_path)
    try:
        yield resident
    finally:
        resident.close()


def _settle(resident: _Resident) -> None:
    resident.runtime.settle_episode_invocation(
        resident.writer, resident.env.call_correlation, "a completion"
    )


def _settle_failed(resident: _Resident) -> None:
    resident.runtime.settle_episode_invocation(
        resident.writer, resident.env.call_correlation, None, error="upstream failed"
    )


def _cancel(resident: _Resident) -> None:
    resident.runtime.cancel_workflow(resident.workflow_id)


@dataclass(frozen=True)
class _Transition:
    name: str
    run: Callable[[_Resident], None]
    reason: ClaimTerminalReason


_TRANSITIONS = (
    _Transition("settle", _settle, ClaimTerminalReason.COMPLETED),
    _Transition("settle-failed", _settle_failed, ClaimTerminalReason.FAILED),
    _Transition("cancel", _cancel, ClaimTerminalReason.FAILED),
)


def _run(resident: _Resident, transition: _Transition) -> None:
    transition.run(resident)
    resident.settle_loop()


def _clean(tmp_path: Path, transition: _Transition) -> tuple[int, Counter[str]]:
    """The durable writes a transition makes and the effects it delivers, unfaulted."""
    with _resident(tmp_path / "clean") as resident:
        resident.capture()
        before, delivered = resident.registry.writes, len(resident.observer.delivered)
        _run(resident, transition)
        effects = Counter(
            type(action).__name__
            for action in resident.observer.delivered[delivered:]
            if not isinstance(action, (Settled, Purge))
        )
        return resident.registry.writes - before, effects


def _cuts(writes: int) -> Sequence[int]:
    return range(1, writes + 1)


@pytest.mark.parametrize("applied", [False, True], ids=["refused", "applied"])
@pytest.mark.parametrize("transition", _TRANSITIONS, ids=lambda t: t.name)
def test_a_faulted_transition_delivers_nothing_early_and_heals(
    tmp_path: Path, transition: _Transition, applied: bool
) -> None:
    writes, clean_effects = _clean(tmp_path, transition)
    assert writes > 0
    for cut in _cuts(writes):
        with _resident(tmp_path / f"cut-{cut}") as resident:
            resident.capture()
            claim = resident.claim()
            delivered = len(resident.observer.delivered)
            resident.registry.applied = applied
            resident.registry.fail_from = resident.registry.writes + cut

            _run(resident, transition)

            assert resident.observer.violations == [], f"cut {cut}"
            if not applied:
                assert claim.holds_credit, f"cut {cut}: released before durable"
            resident.heal()

            assert resident.observer.violations == [], f"cut {cut}"
            effects = Counter(
                type(action).__name__
                for action in resident.observer.delivered[delivered:]
                if not isinstance(action, (Settled, Purge))
            )
            assert effects == clean_effects, f"cut {cut}"
            assert claim.state is ClaimState.TERMINAL, f"cut {cut}"
            assert claim.terminal_reason is transition.reason, f"cut {cut}"
            assert resident.released == [claim.replica_id], f"cut {cut}"
            assert not resident.credit_held(), f"cut {cut}"
            resident.assert_durable_matches_memory()


@pytest.mark.parametrize("applied", [False, True], ids=["refused", "applied"])
@pytest.mark.parametrize("transition", _TRANSITIONS, ids=lambda t: t.name)
def test_a_crash_at_a_faulted_cut_restores_without_an_early_or_lost_release(
    tmp_path: Path, transition: _Transition, applied: bool
) -> None:
    writes, _ = _clean(tmp_path, transition)
    for cut in _cuts(writes):
        with _resident(tmp_path / f"cut-{cut}") as resident:
            resident.capture()
            claim = resident.claim()
            resident.registry.applied = applied
            resident.registry.fail_from = resident.registry.writes + cut
            _run(resident, transition)
            released_before = list(resident.released)
            invocation = durable_invocation(
                resident.registry.ledger_blobs,
                resident.workflow_id,
                claim.invocation_id,
            )
            writer = resident.registry.record(resident.writer)
            # A durably settled agent terminalizes its boundary when restored.
            durable = invocation is InvocationState.TERMINAL or (
                writer is not None and writer.status in TERMINAL_TASK_STATUSES
            )

            resident.crash_and_restore()

            assert resident.observer.violations == [], f"cut {cut}"
            if durable:
                assert claim.state is ClaimState.TERMINAL, f"cut {cut}"
                assert claim.terminal_reason is transition.reason, f"cut {cut}"
                assert resident.released == [claim.replica_id], f"cut {cut}"
            else:
                # The transition never became durable, so nothing was released for it;
                # the restored workflow re-drives its boundary, which releases at most
                # once.
                assert released_before == [], f"cut {cut}"
                expected = [] if claim.holds_credit else [claim.replica_id]
                assert resident.released == expected, f"cut {cut}"
            assert [c.invocation_id for c in resident.stores.claims.all()] == [
                claim.invocation_id
            ], f"cut {cut}"
            resident.assert_durable_matches_memory()


def test_the_observer_catches_an_action_delivered_before_its_cause_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _resident(tmp_path) as resident:
        resident.capture()
        monkeypatch.setattr(
            resident.runtime._actions,
            "file_locked",
            lambda _workflow_id, *actions: resident.runtime._actions.queue_locked(
                *actions
            ),
        )
        resident.registry.fail_from = resident.registry.writes + 1

        _run(resident, _TRANSITIONS[0])

        assert resident.observer.violations
        assert not resident.claim().holds_credit


def test_a_failed_interrupt_is_retried_without_repeating_its_credit_release(
    tmp_path: Path,
) -> None:
    with _resident(tmp_path) as resident:
        resident.capture()
        claim = resident.claim()
        stub = resident.runtime._worker_registry
        publish = stub.publish_interrupt
        published: list[Any] = []

        def flaky(*args: Any) -> int:
            published.append(args[1])
            if len(published) == 1:
                raise ConnectionError("dispatch channel unavailable")
            return int(publish(*args))

        setattr(stub, "publish_interrupt", flaky)
        resident.observer.delivered.clear()
        _run(resident, _TRANSITIONS[2])
        assert resident.released == [claim.replica_id]
        assert resident.runtime._durability.pending(resident.workflow_id)

        resident.heal()

        assert [message.task_id for message in published] == [resident.writer] * 2
        assert resident.observer.effects() == Counter(
            {"CreditRelease": 1, "Interrupt": 2}
        )
        assert resident.released == [claim.replica_id]


class _Spawning:
    """A producer whose success fans out two children."""

    def __init__(self) -> None:
        self.registry = _FaultyRegistry()
        self.runtime = _runtime(self.registry)
        self.workflow_id, ids = asyncio.run(_register(self.runtime, AUTORESEARCH))
        self.planner = ids["planner"]
        assert pop_ready(self.runtime) == self.planner
        record_dispatch(self.runtime, self.planner, "wkr-1", "dsp-1")
        self.payload = _planned(self.runtime, self.planner, ["h1", "h2"])

    def succeed(self) -> None:
        self.runtime.mark_succeeded(self.planner, "wkr-1", self.payload, _TS, "dsp-1")

    def assert_consistent(self) -> None:
        """The durable children are the ledger's, each with its record, once."""
        engine = self.runtime.orchestration_engine(self.workflow_id)
        assert engine is not None
        assert self.registry.ledger(self.workflow_id) == engine.to_snapshot()
        children = self.registry.dynamic_task_ids.get(self.workflow_id, set())
        assert len(children) == 2
        for child in children:
            assert self.registry.record(child) is not None
            assert engine.work_item(child) is not None
        assert self.registry.record(self.planner).status == TaskStatus.DONE


def _restarted(runtime: TaskRuntime, registry: FakeRegistry) -> TaskRuntime:
    """A new runtime over the same durable state and shared content store."""
    return _runtime(registry, runtime._results)


def _spawn_writes() -> int:
    spawning = _Spawning()
    before = spawning.registry.writes
    spawning.succeed()
    return spawning.registry.writes - before


@pytest.mark.parametrize("crash", [False, True], ids=["heal", "crash"])
@pytest.mark.parametrize("applied", [False, True], ids=["refused", "applied"])
def test_a_faulted_fan_out_keeps_its_children_whole(applied: bool, crash: bool) -> None:
    writes = _spawn_writes()
    assert writes > 1
    for cut in _cuts(writes):
        spawning = _Spawning()
        spawning.registry.applied = applied
        spawning.registry.fail_from = spawning.registry.writes + cut
        spawning.succeed()
        # Nothing it materialized is published while its records may be lost.
        for task_id in iter(lambda: pop_ready(spawning.runtime), None):
            assert (
                spawning.runtime.begin_publish(
                    task_id, cast(Any, SimpleNamespace(id="wkr-2", node_id="n")), "d"
                )
                is PublishGate.NOT_DURABLE
            ), f"cut {cut}"
        spawning.registry.heal()
        if crash:
            spawning.runtime.shutdown()
            spawning.runtime = _restarted(spawning.runtime, spawning.registry)
            assert asyncio.run(spawning.runtime.rehydrate()) == 1
            # The success was never acknowledged, so its stream redelivers it.
            spawning.succeed()
        for _ in range(5):
            spawning.runtime._durability.run_due()
        assert not spawning.runtime._durability.pending(spawning.workflow_id)
        spawning.assert_consistent()
        spawning.runtime.shutdown()


def test_an_issue_a_cancel_overtook_is_not_delivered(tmp_path: Path) -> None:
    with _resident(tmp_path) as resident:
        resident.capture()
        env = resident.env
        with resident.runtime._lock:
            resident.runtime._actions.queue_locked(
                Issue(resident.writer, env.call_correlation, env.invocation_id)
            )
        resident.runtime.cancel_workflow(resident.workflow_id)
        resident.settle_loop()

        assert resident.originated == [env]


def test_a_success_after_a_cancel_keeps_the_released_credit(tmp_path: Path) -> None:
    with _resident(tmp_path) as resident:
        resident.capture()
        claim = resident.claim()
        _run(resident, _TRANSITIONS[2])
        assert claim.terminal_reason is ClaimTerminalReason.FAILED

        _run(resident, _TRANSITIONS[0])

        assert claim.terminal_reason is ClaimTerminalReason.FAILED
        assert resident.released == [claim.replica_id]
        assert resident.observer.violations == []
