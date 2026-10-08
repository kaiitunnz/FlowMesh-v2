"""The runtime lock contract catches each kind of breach it checks for."""

import asyncio
from typing import Any

from server.orchestration import OrchestrationEngine
from server.task.runtime import TaskRuntime
from server.task.runtime.after_commit import Settled
from tests.server import lock_contract
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_v2_orchestration import (
    AUTORESEARCH,
    FakeRegistry,
    _bundle,
    _register,
)


def _registered() -> tuple[TaskRuntime, str, dict[str, str]]:
    runtime = _runtime(FakeRegistry())
    workflow_id, ids = asyncio.run(_register(runtime, AUTORESEARCH))
    return runtime, workflow_id, ids


def _breaches(trips: list[lock_contract.Trip]) -> set[tuple[str, str, bool]]:
    return {(trip.kind, trip.name, trip.from_source) for trip in trips}


def test_a_lock_bound_helper_source_calls_without_the_lock_is_caught() -> None:
    runtime, _, ids = _registered()

    with lock_contract.recorded() as trips:
        runtime._committer.persist_locked(ids["planner"])

    assert ("unlocked", "TransitionCommitter.persist_locked", False) in _breaches(trips)
    assert ("unlocked", "TransitionCommitter._write_locked", True) in _breaches(trips)


def test_a_reentrant_acquisition_passes() -> None:
    runtime, _, ids = _registered()

    with lock_contract.recorded() as trips:
        with runtime._lock, runtime._cv:
            runtime._committer.persist_locked(ids["planner"])

    assert trips == []


def test_a_registered_engine_read_without_the_lock_is_caught() -> None:
    runtime, workflow_id, _ = _registered()
    engine = runtime._engines[workflow_id]

    with lock_contract.recorded() as trips:
        engine.blocked_input_agents()

    assert _breaches(trips) == {
        ("unlocked", "OrchestrationEngine.blocked_input_agents", False)
    }


def test_a_standalone_engine_needs_no_lock() -> None:
    engine = OrchestrationEngine.build("wfl-x", "owner", "org", _bundle(AUTORESEARCH))

    with lock_contract.recorded() as trips:
        engine.blocked_input_agents()

    assert trips == []


def test_a_delivery_under_the_lock_is_caught() -> None:
    runtime, workflow_id, _ = _registered()
    delivered: list[Any] = []
    runtime._committer.notify_terminal_transition = delivered.append  # type: ignore[method-assign,assignment]

    with lock_contract.recorded() as trips:
        with runtime._lock:
            runtime._actions.queue_locked(Settled(workflow_id))
            runtime._act_after_commit()

    assert delivered == [workflow_id]
    assert ("locked", "TaskRuntime._deliver", True) in _breaches(trips)


def test_a_delivery_after_the_transition_runs_off_the_lock() -> None:
    runtime, workflow_id, _ = _registered()

    with lock_contract.recorded() as trips:
        with runtime._transition(raises=False):
            runtime._actions.file_locked(workflow_id, Settled(workflow_id))

    assert trips == []
