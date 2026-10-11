"""A transition that leaves a workflow's ledger and control as stored writes nothing."""

import asyncio
import itertools

import pytest

from server.clients.redis import workflow_key, workflow_revision_key
from server.registries.workflow import WorkflowRegistry
from server.task.runtime import TaskRuntime
from tests.server.redis_helpers import raw_control
from tests.server.task.test_settled_history import (
    _register,
    _registry,
    _runtime,
    _succeed_ready,
)
from tests.server.task.test_v2_orchestration import LINEAR


@pytest.fixture(autouse=True)
def _ticking_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count()
    monkeypatch.setattr(
        "server.registries.workflow.now_iso",
        lambda: f"2026-06-01T00:00:{next(ticks):09.3f}Z",
    )


def _stamp(registry: WorkflowRegistry, workflow_id: str) -> tuple[str, str]:
    raw = raw_control(registry)
    return (
        raw.hget(workflow_key(workflow_id), "updated_at"),
        raw.get(workflow_revision_key(workflow_id)),
    )


def _restarted(registry: WorkflowRegistry, workflow_id: str) -> TaskRuntime:
    runtime = _runtime(registry)
    assert asyncio.run(runtime.restore([workflow_id])) == [workflow_id]
    return runtime


def test_a_cancel_of_a_settled_workflow_writes_nothing() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, LINEAR)
    runtime.cancel_workflow(workflow_id)
    assert runtime.workflow_settlement(workflow_id).settled
    stamp = _stamp(registry, workflow_id)

    assert runtime.cancel_workflow(workflow_id) == []

    assert _stamp(registry, workflow_id) == stamp


def test_a_restart_writes_nothing_of_an_open_workflow() -> None:
    registry = _registry()
    workflow_id = _register(_runtime(registry), LINEAR)
    stamp = _stamp(registry, workflow_id)

    runtime = _restarted(registry, workflow_id)

    assert _stamp(registry, workflow_id) == stamp
    assert _succeed_ready(runtime)
    assert runtime.workflow_settlement(workflow_id).settled


def test_a_restart_writes_nothing_of_a_settled_workflow() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, LINEAR)
    assert _succeed_ready(runtime)
    stamp = _stamp(registry, workflow_id)

    _restarted(registry, workflow_id)

    assert _stamp(registry, workflow_id) == stamp


def _bumped(before: tuple[str, str], after: tuple[str, str]) -> bool:
    return after[0] > before[0] and int(after[1]) > int(before[1])


def test_an_advance_after_a_restart_still_writes_the_ledger() -> None:
    registry = _registry()
    workflow_id = _register(_runtime(registry), LINEAR)
    runtime = _restarted(registry, workflow_id)
    stamp = _stamp(registry, workflow_id)

    assert _succeed_ready(runtime)

    assert _bumped(stamp, _stamp(registry, workflow_id))


def test_a_cancel_that_moves_tasks_still_writes_the_ledger() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, LINEAR)
    stamp = _stamp(registry, workflow_id)

    assert runtime.cancel_workflow(workflow_id)

    assert _bumped(stamp, _stamp(registry, workflow_id))
