"""A root restart releases a resident credit its ledger already settled.

A crash between the ledger terminal and the credit release leaves the claim holding
its credit with nothing left to settle it; startup replays the ledger terminal.
"""

import asyncio
from typing import Any

import pytest

from server.resident.state import ClaimState, ClaimTerminalReason, ResidentSnapshot
from server.startup import rehydrate_root_state
from server.task.runtime import TaskRuntime, TransitionNotDurable
from tests.server.resident.test_service import _admission, _build, _env
from tests.server.task.test_agent_episode_runtime import _held_boundary
from tests.server.task.test_held_termination_release import _Scenario
from tests.server.task.test_v2_orchestration import FakeRegistry, _runtime


class _SnapshotRegistry:
    def __init__(self, snapshot: ResidentSnapshot) -> None:
        self._snapshot = snapshot

    async def load_snapshot_async(self) -> ResidentSnapshot:
        return self._snapshot


def _admit(workflow_id: str, invocation_id: str) -> ResidentSnapshot:
    """A resident control holding one credit for a workflow's boundary invocation."""
    svc, stores, _settled, _delivery = _build()
    binding = _admission().model_copy(update={"workflow_id": workflow_id})
    svc._resolve_dependency = lambda _task_id: binding
    asyncio.run(svc._originate(_env(invocation_id)))
    (claim,) = stores.claims.by_invocation(invocation_id)
    assert claim.holds_credit
    return stores.to_snapshot()


def _restart(registry: FakeRegistry, snapshot: ResidentSnapshot) -> Any:
    runtime = _runtime(registry)
    svc, stores, _settled, _delivery = _build()
    asyncio.run(
        rehydrate_root_state(runtime, svc, _SnapshotRegistry(snapshot))  # type: ignore[arg-type]
    )
    svc.shutdown()
    return stores


def _held(registry: FakeRegistry) -> tuple[TaskRuntime, str, Any]:
    runtime = _runtime(registry)
    workflow_id, _writer, _engine, env = asyncio.run(_held_boundary(runtime))
    return runtime, workflow_id, env


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("settled", ClaimTerminalReason.COMPLETED),
        ("failed", ClaimTerminalReason.FAILED),
        ("cancelled", ClaimTerminalReason.FAILED),
    ],
)
def test_a_restart_releases_a_credit_the_ledger_settled(
    path: str, reason: ClaimTerminalReason
) -> None:
    registry = FakeRegistry()
    runtime, workflow_id, env = _held(registry)
    snapshot = _admit(workflow_id, env.invocation_id)
    # The ledger terminal commits; the root crashes before the credit release runs.
    if path == "cancelled":
        runtime.cancel_workflow(workflow_id)
    else:
        value = "draft" if path == "settled" else None
        error = "upstream refused" if path == "failed" else None
        assert runtime.settle_episode_invocation(
            env.task_id, env.call_correlation, value, error=error
        )

    stores = _restart(registry, snapshot)
    (claim,) = stores.claims.by_invocation(env.invocation_id)
    assert claim.state is ClaimState.TERMINAL and claim.terminal_reason is reason
    assert stores.credit_ledger.held(claim.replica_id) == 0


def test_a_restart_holds_a_credit_whose_invocation_is_still_open() -> None:
    registry = FakeRegistry()
    _, workflow_id, env = _held(registry)
    snapshot = _admit(workflow_id, env.invocation_id)

    stores = _restart(registry, snapshot)
    (claim,) = stores.claims.by_invocation(env.invocation_id)
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held(claim.replica_id) == 1


def _crash_before_ledger_save(registry: FakeRegistry) -> None:
    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("root crashed")

    registry.save_ledger = crash  # type: ignore[method-assign]


def test_a_restart_releases_a_credit_whose_cancel_only_its_records_hold() -> None:
    registry = FakeRegistry()
    runtime, workflow_id, env = _held(registry)
    snapshot = _admit(workflow_id, env.invocation_id)
    _crash_before_ledger_save(registry)
    with pytest.raises(TransitionNotDurable), runtime.acknowledging():
        runtime.cancel_workflow(workflow_id)
    runtime.shutdown()
    del registry.save_ledger

    stores = _restart(registry, snapshot)
    (claim,) = stores.claims.by_invocation(env.invocation_id)
    assert claim.state is ClaimState.TERMINAL
    assert claim.terminal_reason is ClaimTerminalReason.FAILED
    assert stores.credit_ledger.held(claim.replica_id) == 0


def test_a_restart_releases_a_credit_whose_failure_only_its_records_hold() -> None:
    scenario = _Scenario()
    snapshot = _admit(scenario.workflow_id, scenario.invocation_id)
    _crash_before_ledger_save(scenario.registry)
    assert scenario.report_success() is not None
    del scenario.registry.save_ledger

    stores = _restart(scenario.registry, snapshot)
    (claim,) = stores.claims.by_invocation(scenario.invocation_id)
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
