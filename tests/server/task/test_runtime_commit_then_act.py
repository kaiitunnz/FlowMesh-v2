"""A runtime transition's side effects run only after its durable commit."""

import asyncio
import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import InvocationState, LedgerSnapshot
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime, TransitionNotDurable
from server.task.workflow_retry import WorkflowRetryScheduler
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.result_store import make_result_reader
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.server.task.test_worker_originated_boundary import _WorkerStub


def _runtime(registry: FakeRegistry) -> TaskRuntime:
    """A runtime whose durability retry runs only when a test drives it."""
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("commit-then-act"),
        credential_vault=InMemoryCredentialVault(),
        durability_retry=lambda fire, logger: WorkflowRetryScheduler(
            fire, logger, base_delay_sec=0.0, run_thread=False
        ),
    )


def _durable_invocation(
    registry: FakeRegistry, workflow_id: str, invocation_id: str
) -> InvocationState | None:
    if (blob := registry.ledger_blobs.get(workflow_id)) is None:
        return None
    snapshot = LedgerSnapshot.model_validate_json(blob)
    return next(
        (i.state for i in snapshot.invocations if i.invocation_id == invocation_id),
        None,
    )


class _Resident:
    """A resident agent boundary issued to resident admission, with every credit
    release recorded beside the durable state of its invocation at that moment."""

    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.runtime = _runtime(self.registry)
        self.issued: list[tuple[Any, InvocationState | None]] = []
        self.runtime._resident_originate = self._originate
        self.releases: list[tuple[str, bool, InvocationState | None]] = []
        self.runtime.set_resident_terminal_hook(self._release)
        self.workflow_id, ids = asyncio.run(_register(self.runtime, _RESIDENT_WF))
        self.writer = ids["writer"]
        self._save = self.registry.save_ledger_snapshot

    def _originate(self, env: Any) -> bool:
        self.issued.append((env, self._durable(env.invocation_id)))
        return True

    def _release(self, invocation_id: str, failed: bool) -> None:
        self.releases.append((invocation_id, failed, self._durable(invocation_id)))

    def _durable(self, invocation_id: str) -> InvocationState | None:
        return _durable_invocation(self.registry, self.workflow_id, invocation_id)

    def capture(self) -> Any:
        _capture_resident_boundary(self.runtime, self.writer)
        ((env, _),) = self.issued
        return env

    def ledger_down(self) -> None:
        def down(*_: Any, **__: Any) -> None:
            raise ConnectionError("control redis unavailable")

        self.registry.save_ledger_snapshot = down  # type: ignore[method-assign]

    def ledger_up(self) -> None:
        self.registry.save_ledger_snapshot = self._save  # type: ignore[method-assign]


@pytest.mark.parametrize("error", [None, "upstream failed"])
def test_a_failed_save_outside_a_report_holds_the_credit_for_its_retry(
    error: str | None,
) -> None:
    resident = _Resident()
    env = resident.capture()
    value = None if error is not None else "a completion"

    resident.ledger_down()
    assert resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert resident.releases == []
    resident.ledger_up()

    # The resident controller settles again; the boundary already settled in memory,
    # so the retry is absorbed, and it makes the first settle durable.
    assert not resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert not resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert resident.releases == [
        (env.invocation_id, error is not None, InvocationState.TERMINAL)
    ]


def test_a_settle_no_caller_retries_is_made_durable_by_the_retry_alone() -> None:
    resident = _Resident()
    env = resident.capture()

    resident.ledger_down()
    resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, "a completion"
    )
    assert resident.runtime._durability.run_due() == [resident.workflow_id]
    assert resident.releases == []
    resident.ledger_up()

    assert resident.runtime._durability.run_due() == [resident.workflow_id]

    assert resident.releases == [(env.invocation_id, False, InvocationState.TERMINAL)]
    assert not resident.runtime._durability.pending(resident.workflow_id)


def test_a_settle_inside_a_held_report_releases_nothing_before_its_save() -> None:
    resident = _Resident()
    env = resident.capture()

    resident.ledger_down()
    with resident.runtime._transition(raises=False):
        resident.runtime._settle_episode_invocation(
            resident.writer, env.call_correlation, "a completion"
        )

    assert resident.releases == []
    assert resident._durable(env.invocation_id) is InvocationState.ISSUED


def test_a_boundary_issues_to_its_handler_only_once_its_routing_is_durable() -> None:
    resident = _Resident()

    resident.ledger_down()
    with pytest.raises(TransitionNotDurable):
        _capture_resident_boundary(resident.runtime, resident.writer)

    # Nothing reached resident admission for an invocation the ledger may lose.
    assert resident.issued == []
    resident.ledger_up()

    assert resident.runtime._durability.run_due() == [resident.workflow_id]

    ((env, durable),) = resident.issued
    assert durable is InvocationState.ISSUED
    assert resident._durable(env.invocation_id) is InvocationState.ISSUED


def test_a_crash_after_a_held_routing_leaves_no_claim_behind() -> None:
    resident = _Resident()
    resident.ledger_down()
    with pytest.raises(TransitionNotDurable):
        _capture_resident_boundary(resident.runtime, resident.writer)
    resident.runtime.shutdown()
    resident.ledger_up()

    restored = _runtime(resident.registry)
    restored._resident_originate = resident._originate
    assert asyncio.run(restored.rehydrate()) == 1

    # The routing never became durable, so its invocation never reached resident
    # admission: no claim exists for an invocation the restored ledger lacks, and the
    # agent runs its step again.
    assert resident.issued == []
    record = restored.get_record(resident.writer)
    assert record is not None and record.status == TaskStatus.PENDING
