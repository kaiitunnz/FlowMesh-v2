"""The worker-local mediated-egress sidecar lane."""

import queue
from typing import Any

import pytest

from shared.tools.contract import (
    MediatedOperationOutcome,
    MediatedOperationPermit,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)
from shared.utils.ids import new_mediated_permit_id
from tests.shared.outcome_helpers import InMemoryContentStore
from worker.lifecycle import PendingEgressRequestStore
from worker.mediated_egress_sidecar import MediatedEgressSidecar

_WORKER = "wkr-1"
_GEN = 3
_AGENT = "tsk-agent"
_CALL = "m0"
_REQUEST = ToolRequest(interface=SEARCH_INTERFACE, query="weather", max_results=3)


class _StubEgress:
    """A search egress backend with a fixed outcome and an egress-call counter."""

    interface = SEARCH_INTERFACE

    def __init__(self, outcome: ToolOutcome) -> None:
        self._outcome = outcome
        self.calls = 0

    def digest(self, request: Any) -> str:
        return tool_request_digest(
            request.interface, request.query, request.max_results
        )

    def execute(self, envelope: Any, request: Any) -> ToolOutcome:
        self.calls += 1
        return self._outcome


def _permit(**overrides: Any) -> MediatedOperationPermit:
    fields: dict[str, Any] = {
        "permit_id": new_mediated_permit_id(),
        "agent_task_id": _AGENT,
        "call_correlation": _CALL,
        "interface": SEARCH_INTERFACE,
        "subject": SEARCH_INTERFACE,
        "invocation_id": "inv-1",
        "idempotency_key": "idm-1",
        "request_digest": tool_request_digest(SEARCH_INTERFACE, "weather", 3),
        "target_id": _WORKER,
        "target_generation": _GEN,
        "deadline_epoch": 2_000_000_000.0,
        "max_results": 5,
        "timeout_sec": 10.0,
        "result_char_cap": 4000,
    }
    fields.update(overrides)
    return MediatedOperationPermit(**fields)


class _Harness:
    def __init__(
        self,
        outcome: ToolOutcome | None = None,
        store: InMemoryContentStore | None = None,
    ) -> None:
        self.pending = PendingEgressRequestStore()
        self.reports: queue.Queue[MediatedOperationOutcome] = queue.Queue()
        self.egress = _StubEgress(
            outcome or ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="x")
        )
        self.sidecar = MediatedEgressSidecar(
            pending_requests=self.pending,
            audience=lambda: (_WORKER, _GEN),
            egresses=(self.egress,),
            outcome_sink=self.reports.put,
            content_store=store,
        )

    def stash(self) -> None:
        self.pending.put(_AGENT, _CALL, _REQUEST)

    def report(self, timeout: float = 5.0) -> MediatedOperationOutcome:
        return self.reports.get(timeout=timeout)

    def stop(self) -> None:
        self.sidecar.stop()


def test_no_worker_private_request_is_terminal() -> None:
    h = _Harness()
    h.sidecar.submit_permit(_permit())
    report = h.report()
    assert report.error is not None and "no worker-private request" in report.error
    h.stop()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"request_digest": "deadbeef"}, "digest"),
        ({"target_id": "wkr-other"}, "audience"),
        ({"target_generation": 99}, "generation"),
        ({"deadline_epoch": 1.0}, "expired"),
        ({"interface": "other/v1"}, "interface"),
    ],
)
def test_fence_rejection_is_terminal_not_retried(
    overrides: dict[str, Any], reason: str
) -> None:
    h = _Harness(ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="x"))
    h.stash()
    h.sidecar.submit_permit(_permit(**overrides))
    report = h.report()
    assert report.error is not None and reason in report.error
    # A rejected operation never egresses; custody is retained for the reap.
    assert h.egress.calls == 0
    assert h.pending.peek(_AGENT, _CALL) is not None
    h.stop()


def test_non_success_outcome_is_inline() -> None:
    out = ToolOutcome(status=ToolOutcomeStatus.QUOTA, value="budget exhausted")
    h = _Harness(out)
    h.stash()
    h.sidecar.submit_permit(_permit())
    report = h.report()
    assert report.outcome == out and report.outcome_ref is None
    h.stop()


def test_successful_result_is_reported_by_reference() -> None:
    store = InMemoryContentStore()
    h = _Harness(
        ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="results..."), store
    )
    h.stash()
    h.sidecar.submit_permit(_permit())
    report = h.report()
    assert report.outcome is None
    assert report.outcome_ref is not None
    assert report.outcome_ref.idempotency_key == "idm-1"
    h.stop()


def test_custody_is_retained_until_reap() -> None:
    store = InMemoryContentStore()
    h = _Harness(ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="r"), store)
    h.stash()
    h.sidecar.submit_permit(_permit())
    h.report()
    # Non-destructive: the request survives the egress until the committed-outcome reap.
    assert h.pending.peek(_AGENT, _CALL) is not None
    h.sidecar.reap(_AGENT, _CALL)
    assert h.pending.peek(_AGENT, _CALL) is None
    h.stop()


def test_permit_replay_is_ignored() -> None:
    out = ToolOutcome(status=ToolOutcomeStatus.QUOTA, value="q")
    h = _Harness(out)
    h.stash()
    permit = _permit()
    h.sidecar.submit_permit(permit)
    h.report()
    h.sidecar.submit_permit(permit)
    with pytest.raises(queue.Empty):
        h.report(timeout=0.3)
    h.stop()


def test_redrive_after_materialize_recovers_the_prior_outcome() -> None:
    store = InMemoryContentStore()
    h = _Harness(
        ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="results..."), store
    )
    h.stash()
    h.sidecar.submit_permit(_permit())
    first = h.report()
    assert first.outcome_ref is not None
    # Custody retained; a fresh permit (same idempotency key) recovers by reference
    # without egressing again.
    egressed = h.egress.calls
    h.sidecar.submit_permit(_permit())
    second = h.report()
    assert second.outcome_ref is not None
    assert second.outcome_ref.content_digest == first.outcome_ref.content_digest
    assert h.egress.calls == egressed
    h.stop()
