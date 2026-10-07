"""Physical attempts and invocations of one workflow instance."""

from enum import Enum, auto

from shared.utils import (
    new_attempt_id,
    new_invocation_id,
)

from ...utils.time import now_iso
from ..outcomes import (
    is_compensable,
    is_replayable,
    next_on_acknowledge,
    next_on_reissue,
    next_on_terminal,
)
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    Attempt,
    AttemptStatus,
    EffectReceipt,
    Invocation,
    InvocationState,
    PublicationOutcome,
    WorkItem,
    WorkItemStatus,
)
from .boundaries import BoundaryLedger
from .embodiments import EmbodimentLedger
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger

_OPEN_ATTEMPT_STATUSES = frozenset({AttemptStatus.ISSUED, AttemptStatus.RUNNING})


class _LossResolution(Enum):
    """How the loss of a work item's worker resolves it."""

    NOTHING = auto()
    PREPARE_AGAIN = auto()
    BOUNDARY_FAILS = auto()
    RUN_AGAIN = auto()
    FAILS = auto()


_RERUN_ON_LOSS = frozenset({_LossResolution.PREPARE_AGAIN, _LossResolution.RUN_AGAIN})


class AttemptLifecycle:
    """Records the physical attempts and invocations of dispatchable work items,
    their effect receipts, and how a lost worker resolves them."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        embodiments: EmbodimentLedger,
        inputs: AcceptedInputLedger,
        boundaries: BoundaryLedger,
    ) -> None:
        self._ledger = ledger
        self._embodiments = embodiments
        self._inputs = inputs
        self._boundaries = boundaries
        self.receipts: dict[str, EffectReceipt] = {}

    def on_dispatched(self, task_id: str, worker_id: str | None) -> None:
        """Record a physical attempt and issue or reissue the work item's invocation."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        if wi.invocation_id is None:
            invocation = Invocation(
                invocation_id=new_invocation_id(),
                work_item_id=wi.work_item_id,
                state=InvocationState.ISSUED,
                replayable=is_replayable(wi.effect_class, wi.replay_contract),
                compensable=is_compensable(wi.effect_class, wi.replay_contract),
            )
            wi.invocation_id = invocation.invocation_id
            self._ledger.invocations[invocation.invocation_id] = invocation
        else:
            invocation = self._ledger.invocations[wi.invocation_id]
            invocation.state = next_on_reissue(invocation.state)
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
            attempt_no=len(wi.attempt_ids) + 1,
            worker_id=worker_id,
            started_at=now_iso(),
            alternative_id=(
                selection.alternative_id
                if (
                    selection := self._embodiments.embodiment_selections.get(
                        wi.work_item_id
                    )
                )
                else None
            ),
        )
        wi.attempt_ids.append(attempt.attempt_id)
        self._ledger.attempts[attempt.attempt_id] = attempt
        wi.status = WorkItemStatus.DISPATCHED
        self._ledger.emit(
            "attempt_issued",
            work_item_id=wi.work_item_id,
            attempt_id=attempt.attempt_id,
            invocation_id=wi.invocation_id or "",
            operator_id=wi.operator_id,
        )

    def on_started(self, task_id: str) -> None:
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.invocation_id is None:
            return
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.RUNNING
        invocation = self._ledger.invocations[wi.invocation_id]
        invocation.state = next_on_acknowledge(invocation.state)
        self._ledger.emit(
            "invocation_acknowledged",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

    def settle_attempt_terminal(
        self, wi: WorkItem, outcome: PublicationOutcome
    ) -> None:
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        if wi.invocation_id is not None:
            invocation = self._ledger.invocations[wi.invocation_id]
            invocation.state = next_on_terminal(invocation.state)
            self._ledger.emitter.emit_boundary(invocation)
            self._record_receipt(wi, outcome)

    def fail_open_attempt(self, wi: WorkItem, error: str) -> None:
        """Close the work item's attempt as failed while it is still in flight; one
        already closed keeps its outcome."""
        attempt = self._ledger.latest_attempt(wi)
        if attempt is None or attempt.status not in _OPEN_ATTEMPT_STATUSES:
            return
        attempt.status = AttemptStatus.FAILED
        attempt.finished_at = now_iso()
        attempt.error = error
        self._ledger.emitter.emit_attempt(attempt)

    def on_returned(self, task_id: str) -> bool:
        """Close an in-flight attempt handed back without an outcome, and re-ready it;
        returns whether there was one.

        The attempt is not charged: the work item runs again under its invocation.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.DISPATCHED:
            return False
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.RETURNED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "attempt_returned", work_item_id=wi.work_item_id, operator_id=wi.operator_id
        )
        return True

    def retries_on_loss(self, task_id: str) -> bool:
        """Whether the loss of the task's worker runs its work item again, as
        ``on_uncertain`` resolves it, rather than failing it."""
        return (
            self.resolve_loss(self._ledger.work_item_for_task(task_id))
            in _RERUN_ON_LOSS
        )

    def resolve_loss(self, wi: WorkItem | None) -> _LossResolution:
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return _LossResolution.NOTHING
        if wi.invocation_id is None:
            if wi.work_item_id in self._embodiments.input_preparations:
                return _LossResolution.PREPARE_AGAIN
            return _LossResolution.NOTHING
        if (
            wi.status is WorkItemStatus.BLOCKED
            and self._boundaries.has_pending_local_boundary(wi)
        ):
            return _LossResolution.BOUNDARY_FAILS
        if self._ledger.invocations[wi.invocation_id].replayable:
            return _LossResolution.RUN_AGAIN
        return _LossResolution.FAILS

    def suspending_worker(self, task_id: str) -> str | None:
        """The worker whose step suspended the task on a mediated boundary that awaits
        its outcome, or None when the task is not suspended on one."""
        wi = self._ledger.work_item_for_task(task_id)
        if (
            wi is None
            or wi.status is not WorkItemStatus.BLOCKED
            or not self._boundaries.awaits_mediated_outcome(wi)
            or (attempt := self._ledger.latest_attempt(wi)) is None
        ):
            return None
        return attempt.worker_id

    def close_latest_attempt(self, task_id: str) -> None:
        """Settle a still-running attempt of a continuing episode, bounding its history.

        A continue-boundary re-dispatches without suspending, so its finished attempt is
        marked succeeded here rather than left perpetually running.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is not None and (attempt := self._ledger.latest_attempt(wi)) is not None:
            if attempt.status in (AttemptStatus.ISSUED, AttemptStatus.RUNNING):
                attempt.status = AttemptStatus.SUCCEEDED
                attempt.finished_at = now_iso()
                self._ledger.emitter.emit_attempt(attempt)

    def _record_receipt(self, wi: WorkItem, outcome: PublicationOutcome) -> None:
        if wi.invocation_id is None or wi.invocation_id in self.receipts:
            return
        self.receipts[wi.invocation_id] = EffectReceipt(
            invocation_id=wi.invocation_id,
            work_item_id=wi.work_item_id,
            outcome=outcome,
        )
        self._ledger.emit(
            "effect_receipt",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

    def reconcile_pending(self, task_id: str) -> bool:
        """Re-derive readiness for a task whose durable record shows PENDING.

        Returns whether the work item is ready to admit. A work item the snapshot still
        shows in flight — a crash after a retry persisted the PENDING record but before
        the ledger caught up — is reset to ready with its lost attempt marked, so the
        retry is not orphaned; a work item whose predecessors have not all settled, or
        whose declared inputs have not all been accepted, stays blocked.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return False
        cont = self._ledger.continuations.get(wi.work_item_id)
        if cont is not None and (
            cont.waiting_on
            or not cont.required_ports
            <= {
                a.target_port
                for a in self._inputs.accepted_inputs_for(wi.activation_id)
            }
        ):
            wi.status = WorkItemStatus.BLOCKED
            return False
        if (
            wi.status is WorkItemStatus.BLOCKED
            and self._boundaries.awaits_mediated_outcome(wi)
        ):
            # A crash beat the ledger save of the boundary's settle; the boundary is
            # re-issued, and the episode resumes only with its outcome.
            return False
        if wi.status is WorkItemStatus.DISPATCHED:
            if attempt := self._ledger.latest_attempt(wi):
                attempt.status = AttemptStatus.LOST
                attempt.finished_at = now_iso()
                self._ledger.emitter.emit_attempt(attempt)
            self._ledger.emit(
                "attempt_lost_on_restart",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
        wi.status = WorkItemStatus.READY
        return True

    def latest_attempt_open(self, task_id: str) -> bool:
        """Whether the work item's latest attempt still expects a terminal report.

        A reroute that re-enqueues an episode closes the attempt that produced the turn,
        so a completion whose attempt is already closed is a superseded replay —
        applying it would preempt the live turn. A genuine terminal report lands while
        its attempt is still issued or running.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return False
        attempt = self._ledger.latest_attempt(wi)
        return attempt is not None and attempt.status in (
            AttemptStatus.ISSUED,
            AttemptStatus.RUNNING,
        )
