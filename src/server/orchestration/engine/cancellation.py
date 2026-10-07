"""Whole-subtree cancellation and failure of one workflow instance."""

from ...task.v2.representations.operators import OperatorKind
from ..state import TERMINAL_WORK_ITEM_STATUSES, PublicationOutcome
from .advance import Advance, RegionError
from .attempts import AttemptLifecycle
from .dataflow import RegionFlow
from .ledger import OrchestrationLedger
from .publications import PublicationLedger
from .scopes import ScopeProgress
from .topology import PlanTopology


class ScopeCancellation:
    """Cancels or fails a scope subtree as a recorded transition."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        publication: PublicationLedger,
        scope_progress: ScopeProgress,
        flow: RegionFlow,
        attempt_lifecycle: AttemptLifecycle,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._publication = publication
        self._scope_progress = scope_progress
        self._flow = flow
        self._attempt_lifecycle = attempt_lifecycle

    def fail_scope_tree(self, scope_id: str, reason: str) -> Advance:
        self._ledger.emit("instance_failed", detail={"reason": reason})
        for sid in self._ledger.scope_subtree(scope_id):
            self._scope_progress.revoke_progress(sid)
        advance = Advance()
        for wi in list(self._ledger.work_items.values()):
            if wi.status in TERMINAL_WORK_ITEM_STATUSES or self._topology.kind(
                wi.operator_id
            ) not in (OperatorKind.LEAF, OperatorKind.AGENT):
                continue
            self._attempt_lifecycle.fail_open_attempt(wi, reason)
            advance.extend(self._flow.settle_failed_wi(wi))
        for slot in list(self._publication.slots.values()):
            self._publication.write_publication(
                slot, PublicationOutcome.DECLARED_FAILURE, None
            )
        return advance

    def cancel_scope(self, scope_id: str) -> Advance:
        """Cancel a scope subtree as a durable, recorded-before-terminal event."""
        if scope_id not in self._ledger.scopes:
            raise RegionError(f"{scope_id!r} is no cancellable scope")
        advance = Advance()
        for sid in self._ledger.scope_subtree(scope_id):
            advance.extend(self._flow.cancel_one_scope(sid))
        return advance
