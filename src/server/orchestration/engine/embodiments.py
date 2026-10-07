"""Embodiment selections and input resolutions of one workflow instance."""

from shared.content import ContentReference
from shared.inference import InputResolutionBinding

from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    EmbodimentSelection,
    InputPreparation,
    InputResolution,
)
from .ledger import OrchestrationLedger
from .topology import PlanTopology


class EmbodimentLedger:
    """Holds each work item's embodiment selection, input resolution and input
    preparation."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self.embodiment_selections: dict[str, EmbodimentSelection] = {}
        self.input_resolutions: dict[str, InputResolution] = {}
        self.input_preparations: dict[str, InputPreparation] = {}

    def embodiment_selection(self, task_id: str) -> EmbodimentSelection | None:
        """The embodiment a task is already bound to, if one was resolved."""
        wi = self._ledger.work_item_for_task(task_id)
        return self.embodiment_selections.get(wi.work_item_id) if wi else None

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a resolved embodiment is committed to the run that carries it.

        An embodiment changes only before its candidate-specific issue or delivery. A
        resident candidate commits at its invocation, after which reconciliation reuses
        that invocation and its idempotency and credit path rather than running the
        other embodiment; a local candidate carries no invocation and commits when its
        attempt is issued, which is where it was delivered to a worker.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.work_item_id not in self.embodiment_selections:
            return False
        return wi.invocation_id is not None or bool(wi.attempt_ids)

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> EmbodimentSelection | None:
        """Bind a task to one embodiment durably, before its worker message goes out.

        A pinned selection is kept: the caller receives the standing one rather than a
        replacement.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        if (standing := self.embodiment_selections.get(wi.work_item_id)) is not None:
            if self.embodiment_pinned(task_id):
                return standing
        selection = EmbodimentSelection(
            work_item_id=wi.work_item_id,
            alternative_id=alternative_id,
            plan_version=self._topology.bundle.plan.plan_version.content_digest,
            selector=selector,
            evidence=evidence,
        )
        self.embodiment_selections[wi.work_item_id] = selection
        self._ledger.emit(
            "embodiment_selected",
            work_item_id=wi.work_item_id,
            detail={"alternative_id": alternative_id, "selector": selector},
        )
        return selection

    def input_resolution(self, task_id: str) -> InputResolution | None:
        """The resolution a task's inputs were materialized under, if one exists."""
        wi = self._ledger.work_item_for_task(task_id)
        return self.input_resolutions.get(wi.work_item_id) if wi else None

    def input_preparation(self, task_id: str) -> InputPreparation | None:
        """The preparation dispatch a task's inputs are being resolved by, if any."""
        wi = self._ledger.work_item_for_task(task_id)
        return self.input_preparations.get(wi.work_item_id) if wi else None

    def on_input_preparation_dispatched(
        self, task_id: str, worker_id: str | None
    ) -> None:
        """Record that a work item's inputs are being resolved on a worker.

        This deliberately mints neither an invocation nor an attempt: both are
        candidate-specific commitments, and a work item whose inputs are still being
        resolved has not chosen an embodiment to commit to.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        self.input_preparations[wi.work_item_id] = InputPreparation(
            work_item_id=wi.work_item_id, worker_id=worker_id
        )
        self._ledger.emit(
            "input_preparation_dispatched",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
        )

    def record_input_resolution(
        self,
        task_id: str,
        binding: InputResolutionBinding,
        reference: ContentReference | None = None,
    ) -> InputResolution | None:
        """Record how a work item's inputs resolved, before its embodiment runs.

        A standing resolution is kept: a re-drive that reaches the same request records
        nothing new, and one that reaches a different request leaves the recorded
        binding in place for the reconciliation that compares against it.

        A resolution carrying its request's reference commits both together, so the
        request a later run hydrates is durable exactly when the binding proving what it
        is becomes durable.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return None
        if (standing := self.input_resolutions.get(wi.work_item_id)) is not None:
            return standing
        resolution = InputResolution(
            work_item_id=wi.work_item_id, binding=binding, reference=reference
        )
        self.input_resolutions[wi.work_item_id] = resolution
        self._ledger.emit(
            "input_resolved",
            work_item_id=wi.work_item_id,
            detail={
                "request_digest": binding.request_digest,
                "cardinality": str(binding.cardinality),
            },
        )
        return resolution
