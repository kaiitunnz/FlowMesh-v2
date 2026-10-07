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
    """Holds and records each work item's embodiment selection, input resolution and
    input preparation."""

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
        """Whether a resolved embodiment is committed to the run that carries it."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.work_item_id not in self.embodiment_selections:
            return False
        return wi.invocation_id is not None or bool(wi.attempt_ids)

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> EmbodimentSelection | None:
        """Bind a task to one embodiment durably, before its worker message goes out."""
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
        """Record that a work item's inputs are being resolved on a worker."""
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
        """Record how a work item's inputs resolved, before its embodiment runs."""
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
