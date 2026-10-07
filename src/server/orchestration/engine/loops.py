"""Loop iterations of one workflow instance."""

from shared.utils import (
    new_activation_id,
)

from ..guardrails import ScopeBudget
from ..state import (
    Activation,
    CapabilityStatus,
    ProgressAxis,
    PublicationOutcome,
    Record,
    ValueRef,
)
from .advance import Advance, RegionError
from .dataflow import RegionFlow
from .ledger import OrchestrationLedger
from .publications import PublicationLedger
from .scopes import ScopeProgress


class LoopProgress:
    """Re-materializes a loop body per iteration, seals a loop and releases its
    carried value once its loop-time capability closes."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        publication: PublicationLedger,
        scope_progress: ScopeProgress,
        flow: RegionFlow,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._publication = publication
        self._scope_progress = scope_progress
        self._flow = flow
        self._budget = budget

    def loop_feedback(self, loop: str, *, value_ref: ValueRef | None = None) -> str:
        """Re-materialize a loop body at the next loop-time coordinate.

        Enforces well-founded logical time: loop_time strictly increases and stays under
        the iteration budget, so a finite prefix is acyclic after time unrolling.
        Returns the iteration activation id.
        """
        scope_id = self._scope_progress.require_loop_scope(loop)
        loop_op = self._ledger.scopes[
            scope_id
        ].owner_operator_id or self._ledger.handle_operator(loop)
        cap = self._scope_progress.require_capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is not CapabilityStatus.OPEN:
            raise RegionError(
                f"loop {loop_op!r} is {cap.status.value}; no feedback may arrive"
            )
        next_time = self._ledger.loop_time.get(scope_id, 0) + 1
        if next_time > self._budget.max_loop_iterations:
            self._scope_progress.exhaust_budget(
                "loop_iterations", self._budget.max_loop_iterations
            )
        self._scope_progress.charge_activation()
        self._ledger.loop_time[scope_id] = next_time
        cap.coordinate = next_time
        cap.outstanding += 1
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=scope_id,
            operator_id=loop_op,
            kind="iteration",
            loop_time=next_time,
        )
        self._ledger.add_activation(activation)
        self._ledger.records.append(
            Record(
                operator_id=loop_op,
                activation_id=activation.activation_id,
                scope_id=scope_id,
                loop_time=next_time,
                value_ref=value_ref,
            )
        )
        self._ledger.emit(
            "loop_feedback", operator_id=loop_op, detail={"loop_time": str(next_time)}
        )
        return activation.activation_id

    def settle_iteration(self, iteration_activation_id: str) -> Advance:
        """Mark a loop iteration terminal and drain the loop-time capability."""
        activation = self._ledger.activations.get(iteration_activation_id)
        if activation is None or activation.kind != "iteration":
            raise RegionError(f"unknown loop iteration {iteration_activation_id!r}")
        cap = self._scope_progress.require_capability(
            activation.scope_id, ProgressAxis.LOOP_TIME
        )
        cap.outstanding = max(0, cap.outstanding - 1)
        return self._maybe_egress_loop(activation.scope_id)

    def loop_seal(self, loop: str) -> Advance:
        """Seal a loop: no further feedback; egress once pending iterations drain."""
        scope_id = self._scope_progress.require_loop_scope(loop)
        cap = self._scope_progress.require_capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "loop_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._maybe_egress_loop(scope_id)

    def _maybe_egress_loop(self, scope_id: str) -> Advance:
        if not scope_id or scope_id in self._ledger.released_scopes:
            return Advance()
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.LOOP_TIME))
        if cap is None or not cap.closed:
            return Advance()
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        self._scope_progress.frontier_closed(scope_id)
        loop_op = self._ledger.scopes[scope_id].owner_operator_id or ""
        self._ledger.emit(
            "loop_egress", operator_id=loop_op, detail={"scope": scope_id}
        )
        carried = self._ledger.latest_carried(scope_id)
        self._publication.publish(loop_op, PublicationOutcome.SUCCESS, carried)
        return self._flow.deliver_record(
            loop_op, self._ledger.control_activation(loop_op), carried
        )
