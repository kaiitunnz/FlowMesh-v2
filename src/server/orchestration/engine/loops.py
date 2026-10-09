"""Loop instances of one workflow instance: ingress, routed feedback and exit, and
release once the loop's frontier closes."""

from ...task.v2.representations.operators import LoopContextRegion
from ...task.v2.representations.template import BoundaryKind
from ..guardrails import ScopeBudget
from ..state import (
    CapabilityStatus,
    ControlStatus,
    IterationKind,
    IterationResolution,
    LoopInstance,
    LoopInstanceStatus,
    Occurrence,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    TimeFrame,
    ValueRef,
)
from .advance import Advance, RegionError
from .dataflow import RegionFlow
from .edges import Incoming
from .ledger import OrchestrationLedger
from .occurrences import OccurrenceFactory
from .publications import PublicationLedger
from .scopes import ScopeProgress
from .topology import PlanTopology

_RETURN_ITERATION = {
    BoundaryKind.FEEDBACK: IterationKind.FEEDBACK,
    BoundaryKind.EGRESS: IterationKind.EXIT,
}


class LoopProgress:
    """Runs a loop's body at each logical time its feedback enables.

    Ingress binds the carried seed and the invariants once and enters the body at time
    0. Each time resolves exactly once: a feedback bundle enables the next time at
    once, and an exit seals the loop, whose exiting value leaves only after every
    earlier time's work has drained.
    """

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        publication: PublicationLedger,
        scope_progress: ScopeProgress,
        flow: RegionFlow,
        factory: OccurrenceFactory,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._publication = publication
        self._scope_progress = scope_progress
        self._flow = flow
        self._factory = factory
        self._budget = budget

    @property
    def max_iterations(self) -> int:
        return self._budget.max_loop_iterations

    def ingress(self, key: str, inputs: list[Incoming], advance: Advance) -> None:
        """Enter a loop whose inputs resolved: bind its seed and invariants, open its
        progress scope, and run its body at time 0."""
        if key in self._ledger.loop_by_occurrence:
            return
        occurrence = self._ledger.occurrence(key)
        loop = self._topology.operators[occurrence.operator_id]
        assert isinstance(loop, LoopContextRegion)
        bound = {i.port: i.value for i in inputs if i.port and i.value is not None}
        opener = occurrence.activation_id or self._ledger.control_activation(
            occurrence.operator_id
        )
        scope_id = self._scope_progress.open_loop(opener, occurrence.scope_id)
        instance = LoopInstance(
            scope_id=scope_id,
            occurrence=key,
            context_id=occurrence.context_id,
            parent_time=occurrence.time,
            carried={p.name: bound.get(p.name, _EMPTY) for p in loop.carried},
            invariants={p.name: bound.get(p.name, _EMPTY) for p in loop.invariants},
        )
        self._ledger.loop_instances[scope_id] = instance
        self._ledger.active_loops.add(scope_id)
        self._ledger.loop_by_occurrence[key] = scope_id
        self._ledger.scope_occurrence[scope_id] = key
        self._enter_time(instance, 0, advance)

    def _enter_time(self, instance: LoopInstance, time: int, advance: Advance) -> None:
        """Materialize the body at one logical time and admit what its inputs allow."""
        loop_op = self._ledger.occurrence(instance.occurrence).operator_id
        if time >= self.max_iterations:
            self.fail(
                instance,
                f"LoopIterationBudgetExceeded: loop {loop_op} may run "
                f"{self.max_iterations} iterations",
                advance,
            )
            return
        loop = self._topology.operators[loop_op]
        assert isinstance(loop, LoopContextRegion) and loop.body_ref
        frame = TimeFrame(loop=instance.scope_id, iteration=time)
        try:
            keys = self._factory.enter(
                loop.body_ref,
                instance.context_id,
                (*instance.parent_time, frame),
                instance.scope_id,
            )
        except RegionError as exc:
            self.fail(instance, f"ScopeBudgetExceeded: {exc}", advance)
            return
        instance.times = time + 1
        if (cap := self._capability(instance)) is not None:
            cap.coordinate = time
        self._ledger.emit(
            "loop_time_entered",
            operator_id=loop_op,
            detail={"scope": instance.scope_id, "time": str(time)},
        )
        for key in keys:
            self._flow.evaluate(key, advance)

    def on_return(self, key: str, advance: Advance) -> None:
        """Accept the feedback or exit a body occurrence routed at its time."""
        occurrence = self._ledger.occurrence(key)
        frame = occurrence.time[-1]
        instance = self._ledger.loop_instances.get(frame.loop)
        if instance is None:
            return
        for (kind, port), bundle in self._flow.edges.return_bundles(
            key, BoundaryKind.FEEDBACK, BoundaryKind.EGRESS
        ).items():
            self._accept(
                instance, frame.iteration, _RETURN_ITERATION[kind], bundle, advance
            )

    def _accept(
        self,
        instance: LoopInstance,
        time: int,
        kind: IterationKind,
        bundle: dict[str, ValueRef],
        advance: Advance,
    ) -> None:
        """Record the one resolution of a loop time, idempotently.

        A feedback enables the next time at once; an exit seals the loop. A second,
        different resolution of the same time is a loop-control violation.
        """
        existing = self._ledger.iterations.get((instance.scope_id, time))
        if existing is not None:
            if existing.kind is not kind:
                self.fail(
                    instance,
                    f"LoopControlViolation: time {time} resolved both "
                    f"{existing.kind.value} and {kind.value}",
                    advance,
                )
            return
        if instance.status is not LoopInstanceStatus.OPEN:
            return
        self._ledger.iterations[(instance.scope_id, time)] = IterationResolution(
            loop=instance.scope_id, iteration=time, kind=kind, bundle=bundle
        )
        loop_op = self._ledger.occurrence(instance.occurrence).operator_id
        self._ledger.emit(
            "loop_feedback" if kind is IterationKind.FEEDBACK else "loop_exit",
            operator_id=loop_op,
            detail={"scope": instance.scope_id, "time": str(time)},
        )
        if kind is IterationKind.FEEDBACK:
            self._enter_time(instance, time + 1, advance)
            return
        instance.status = LoopInstanceStatus.EXITED
        instance.exit_time = time
        instance.exit_bundle = bundle
        if (cap := self._capability(instance)) is not None:
            cap.status = CapabilityStatus.SEALED
        self.maybe_release(instance, advance)

    def maybe_release(self, instance: LoopInstance, advance: Advance) -> None:
        """Release an exited loop's value once nothing at or before its exit time can
        still arrive: every body occurrence, nested scope and child it opened has
        drained."""
        if instance.status is not LoopInstanceStatus.EXITED or not (
            self._scope_progress.scope_drained(instance.scope_id)
        ):
            return
        instance.status = LoopInstanceStatus.RELEASED
        self._ledger.released_scopes.add(instance.scope_id)
        self._scope_progress.frontier_closed(instance.scope_id)
        key = instance.occurrence
        occurrence = self._ledger.occurrence(key)
        state = self._ledger.control_state(key)
        state.status = ControlStatus.LIVE
        state.outputs = dict(instance.exit_bundle)
        self._ledger.emit(
            "loop_egress",
            operator_id=occurrence.operator_id,
            detail={"scope": instance.scope_id, "time": str(instance.exit_time)},
        )
        self._publish(occurrence, PublicationOutcome.SUCCESS, instance.exit_bundle)
        self._flow.propagate(
            key, advance, value=next(iter(instance.exit_bundle.values()), None)
        )

    def check_drained_time(self, instance: LoopInstance, advance: Advance) -> None:
        """Fail an open loop whose latest time drained without routing feedback or an
        exit."""
        if instance.status is not LoopInstanceStatus.OPEN or not instance.times:
            return
        time = instance.times - 1
        if (instance.scope_id, time) in self._ledger.iterations:
            return
        if self._scope_progress.scope_drained(instance.scope_id):
            self.fail(
                instance,
                f"LoopControlViolation: time {time} routed neither feedback nor exit",
                advance,
            )

    def fail(self, instance: LoopInstance, reason: str, advance: Advance) -> None:
        """Fail a loop: withdraw its later times, cancel its outstanding body work,
        and fail the loop occurrence itself."""
        if instance.status in (LoopInstanceStatus.FAILED, LoopInstanceStatus.CANCELLED):
            return
        instance.status = LoopInstanceStatus.FAILED
        if instance.times and (instance.scope_id, instance.times - 1) not in (
            self._ledger.iterations
        ):
            self._ledger.iterations[(instance.scope_id, instance.times - 1)] = (
                IterationResolution(
                    loop=instance.scope_id,
                    iteration=instance.times - 1,
                    kind=IterationKind.FAILED,
                    reason=reason,
                )
            )
        for scope_id in self._ledger.scope_subtree(instance.scope_id):
            advance.extend(self._flow.cancel_one_scope(scope_id))
        self._ledger.released_scopes.add(instance.scope_id)
        self._flow.fail_control(instance.occurrence, reason, advance)

    def cancelled(self, scope_id: str) -> None:
        """Withdraw a loop whose scope a cancellation reached."""
        instance = self._ledger.loop_instances.get(scope_id)
        if instance is None or instance.status in (
            LoopInstanceStatus.RELEASED,
            LoopInstanceStatus.FAILED,
            LoopInstanceStatus.CANCELLED,
        ):
            return
        instance.status = LoopInstanceStatus.CANCELLED
        if (
            instance.times
            and (scope_id, instance.times - 1) not in self._ledger.iterations
        ):
            self._ledger.iterations[(scope_id, instance.times - 1)] = (
                IterationResolution(
                    loop=scope_id,
                    iteration=instance.times - 1,
                    kind=IterationKind.CANCELLED,
                )
            )
        state = self._ledger.control_state(instance.occurrence)
        if state.status is ControlStatus.PENDING:
            state.status = ControlStatus.CANCELLED
            self._publish(
                self._ledger.occurrence(instance.occurrence),
                PublicationOutcome.EXPLICIT_EMPTY,
                {},
            )

    def _publish(
        self,
        occurrence: Occurrence,
        outcome: PublicationOutcome,
        bundle: dict[str, ValueRef],
    ) -> None:
        """Publish a root loop's declared output from the carried port it names."""
        if occurrence.context_id or occurrence.time:
            return
        for decl in self._topology.bundle.template.result_declarations:
            if decl.source_ref != occurrence.operator_id:
                continue
            value = bundle.get(decl.source_port or "", _EMPTY)
            self._publication.publish(occurrence.operator_id, outcome, value)

    def _capability(self, instance: LoopInstance) -> ProgressCapability | None:
        return self._ledger.capabilities.get(
            (instance.scope_id, ProgressAxis.LOOP_TIME)
        )


_EMPTY = ValueRef(kind="empty")
