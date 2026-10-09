"""How each edge into an operator occurrence resolved.

An edge's resolution is a function of durable facts: its source occurrence's terminal
outcome and, for a branch, the port its decision selected. A selected route is live,
an unselected one dead, and a failed or cancelled source resolves its edges the same.
"""

from dataclasses import dataclass
from enum import StrEnum

from ...task.v2.representations.operators import BranchRegion, OperatorKind
from ...task.v2.representations.template import (
    BoundaryKind,
    DefinitionKind,
    DependencyUse,
    TemplateEdge,
)
from ..state import (
    ControlStatus,
    Occurrence,
    PublicationOutcome,
    ValueRef,
    WorkItem,
    WorkItemStatus,
)
from .ledger import OrchestrationLedger, occurrence_key
from .topology import PlanTopology


class EdgeState(StrEnum):
    """The resolution of one edge instance."""

    PENDING = "pending"
    """Its source occurrence has not resolved."""
    LIVE = "live"
    """Its source delivered a value."""
    EMPTY = "empty"
    """Its source completed with a declared empty value; still a live route."""
    DEAD = "dead"
    """No record can travel this route for this occurrence."""
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Incoming:
    """One resolved input of an occurrence: the edge, how it resolved, and its value."""

    edge: TemplateEdge
    state: EdgeState
    value: ValueRef | None = None

    @property
    def use(self) -> DependencyUse:
        return self.edge.use

    @property
    def port(self) -> str | None:
        return self.edge.to_port


def project(value: ValueRef | None, steps: tuple[str | int, ...]) -> ValueRef | None:
    """A value narrowed by a further projection."""
    if value is None or not steps:
        return value
    return value.model_copy(update={"projection": (*value.projection, *steps)})


def wi_value(wi: WorkItem) -> ValueRef:
    """The value a settled work item delivers."""
    if wi.value_ref is not None:
        return wi.value_ref
    return ValueRef(kind="legacy_task_result", legacy_task_id=wi.legacy_task_id)


class EdgeResolver:
    """Resolves the edges into an occurrence from the durable state of their sources."""

    def __init__(self, ledger: OrchestrationLedger, topology: PlanTopology) -> None:
        self._ledger = ledger
        self._topology = topology

    def sibling(self, occurrence: Occurrence, operator_id: str) -> str:
        """The key of another operator's occurrence in the same context and time."""
        return occurrence_key(operator_id, occurrence.context_id, occurrence.time)

    def incoming(self, key: str) -> list[Incoming]:
        """Every input of an occurrence: its edges and its definition entry bindings."""
        occurrence = self._ledger.occurrence(key)
        resolved = [
            Incoming(
                edge,
                *self.source_state(
                    self.sibling(occurrence, edge.from_op),
                    edge.from_port,
                    edge.projection,
                ),
            )
            for edge in self._topology.incoming.get(occurrence.operator_id, ())
        ]
        for entry in self._topology.entries_into.get(occurrence.operator_id, ()):
            value = self.entry_value(occurrence, entry.from_port or "")
            resolved.append(
                Incoming(
                    entry,
                    EdgeState.LIVE if value is not None else EdgeState.PENDING,
                    project(value, entry.projection),
                )
            )
        return resolved

    def return_bundles(
        self, key: str, *kinds: BoundaryKind
    ) -> dict[tuple[BoundaryKind, str | None], dict[str, ValueRef]]:
        """The bundles a settled occurrence carries out of its definition on its live
        routes, by boundary kind and the source port each leaves through."""
        operator_id = self._ledger.occurrence(key).operator_id
        bundles: dict[tuple[BoundaryKind, str | None], dict[str, ValueRef]] = {}
        for edge in self._topology.returns_from.get(operator_id, ()):
            if edge.boundary not in kinds or edge.boundary is None:
                continue
            state, value = self.source_state(key, edge.from_port, edge.projection)
            if state not in (EdgeState.LIVE, EdgeState.EMPTY):
                continue
            bundle = bundles.setdefault((edge.boundary, edge.from_port), {})
            bundle[edge.to_port or ""] = value or ValueRef(kind="empty")
        return bundles

    def source_state(
        self,
        source: str,
        port: str | None = None,
        steps: tuple[str | int, ...] = (),
    ) -> tuple[EdgeState, ValueRef | None]:
        """How a route out of ``source`` (through ``port``) resolved, with the value
        it carries narrowed by ``steps``."""
        operator_id = self._ledger.occurrence(source).operator_id
        kind = self._topology.kind(operator_id)
        if kind in (OperatorKind.LEAF, OperatorKind.AGENT):
            wi_id = self._ledger.wi_by_occurrence.get(source)
            wi = self._ledger.work_items.get(wi_id) if wi_id else None
            if wi is None:
                return EdgeState.PENDING, None
            return self._wi_state(wi, steps)
        state = self._ledger.control_states.get(source)
        if state is None:
            return EdgeState.PENDING, None
        match state.status:
            case ControlStatus.PENDING:
                return EdgeState.PENDING, None
            case ControlStatus.DEAD:
                return EdgeState.DEAD, None
            case ControlStatus.FAILED:
                return EdgeState.FAILED, None
            case ControlStatus.CANCELLED:
                return EdgeState.CANCELLED, None
        op = self._topology.operators.get(operator_id)
        if isinstance(op, BranchRegion):
            decision = self._ledger.branch_decisions.get(source)
            if decision is None or decision.port != port:
                return EdgeState.DEAD, None
        if port is None and op is not None and len(op.outputs) == 1:
            port = op.outputs[0].name
        value = state.outputs.get(port or "")
        if value is None:
            # A live control that stored nothing under the port it was read through
            # carries no value to substitute.
            return EdgeState.FAILED, None
        if value.kind == "empty":
            return EdgeState.EMPTY, value
        return EdgeState.LIVE, project(value, steps)

    @staticmethod
    def _wi_state(
        wi: WorkItem, steps: tuple[str | int, ...]
    ) -> tuple[EdgeState, ValueRef | None]:
        match wi.status:
            case WorkItemStatus.SKIPPED:
                return EdgeState.DEAD, None
            case WorkItemStatus.CANCELLED:
                return EdgeState.CANCELLED, None
            case WorkItemStatus.SETTLED:
                match wi.outcome:
                    case PublicationOutcome.SUCCESS:
                        return EdgeState.LIVE, project(wi_value(wi), steps)
                    case PublicationOutcome.EXPLICIT_EMPTY:
                        return EdgeState.EMPTY, ValueRef(kind="empty")
                    case _:
                        return EdgeState.FAILED, None
        return EdgeState.PENDING, None

    def entry_value(self, occurrence: Occurrence, port: str) -> ValueRef | None:
        """The value a definition input carries into an occurrence's context and time.

        A loop body reads its carried bundle for the occurrence's time and its loop's
        invariants; a child reads the values it was entered with.
        """
        definition_id = self._topology.definition_of.get(occurrence.operator_id)
        definition = self._topology.definitions.get(definition_id or "")
        if definition is None:
            return None
        if definition.kind is DefinitionKind.CHILD:
            context = self._ledger.child_contexts.get(occurrence.context_id)
            return context.entries.get(port) if context else None
        if not occurrence.time:
            return None
        frame = occurrence.time[-1]
        instance = self._ledger.loop_instances.get(frame.loop)
        if instance is None:
            return None
        if port in instance.invariants:
            return instance.invariants[port]
        if frame.iteration == 0:
            return instance.carried.get(port)
        previous = self._ledger.iterations.get((frame.loop, frame.iteration - 1))
        return previous.bundle.get(port) if previous else None
