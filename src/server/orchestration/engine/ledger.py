"""The shared ledger state of one workflow instance and its observation helpers."""

from collections import Counter
from collections.abc import Callable, Iterable

from ...task.v2.representations.admission import ResidentAdmissionBinding
from ...task.v2.representations.operators import (
    AgentOperator,
    OperatorKind,
    ServiceDependency,
    operator_service_dependency,
)
from ...task.v2.representations.plan import EpisodeSpec, InferenceEmbodimentMenu
from ..journal import AppendOnlyList, LedgerJournal, TrackedDict, TrackedSet
from ..outcomes import classify_recovery
from ..private_state import PrivateStateLedger
from ..state import (
    Activation,
    Attempt,
    BranchDecision,
    ChildContext,
    Continuation,
    ControlState,
    ControlStatus,
    DeliveryContext,
    Invocation,
    IterationResolution,
    LedgerSnapshot,
    LoopInstance,
    Occurrence,
    OrchestrationEvent,
    ProgressAxis,
    ProgressCapability,
    Record,
    RecoveryDisposition,
    RegionJoinAggregate,
    Scope,
    WorkItem,
)
from ..telemetry import TelemetrySpanEmitter
from .failures import FailureLedger
from .topology import PlanTopology

# Activations created as a workflow runs, each charged to its activation budget: a
# spawned child, and an operator occurrence inside a region definition.
DYNAMIC_ACTIVATION_KINDS = frozenset({"child", "occurrence"})

_EVENT_FIELDS = frozenset(
    {"operator_id", "work_item_id", "attempt_id", "invocation_id", "slot_key"}
)


def control_key(occurrence: str) -> str:
    return f"control:{occurrence}"


def occurrence_key(operator_id: str, at: DeliveryContext) -> str:
    """The key of an operator's occurrence in a child context and nested loop time.

    A root occurrence is keyed by its operator id alone.
    """
    if not at.context_id and not at.time:
        return operator_id
    frames = "".join(f"/{frame.loop}:{frame.iteration}" for frame in at.time)
    return f"{operator_id}@{at.context_id}{frames}"


class OrchestrationLedger:
    """Holds one workflow instance's shared work-item, activation, scope, capability,
    attempt, invocation, record and trace collections, with their indexes, and appends
    the instance's events.

    Collections are mutated in place and never reassigned once restored: the span
    emitter and every component read them live.
    """

    def __init__(
        self,
        snapshot: LedgerSnapshot,
        topology: PlanTopology,
        failures: FailureLedger,
        emitter: TelemetrySpanEmitter,
        journal: LedgerJournal,
    ) -> None:
        self._topology = topology
        self._failures = failures
        self.emitter = emitter
        self.journal = journal
        self.workflow_instance = snapshot.instance
        self.root_scope = snapshot.root_scope
        self.root_grant = snapshot.root_grant
        self.next_seq = snapshot.next_seq
        self.private_state = PrivateStateLedger()
        self.scopes: TrackedDict[str, Scope] = TrackedDict(journal, "scopes")
        self.activations: TrackedDict[str, Activation] = TrackedDict(
            journal, "activations"
        )
        # Per-scope and dynamic activation counts that number and budget each child.
        self.scope_population: Counter[str] = Counter()
        self.scope_children: Counter[str] = Counter()
        self.dynamic_activations = 0
        self.work_items: TrackedDict[str, WorkItem] = TrackedDict(journal, "work_items")
        self.continuations: TrackedDict[str, Continuation] = TrackedDict(
            journal, "continuations"
        )
        self.records: AppendOnlyList[Record] = AppendOnlyList()
        self.region_aggregates: AppendOnlyList[RegionJoinAggregate] = AppendOnlyList()
        self.aggregate_by_join: dict[str, RegionJoinAggregate] = {}
        self.invocations: TrackedDict[str, Invocation] = TrackedDict(
            journal, "invocations"
        )
        self.attempts: TrackedDict[str, Attempt] = TrackedDict(journal, "attempts")
        self.capabilities: TrackedDict[tuple[str, ProgressAxis], ProgressCapability] = (
            TrackedDict(journal, "progress_capabilities")
        )
        self.trace: AppendOnlyList[OrchestrationEvent] = AppendOnlyList()
        # (agent activation, region operator) -> the synthetic opener activation that
        # owns that region's child-init scope.
        self.region_openers: dict[tuple[str, str], str] = {}
        self.wi_by_task: dict[str, str] = {}
        self.wi_by_occurrence: dict[str, str] = {}
        self.wi_by_activation: dict[str, str] = {}
        self.scope_by_activation: dict[str, str] = {}
        self.owner_acts_by_operator: dict[str, list[str]] = {}
        self.released_scopes: TrackedSet[str] = TrackedSet(journal, "released_scopes")
        # Occurrences of operators inside region definitions; a root operator's
        # occurrence is implicit.
        self.occurrences: TrackedDict[str, Occurrence] = TrackedDict(
            journal, "occurrences"
        )
        self.occurrences_by_scope: dict[str, set[str]] = {}
        self.occurrence_by_activation: dict[str, str] = {}
        self.subscopes: dict[str, dict[str, None]] = {}
        self.activations_by_scope: dict[str, list[str]] = {}
        # A scope's children by child index, and how many settled successfully.
        self.children_by_scope: dict[str, dict[int, str]] = {}
        self.succeeded_children: Counter[str] = Counter()
        # Each operator's first activation outside every region and dynamic context.
        self.static_activations: dict[str, str] = {}
        self.control_states: TrackedDict[str, ControlState] = TrackedDict(
            journal, "control_states"
        )
        self.branch_decisions: TrackedDict[str, BranchDecision] = TrackedDict(
            journal, "branch_decisions"
        )
        self.loop_instances: TrackedDict[str, LoopInstance] = TrackedDict(
            journal, "loop_instances"
        )
        self.loop_by_occurrence: dict[str, str] = {}
        self.iterations: TrackedDict[tuple[str, int], IterationResolution] = (
            TrackedDict(journal, "iteration_resolutions")
        )
        self.child_contexts: TrackedDict[str, ChildContext] = TrackedDict(
            journal, "child_contexts"
        )
        # Loop instances and definition children not yet closed.
        self.active_loops: set[str] = set()
        self.active_contexts: set[str] = set()
        # The occurrence that opened each scope it opened.
        self.scope_occurrence: dict[str, str] = {}
        # Candidates each summary a transition reads walks instead of its whole
        # collection, pruned as the walk finds one settled for good: per scope, its
        # occurrences, children and nested scopes that may still be open; work items a
        # task runs as that may still be unsettled; branch occurrences that may await
        # a selector read; spawn scopes that may still fan out; and work items with
        # every input resolved whose declared input ports may still lack an accepted
        # input.
        self.open_occurrences: dict[str, dict[str, None]] = {}
        self.open_children: dict[str, dict[str, None]] = {}
        self.open_subscopes: dict[str, dict[str, None]] = {}
        self.open_task_items: dict[str, None] = {}
        self.selection_candidates: dict[str, None] = {}
        self.fanout_candidates: dict[str, None] = {}
        self.input_candidates: dict[str, None] = {}

    def occurrence(self, key: str) -> Occurrence:
        """An occurrence by key; a root key is its operator's implicit occurrence."""
        if (occurrence := self.occurrences.get(key)) is not None:
            return occurrence
        return Occurrence(
            key=key, operator_id=key, context_id="", scope_id=self.root_scope.scope_id
        )

    def add_occurrence(self, occurrence: Occurrence) -> None:
        self.occurrences[occurrence.key] = occurrence
        self.occurrences_by_scope.setdefault(occurrence.scope_id, set()).add(
            occurrence.key
        )
        self.open_occurrences.setdefault(occurrence.scope_id, {})[occurrence.key] = None
        self.occurrence_by_activation[occurrence.activation_id] = occurrence.key

    def loop_time(self, activation_id: str) -> int:
        """The iteration of the innermost loop an activation runs in; 0 outside any."""
        key = self.occurrence_by_activation.get(activation_id)
        occurrence = self.occurrences.get(key) if key is not None else None
        return occurrence.time[-1].iteration if occurrence and occurrence.time else 0

    def occurrence_of_work_item(self, wi: WorkItem) -> str:
        """The occurrence a work item realizes; a root work item's is its operator."""
        return self.occurrence_by_activation.get(wi.activation_id, wi.operator_id)

    def bind_scope_occurrence(self, scope_id: str, key: str) -> None:
        """Record the occurrence that opened a scope."""
        self.scope_occurrence[scope_id] = key
        self.fanout_candidates[scope_id] = None

    def add_work_item(self, wi: WorkItem) -> None:
        self.work_items[wi.work_item_id] = wi
        if wi.legacy_task_id:
            self.open_task_items[wi.work_item_id] = None

    def set_continuation(self, continuation: Continuation) -> None:
        self.continuations[continuation.work_item_id] = continuation
        self.offer_inputs(continuation)

    def offer_inputs(self, continuation: Continuation) -> None:
        """Make a declared-input work item an input candidate once nothing it waits
        on is unresolved."""
        if continuation.required_ports and not continuation.waiting_on:
            self.input_candidates[continuation.work_item_id] = None

    def control_state(self, key: str) -> ControlState:
        """A control occurrence's state, created pending on first use."""
        if (state := self.control_states.get(key)) is None:
            state = self.control_states[key] = ControlState(key=key)
        return state

    def control_terminal(self, key: str) -> bool:
        state = self.control_states.get(key)
        return state is not None and state.status is not ControlStatus.PENDING

    def is_dynamic_activation(self, activation_id: str) -> bool:
        act = self.activations.get(activation_id)
        return act is not None and act.kind in DYNAMIC_ACTIVATION_KINDS

    def scope_subtree(self, root: str) -> list[str]:
        order = [root]
        seen = {root}
        cursor = 0
        while cursor < len(order):
            current = order[cursor]
            cursor += 1
            for scope_id in self.subscopes.get(current, ()):
                if scope_id not in seen:
                    seen.add(scope_id)
                    order.append(scope_id)
        return order

    def scope_work_items(
        self, scope_id: str, *, kinds: tuple[str, ...]
    ) -> list[WorkItem]:
        return [
            wi
            for activation_id in self.activations_by_scope.get(scope_id, ())
            if self.activations[activation_id].kind in kinds
            and (
                wi := self.work_items.get(self.wi_by_activation.get(activation_id, ""))
            )
            is not None
        ]

    def scope_control_states(self, scope_id: str) -> list[ControlState]:
        """The control states of the occurrences a scope holds."""
        if scope_id == self.root_scope.scope_id:
            keys: Iterable[str] = [
                key for key in self.control_states if key not in self.occurrences
            ]
        else:
            keys = self.occurrences_by_scope.get(scope_id, ())
        return [state for key in keys if (state := self.control_states.get(key))]

    def root_level(self, scope_id: str) -> bool:
        return self.scopes[scope_id].parent_scope_id == self.root_scope.scope_id

    def add_scope(self, scope: Scope) -> None:
        self.scopes[scope.scope_id] = scope
        if scope.parent_scope_id is not None:
            self.subscopes.setdefault(scope.parent_scope_id, {})[scope.scope_id] = None
            self.open_subscopes.setdefault(scope.parent_scope_id, {})[
                scope.scope_id
            ] = None

    def add_activation(self, activation: Activation) -> None:
        self.activations[activation.activation_id] = activation
        self.activations_by_scope.setdefault(activation.scope_id, []).append(
            activation.activation_id
        )
        self.scope_population[activation.scope_id] += 1
        if activation.kind == "child":
            self.scope_children[activation.scope_id] += 1
            self.children_by_scope.setdefault(activation.scope_id, {})[
                activation.child_index or 0
            ] = activation.activation_id
            self.open_children.setdefault(activation.scope_id, {})[
                activation.activation_id
            ] = None
        if activation.kind in DYNAMIC_ACTIVATION_KINDS:
            self.dynamic_activations += 1
        elif activation.kind != "region":
            self.static_activations.setdefault(
                activation.operator_id, activation.activation_id
            )

    def scope_closed(self, scope_id: str) -> bool:
        """Whether a scope closed: its join released, or it failed and every child it
        admitted has settled."""
        if scope_id in self.released_scopes:
            return True
        cap = self.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        return self._failures.scope_failed(scope_id) and cap is not None and cap.closed

    def emit_scope_owner(self, scope_id: str) -> None:
        if (owner := self.scopes[scope_id].owner_activation_id) is not None:
            self.emitter.emit_activation(owner)

    def template_closure(
        self,
        template: str,
        excluded: Callable[[str, list[str]], bool] | None = None,
    ) -> list[str]:
        """A child template and, under an agent template, the child templates of every
        region it declares, however deep; a template ``excluded`` rejects is left out
        together with what is nested under it."""
        closure: list[str] = []
        frontier = [template]
        while frontier:
            current = frontier.pop()
            if (
                current in closure
                or current in self.wi_by_occurrence
                or (excluded is not None and excluded(current, closure))
            ):
                continue
            closure.append(current)
            if isinstance(op := self._topology.operators.get(current), AgentOperator):
                frontier.extend(
                    nested
                    for ref in op.child_region_refs
                    if (nested := self._topology.child_template_of(ref.spawn_ref))
                    is not None
                )
        return closure

    def recovery_disposition(self, task_id: str) -> RecoveryDisposition | None:
        """Whether the task's operation may be recomputed or must be restored."""
        profile = self._topology.profiles.get(self._operator_for_task(task_id) or "")
        return classify_recovery(profile) if profile else None

    def contract_trace(self) -> list[tuple[str, str]]:
        """A compact (kind, subject) projection of the trace for test inspection."""
        return [
            (e.kind, e.operator_id or e.slot_key or e.work_item_id or "")
            for e in self.trace
        ]

    def region_scope_for(self, agent_activation: str, role: str) -> str | None:
        """The child-init scope an agent's declared role region opened, if entered."""
        agent = self.activations.get(agent_activation)
        op = self._topology.operators.get(agent.operator_id) if agent else None
        if not isinstance(op, AgentOperator):
            return None
        region_op = self._topology.agent_region_op(op, role)
        opener = self.region_openers.get((agent_activation, region_op or ""))
        return self.scope_by_activation.get(opener) if opener else None

    def region_closed(self, region_op: str) -> bool:
        if self._failures.region_failed(region_op):
            return True
        scope_id = (
            self.scope_id_for_join(region_op)
            if self._topology.kind(region_op) is OperatorKind.JOIN
            else self.scope_id_for(region_op)
        )
        return scope_id in self.released_scopes if scope_id else False

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The finite set of embodiments a task's plan node offers, if it offers one."""
        wi = self.work_item_for_task(task_id)
        if wi is None:
            return None
        return next(
            (
                node.embodiment_menu
                for node in self._topology.bundle.plan.nodes
                if node.embodiment_menu is not None
                and node.logical_ref == wi.operator_id
            ),
            None,
        )

    def episode_spec(self, task_id: str) -> EpisodeSpec | None:
        """The run-to-yield episode a task's operator lowers to, if the plan cut it."""
        wi = self.work_item_for_task(task_id)
        if wi is None:
            return None
        for node in self._topology.bundle.plan.nodes:
            if node.episode is None:
                continue
            if node.logical_ref == wi.operator_id or (
                wi.operator_id in node.episode.fused_refs
            ):
                return node.episode
        return None

    def agent_operator(self, task_id: str) -> AgentOperator | None:
        """The agent operator a dispatched task realizes, resolving its work item."""
        wi = self.work_item_for_task(task_id)
        operator_id = wi.operator_id if wi is not None else task_id
        op = self._topology.operators.get(operator_id)
        return op if isinstance(op, AgentOperator) else None

    def service_dependency(self, task_id: str) -> ServiceDependency | None:
        """The normalized resident dependency a dispatched task consumes, or None."""
        wi = self.work_item_for_task(task_id)
        operator_id = wi.operator_id if wi is not None else task_id
        return operator_service_dependency(self._topology.operators.get(operator_id))

    def resident_admission_binding(
        self, workflow_id: str, task_id: str
    ) -> ResidentAdmissionBinding | None:
        """The dependency a task consumes joined with its own plan node's
        annotations."""
        wi = self.work_item_for_task(task_id)
        operator_id = wi.operator_id if wi is not None else task_id
        dependency = operator_service_dependency(
            self._topology.operators.get(operator_id)
        )
        if dependency is None:
            return None
        node = next(
            (
                n
                for n in self._topology.bundle.plan.nodes
                if n.logical_ref == operator_id and n.embodiment_menu is None
            ),
            None,
        )
        return ResidentAdmissionBinding(
            workflow_id=workflow_id,
            dependency=dependency,
            requirement=node.service_family_requirement if node else None,
            intent=node.residency_intent if node else None,
        )

    def invocation_for_task(self, task_id: str) -> Invocation | None:
        wi = self.work_item_for_task(task_id)
        if wi is None or wi.invocation_id is None:
            return None
        return self.invocations.get(wi.invocation_id)

    def control_activation(self, operator_id: str) -> str:
        return self.static_activations.get(operator_id, operator_id)

    def scope_children_ordered(self, scope_id: str) -> list[Activation]:
        """Return a scope's children ordered by ``child_index``."""
        children = self.children_by_scope.get(scope_id, {})
        return [self.activations[children[index]] for index in sorted(children)]

    def scope_id_for_join(self, join_op: str) -> str | None:
        for edge in self._topology.bundle.template.edges:
            if (
                edge.to_op == join_op
                and self._topology.kind(edge.from_op) is OperatorKind.SPAWN
            ):
                return self.scope_id_for(edge.from_op)
        return None

    def handle_operator(self, handle: str) -> str:
        """The operator id a region handle names: the handle, or its activation's."""
        act = self.activations.get(handle)
        return act.operator_id if act else handle

    def resolve_opener_activation(self, handle: str) -> str | None:
        """The opener activation a region handle resolves to.

        An activation-id handle is its own opener (a recursive level); an operator-id
        handle resolves to its scope-owning activation, or, before the scope opens, its
        control activation.
        """
        if handle in self.activations:
            return handle
        if acts := self.owner_acts_by_operator.get(handle):
            return acts[-1]
        ctrl = self.control_activation(handle)
        return ctrl if ctrl in self.activations else None

    def scope_id_for(self, handle: str) -> str | None:
        opener = self.resolve_opener_activation(handle)
        return self.scope_by_activation.get(opener) if opener else None

    def emit(
        self, kind: str, *, detail: dict[str, str] | None = None, **fields: str | None
    ) -> None:
        self.trace.append(
            OrchestrationEvent(
                seq=self.next_seq,
                kind=kind,
                operator_id=fields.get("operator_id"),
                work_item_id=fields.get("work_item_id"),
                attempt_id=fields.get("attempt_id"),
                invocation_id=fields.get("invocation_id"),
                slot_key=fields.get("slot_key"),
                detail=detail
                or {
                    k: v
                    for k, v in fields.items()
                    if k not in _EVENT_FIELDS and v is not None
                },
            )
        )
        self.next_seq += 1

    def work_item_for_task(self, task_id: str) -> WorkItem | None:
        wi_id = self.wi_by_task.get(task_id)
        return self.work_items.get(wi_id) if wi_id else None

    def work_item_id_for_task(self, task_id: str) -> str | None:
        """The episode (work item) id backing a legacy task id, or None."""
        return self.wi_by_task.get(task_id)

    def _operator_for_task(self, task_id: str) -> str | None:
        wi = self.work_item_for_task(task_id)
        return wi.operator_id if wi else None

    def latest_attempt(self, wi: WorkItem) -> Attempt | None:
        return self.attempts.get(wi.attempt_ids[-1]) if wi.attempt_ids else None
