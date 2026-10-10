"""Spawn children and the regions an agent owns."""

from shared.utils import new_activation_id, new_work_item_id

from ...task.v2.representations.operators import (
    AgentOperator,
    LeafOperator,
    OperatorKind,
    SpawnRegion,
)
from ...task.v2.representations.template import BoundaryKind, EntryRole
from ..guardrails import ScopeBudget
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    AcceptedInput,
    AcceptedInputMember,
    Activation,
    CapabilityStatus,
    ChildContext,
    Continuation,
    ProgressAxis,
    PublicationOutcome,
    ValueMember,
    ValueRef,
    WorkItem,
)
from .advance import Advance, RegionError
from .authority import AuthorityLedger
from .dataflow import RegionFlow
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger
from .occurrences import OccurrenceFactory
from .scopes import ScopeProgress
from .topology import CHILD_INIT_OPENERS, PlanTopology, effect_recovery


class SpawnRegions:
    """Creates spawn children with their delegated grants and child-init inputs,
    seals and revokes spawns, and settles the regions an agent owns."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        inputs: AcceptedInputLedger,
        authority: AuthorityLedger,
        scope_progress: ScopeProgress,
        flow: RegionFlow,
        factory: OccurrenceFactory,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._inputs = inputs
        self._authority = authority
        self._scope_progress = scope_progress
        self._flow = flow
        self._factory = factory
        self._budget = budget

    def region_opener(self, agent_activation: str, region_op: str) -> str:
        """Mint or reuse the (agent activation, region) opener that owns its scope.

        Reuses the recursion opener machinery: a synthetic ``region`` activation of the
        spawn region owns a child-init scope nested under the agent's own scope, so the
        scope, its delegated grant, its progress, and its matched join resolve through
        the existing region path. The grant attenuates from the agent's delegate face
        and the region's per-site ceiling, never the agent's blanket ceiling.
        """
        key = (agent_activation, region_op)
        if (opener := self._ledger.region_openers.get(key)) is not None:
            return opener
        agent = self._ledger.activations[agent_activation]
        agent_op = self._topology.operators[agent.operator_id]
        assert isinstance(agent_op, AgentOperator)
        _, agent_delegate = self._authority.agent_face_tuples(agent_op, agent.scope_id)
        opener_act = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=agent.scope_id,
            operator_id=region_op,
            kind="region",
            parent_activation_id=agent_activation,
        )
        self._ledger.add_activation(opener_act)
        self._ledger.region_openers[key] = opener_act.activation_id
        scope_id = self._scope_progress.open_child_init_scope(
            opener_act.activation_id,
            parent_scope_id=agent.scope_id,
            parent_delegate=agent_delegate,
        )
        # An agent occurring in a region definition releases its region's join in its
        # own context and time.
        if occurrence := self._ledger.occurrence_by_activation.get(agent_activation):
            self._ledger.bind_scope_occurrence(scope_id, occurrence)
        return opener_act.activation_id

    def spawn_child(self, spawn: str, *, operator_id: str | None = None) -> str:
        """Materialize one child activation under a spawn/agent's open child scope."""
        activation, _ = self._create_child(
            spawn, operator_id, dispatchable=False, value_ref=None
        )
        return activation.activation_id

    def materialize_child(
        self,
        spawn: str,
        *,
        operator_id: str | None = None,
        value_ref: ValueRef | None = None,
    ) -> Advance:
        """Materialize one dispatchable child leaf and admit it as ready work."""
        advance = Advance()
        activation, wi = self._create_child(
            spawn, operator_id, dispatchable=True, value_ref=value_ref
        )
        self._init_child_input(activation, wi, value_ref)
        self._flow.admit(wi.work_item_id, advance)
        return advance

    def create_fanout_child(self, spawn: str, value_ref: ValueRef) -> str:
        """Create one producer-fanout child (unadmitted) and return its task id."""
        activation, wi = self._create_child(
            spawn, None, dispatchable=True, value_ref=value_ref
        )
        self._init_child_input(activation, wi, value_ref)
        return wi.legacy_task_id

    def _init_child_input(
        self, activation: Activation, wi: WorkItem, value_ref: ValueRef | None
    ) -> None:
        """Declare a child agent's input manifest and mint an inline child-init input.

        A child agent with a declared entry port gains a continuation requiring that
        port. An inline child-init value (a ``spawn_agent`` payload) is minted here; a
        producer-collection reference is left for the runtime to resolve and record.
        """
        entry_port = self._topology.agent_entry_port(wi.operator_id)
        if entry_port is None:
            return
        self._ledger.set_continuation(
            Continuation(
                work_item_id=wi.work_item_id, required_ports=frozenset({entry_port})
            )
        )
        if value_ref is not None and value_ref.kind == "inline":
            self._inputs.record_accepted_input(
                AcceptedInput(
                    activation_id=activation.activation_id,
                    target_port=entry_port,
                    occurrence_key=str(activation.child_index or 0),
                    provenance="spawn_element",
                    members=(
                        AcceptedInputMember(
                            source_operator_id=self._ledger.scopes[
                                activation.scope_id
                            ].owner_operator_id
                            or activation.operator_id,
                            source_activation_id=activation.activation_id,
                            child_index=activation.child_index,
                            outcome=PublicationOutcome.SUCCESS,
                            value_ref=value_ref,
                        ),
                    ),
                )
            )

    def _create_child(
        self,
        spawn: str,
        operator_id: str | None,
        *,
        dispatchable: bool,
        value_ref: ValueRef | None,
    ) -> tuple[Activation, WorkItem]:
        spawn_op = self._ledger.handle_operator(spawn)
        spawn_op_obj = self._topology.operators[spawn_op]
        child_ref = (
            spawn_op_obj.child_template_ref
            if isinstance(spawn_op_obj, (SpawnRegion, AgentOperator))
            else None
        )
        body_ref = operator_id or child_ref or spawn_op
        body_op = self._topology.operators.get(body_ref)
        if spawn_op in self._authority.denied_spawns:
            self._ledger.emit(
                "child_rejected", operator_id=spawn_op, detail={"reason": "denied"}
            )
            raise RegionError(f"spawn {spawn_op!r} was denied; no child may be created")
        scope_id = self._scope_progress.require_child_init_scope(spawn)
        cap = self._scope_progress.require_capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is not CapabilityStatus.OPEN:
            self._ledger.emit(
                "child_rejected",
                operator_id=spawn_op,
                detail={"reason": cap.status.value},
            )
            raise RegionError(
                f"spawn {spawn_op!r} child-init capability is {cap.status.value}; "
                "no child may be created"
            )
        body_opens_scope = self._topology.kind(body_ref) in CHILD_INIT_OPENERS
        # A leaf child dispatches directly; an agent child dispatches and owns its own
        # child-init scope (recursion). A spawn child body stays trace-level.
        if dispatchable and not isinstance(body_op, (LeafOperator, AgentOperator)):
            raise RegionError(
                f"child body {body_ref!r} is not live-dispatchable; only a leaf or "
                "agent body is"
            )
        if dispatchable and body_opens_scope and not isinstance(body_op, AgentOperator):
            raise RegionError(
                f"child body {body_ref!r} opens a scope; only a leaf or agent body is "
                "live-dispatchable"
            )
        # Validate every budget before materializing, so a rejected child leaves no
        # half-open region whose join could never drain.
        self._scope_progress.charge_activation()
        if body_opens_scope:
            self._scope_progress.check_scope_depth(scope_id)
        index = self._ledger.scope_population[scope_id]
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=scope_id,
            operator_id=body_ref,
            kind="child",
            child_index=index,
        )
        self._ledger.add_activation(activation)
        effect, recovery = effect_recovery(body_op)
        child_wi = WorkItem(
            work_item_id=new_work_item_id(),
            activation_id=activation.activation_id,
            operator_id=body_ref,
            legacy_task_id=activation.activation_id if dispatchable else "",
            value_ref=value_ref,
            child_input=value_ref,
            effect_class=effect,
            recovery=recovery,
            replay_contract=self._topology.replay.get(body_ref),
        )
        self._ledger.add_work_item(child_wi)
        self._ledger.wi_by_activation[activation.activation_id] = child_wi.work_item_id
        if dispatchable:
            self._ledger.wi_by_task[child_wi.legacy_task_id] = child_wi.work_item_id
        cap.outstanding += 1
        self._ledger.emit(
            "child_spawned",
            operator_id=body_ref,
            detail={"scope": scope_id, "index": str(index)},
        )
        # A spawn child body opens its nested scope eagerly; a dispatchable agent child
        # opens its own child-init scope lazily, only when it first spawns.
        if self._topology.kind(body_ref) is OperatorKind.SPAWN:
            self._scope_progress.open_child_init_scope(
                activation.activation_id, parent_scope_id=scope_id
            )
        return activation, child_wi

    def seal_spawn(self, spawn: str) -> Advance:
        """Seal a spawn's child-init capability; no further children may be created."""
        scope_id = self._scope_progress.require_child_init_scope(spawn)
        cap = self._scope_progress.require_capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._flow.maybe_release_join(scope_id)

    def _settle_agent_regions(self, activation_id: str) -> Advance:
        """Settle every declared child region of a completed agent under its residual.

        Each entered region seals its open child-init capability so its join releases on
        drain; a declared-but-never-entered region opens as a zero-child region and
        seals, so its join releases empty. A cancel residual revokes and cancels the
        region's children instead. Opening an unused region is skipped when it would
        exceed the scope-depth budget, since the agent could never have entered it. A
        non-spawning agent owns no region.
        """
        advance = Advance()
        agent = self._ledger.activations.get(activation_id)
        op = self._topology.operators.get(agent.operator_id) if agent else None
        if not isinstance(op, AgentOperator) or agent is None:
            return advance
        room = (
            self._ledger.scopes[agent.scope_id].depth + 1
            <= self._budget.max_scope_depth
        )
        for ref in op.child_region_refs:
            opener = self._ledger.region_openers.get((activation_id, ref.spawn_ref))
            if opener is None:
                if not room:
                    continue
                opener = self.region_opener(activation_id, ref.spawn_ref)
            advance.extend(self._settle_owned_region(opener))
        return advance

    def agent_terminal_regions(self, operator_id: str, activation_id: str) -> Advance:
        """Settle an agent's declared regions when the agent completes, else no-op."""
        if isinstance(self._topology.operators.get(operator_id), AgentOperator):
            return self._settle_agent_regions(activation_id)
        return Advance()

    def _settle_owned_region(self, opener: str) -> Advance:
        scope_id = self._ledger.scope_by_activation.get(opener)
        advance = Advance()
        if scope_id is None or not self._flow.close_owned_region(scope_id, advance):
            return advance
        return advance.extend(self._flow.maybe_release_join(scope_id))

    def revoke_spawn(self, spawn: str) -> None:
        """Revoke a spawn's child-init capability as a progress transition."""
        scope_id = self._scope_progress.require_child_init_scope(spawn)
        cap = self._scope_progress.require_capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED
            self._ledger.emit(
                "child_init_revoked",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )

    def settle_child(
        self,
        child_activation_id: str,
        *,
        outcome: PublicationOutcome = PublicationOutcome.SUCCESS,
        value_ref: ValueRef | None = None,
    ) -> Advance:
        """Record a child activation's terminal outcome and drain its capability."""
        activation = self._ledger.activations.get(child_activation_id)
        if activation is None or activation.kind != "child":
            raise RegionError(f"unknown child activation {child_activation_id!r}")
        wi = self._ledger.work_items[self._ledger.wi_by_activation[child_activation_id]]
        return self._flow.settle_child_wi(wi, activation, outcome, value_ref)

    def enter_definition_child(
        self, spawn_key: str, index: int, element: ValueRef
    ) -> Advance:
        """Create the child of a spawn whose child is a region definition for the
        element at ``index`` of its fan-out; an element already entered is left as it
        is.

        The child is a new activation context: its members occur once in it, at the
        spawn's own time, entered with the spawned element as the definition's param
        and the spawn's captured values. The whole batch is budgeted before any of it
        is created.
        """
        advance = Advance()
        occurrence = self._ledger.occurrence(spawn_key)
        spawn = self._topology.operators[occurrence.operator_id]
        if not isinstance(spawn, SpawnRegion) or spawn.child_definition_ref is None:
            raise RegionError(f"{spawn_key!r} spawns no region definition")
        opener = occurrence.activation_id or self._ledger.control_activation(
            occurrence.operator_id
        )
        scope_id = self._ledger.scope_by_activation.get(opener)
        if scope_id is not None and index in self._ledger.children_by_scope.get(
            scope_id, {}
        ):
            return advance
        cap = self._scope_progress.capability(scope_id, ProgressAxis.CHILD_INIT)
        if scope_id is None or cap is None or cap.status is not CapabilityStatus.OPEN:
            raise RegionError(f"spawn {spawn_key!r} admits no further child")
        definition = self._topology.definitions[spawn.child_definition_ref]
        self._scope_progress.charge_activations(1 + len(definition.members))
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=scope_id,
            operator_id=occurrence.operator_id,
            kind="child",
            child_index=index,
        )
        self._ledger.add_activation(activation)
        wi = WorkItem(
            work_item_id=new_work_item_id(),
            activation_id=activation.activation_id,
            operator_id=occurrence.operator_id,
            legacy_task_id="",
            child_input=element,
        )
        self._ledger.add_work_item(wi)
        self._ledger.wi_by_activation[activation.activation_id] = wi.work_item_id
        cap.outstanding += 1
        captured = self._ledger.control_state(spawn_key).inputs
        entries: dict[str, ValueRef] = {}
        for port in definition.inputs:
            if port.role is EntryRole.PARAM:
                entries[port.name] = element
            elif (value := captured.get(port.name)) is not None:
                entries[port.name] = value
        context_scope = self._scope_progress.open_context_scope(
            activation.activation_id, scope_id
        )
        context = ChildContext(
            context_id=activation.activation_id,
            scope_id=context_scope,
            time=occurrence.time,
            definition_id=definition.definition_id,
            entries=entries,
        )
        self._ledger.child_contexts[activation.activation_id] = context
        self._ledger.active_contexts.add(activation.activation_id)
        self._ledger.emit(
            "child_spawned",
            operator_id=occurrence.operator_id,
            detail={"scope": scope_id, "index": str(activation.child_index)},
        )
        for key in self._factory.enter(definition.definition_id, context):
            self._flow.evaluate(key, advance)
        return advance

    def on_child_return(self, key: str, advance: Advance) -> None:
        """Accept the value a child's definition returns through its return ports."""
        occurrence = self._ledger.occurrence(key)
        context = self._ledger.child_contexts.get(occurrence.context_id)
        if context is None or context.returned:
            return
        definition = self._topology.definitions[context.definition_id]
        bundles = self._flow.edges.return_bundles(key, BoundaryKind.RETURN)
        if not bundles:
            return
        values = next(iter(bundles.values()))
        if len(definition.returns) == 1:
            context.result = values.get(definition.returns[0].name)
        else:
            context.result = ValueRef(
                kind="bundle",
                members=tuple(
                    ValueMember(
                        key=name, outcome=PublicationOutcome.SUCCESS, value_ref=value
                    )
                    for name, value in sorted(values.items())
                ),
            )
        context.returned = True
        self._ledger.emit(
            "child_returned",
            operator_id=occurrence.operator_id,
            detail={"context": context.context_id},
        )
        self.maybe_settle_child(context, advance)

    def maybe_settle_child(self, context: ChildContext, advance: Advance) -> None:
        """Settle a definition child once everything it started has drained: with its
        returned value, or as a failure when it never returned."""
        wi = self._child_work_item(context)
        if wi is None:
            self._ledger.active_contexts.discard(context.context_id)
            return
        if not self._scope_progress.scope_drained(context.scope_id):
            return
        if not context.returned:
            wi.failure_reason = (
                f"child {context.definition_id} drained without returning a value"
            )
            self._settle_definition_child(
                context, wi, PublicationOutcome.DECLARED_FAILURE, advance
            )
            return
        self._settle_definition_child(context, wi, PublicationOutcome.SUCCESS, advance)

    def fail_child(self, context: ChildContext, reason: str, advance: Advance) -> None:
        """Fail a definition child, withdrawing what it still has outstanding."""
        wi = self._child_work_item(context)
        if wi is None:
            return
        for scope_id in self._ledger.scope_subtree(context.scope_id):
            advance.extend(self._flow.cancel_one_scope(scope_id))
        wi.failure_reason = reason
        self._settle_definition_child(
            context, wi, PublicationOutcome.DECLARED_FAILURE, advance
        )

    def _settle_definition_child(
        self,
        context: ChildContext,
        wi: WorkItem,
        outcome: PublicationOutcome,
        advance: Advance,
    ) -> None:
        self._ledger.released_scopes.add(context.scope_id)
        self._ledger.active_contexts.discard(context.context_id)
        activation = self._ledger.activations[wi.activation_id]
        value = context.result if outcome is PublicationOutcome.SUCCESS else None
        advance.extend(self._flow.settle_child_wi(wi, activation, outcome, value))

    def _child_work_item(self, context: ChildContext) -> WorkItem | None:
        wi_id = self._ledger.wi_by_activation.get(context.context_id)
        wi = self._ledger.work_items.get(wi_id) if wi_id else None
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return None
        return wi
