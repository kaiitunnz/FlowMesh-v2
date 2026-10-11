"""Record flow, admission, join release and failure propagation through regions."""

from typing import Any, Protocol

from ...task.v2.representations.operators import (
    AgentOperator,
    BranchRegion,
    JoinCompletion,
    JoinRegion,
    MergeCombination,
    MergeRegion,
    OperatorKind,
    ResidualPolicy,
    SpawnRegion,
    is_spawn_fanout_port,
    spawned_only_region_owners,
)
from ...task.v2.representations.template import DependencyUse
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    Activation,
    AuthorityDecision,
    AuthorityDecisionKind,
    BranchDecision,
    CapabilityStatus,
    ControlStatus,
    DenialKind,
    Occurrence,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    Record,
    RegionAggregateMember,
    RegionJoinAggregate,
    ValueMember,
    ValueRef,
    WorkItem,
    WorkItemStatus,
)
from .advance import (
    Advance,
    dependency_failed,
)
from .authority import AuthorityLedger
from .edges import EdgeResolver, EdgeState, Incoming
from .failures import DECLARED_FAILURE_REASON, FailureLedger
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger, control_key
from .publications import PublicationLedger
from .scopes import ScopeProgress
from .topology import PlanTopology

_LIVE_STATES = frozenset({EdgeState.LIVE, EdgeState.EMPTY})
_BROKEN_STATES = frozenset({EdgeState.FAILED, EdgeState.CANCELLED})


def broken_input_reason(region: str, broken: Incoming) -> str:
    """Why a region failed on an input that failed or was cancelled."""
    ended = "was cancelled" if broken.state is EdgeState.CANCELLED else "failed"
    return f"{region} input {broken.edge.from_op} {ended}"


class ContextRegions(Protocol):
    """What runs a region definition's occurrences: a loop or a spawned child."""

    def ingress(self, key: str, inputs: list[Incoming], advance: Advance) -> None: ...

    def on_return(self, key: str, advance: Advance) -> None: ...

    def on_failure(
        self, occurrence: Occurrence, reason: str, advance: Advance
    ) -> None: ...

    def on_cancelled(self, scope_id: str) -> None: ...


def selected_port(
    branch: BranchRegion, value: Any
) -> tuple[str | None, str | None, str]:
    """The port a branch's rule selects for a selector value: (port, case, reason).

    The value must be a string; with cases it must equal a case value, and without
    them it must name an output port. Anything else selects nothing, with a reason.
    """
    rule = branch.rule
    if rule is None:
        return None, None, "the branch declares no selection rule"
    if not isinstance(value, str):
        kind = "missing" if value is None else type(value).__name__
        return None, None, f"selector value is {kind}, not a string"
    if rule.cases is not None:
        case = next((c for c in rule.cases if c.value == value), None)
        if case is None:
            return None, None, f"selector value {value!r} matches no case"
        return case.port, case.value, ""
    if value not in {port.name for port in branch.outputs}:
        return None, None, f"selector value {value!r} names no output port"
    return value, None, ""


_EARLY_JOINS = frozenset(
    {JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE}
)


def _merged_value(op: MergeRegion, inputs: list[Incoming]) -> ValueRef:
    """A merge's value over its resolved inputs: its one live record under
    ``one_live``, or every live record in input order."""
    if op.combination is MergeCombination.ONE_LIVE:
        live = next((i for i in inputs if i.state in _LIVE_STATES), None)
        return (live.value if live else None) or ValueRef(kind="empty")
    return ValueRef(
        kind="aggregate",
        members=tuple(
            ValueMember(
                key=i.port or str(index),
                outcome=(
                    PublicationOutcome.EXPLICIT_EMPTY
                    if i.state is EdgeState.EMPTY
                    else PublicationOutcome.SUCCESS
                ),
                value_ref=i.value,
            )
            for index, i in enumerate(inputs)
            if i.state in _LIVE_STATES
        ),
    )


def _port_outputs(op: MergeRegion | JoinRegion, value: ValueRef) -> dict[str, ValueRef]:
    """A merge's or join's value under each output port it declares.

    A call's join carries its one child's returned value: each return port the value
    returned through it, and a port the child returned nothing through no value.
    """
    ports = [port.name for port in op.outputs]
    if value.kind == "empty":
        return dict.fromkeys(ports, value)
    if not isinstance(op, JoinRegion) or not op.call:
        return {port: value for port in ports[:1]}
    returned = (
        value.members[0].value_ref
        if value.kind == "aggregate" and len(value.members) == 1
        else None
    )
    if returned is None:
        return {}
    if len(ports) == 1:
        return {ports[0]: returned}
    members = (
        {member.key: member.value_ref for member in returned.members}
        if returned.kind == "bundle"
        else {}
    )
    return {
        port: value_ref
        for port in ports
        if (value_ref := members.get(port)) is not None
    }


class RegionFlow:
    """Delivers records to their successors and admits ready work, releases joins
    under their completion and residual rules, and settles declared failures and
    cancellations through every region they reach."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        failures: FailureLedger,
        publication: PublicationLedger,
        inputs: AcceptedInputLedger,
        authority: AuthorityLedger,
        scope_progress: ScopeProgress,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._failures = failures
        self._publication = publication
        self._inputs = inputs
        self._authority = authority
        self._scope_progress = scope_progress
        self.edges = EdgeResolver(ledger, topology)
        self.contexts: ContextRegions | None = None

    def settle_failed_wi(self, wi: WorkItem) -> Advance:
        """Settle a work item as a declared failure: a child drains its scope, an
        occurrence inside a region definition fails the loop or child running it, and
        anything else cascades over its static successors. A failed agent fails the
        regions it declares either way."""
        activation = self._ledger.activations[wi.activation_id]
        if activation.kind == "occurrence":
            return self._fail_occurrence_wi(wi)
        if activation.kind != "child":
            return self._settle_failure(wi.work_item_id)
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        # A child's terminal failure drains its scope account and lets an all-succeed
        # join fail; it does not cascade over static successors.
        advance = self.settle_child_wi(
            wi, activation, PublicationOutcome.DECLARED_FAILURE, None
        )
        cascade = Advance(failed=[wi.legacy_task_id])
        self._fail_agent_regions(wi, cascade, set())
        self._declare_failures(wi, cascade.failed)
        advance.failed[:0] = cascade.failed
        advance.cancelled.extend(cascade.cancelled)
        return advance

    def reconsider_admission(self, task_id: str) -> Advance:
        """Re-attempt admission of a work item after its input manifest changed."""
        advance = Advance()
        wi = self._ledger.work_item_for_task(task_id)
        if wi is not None and wi.status is WorkItemStatus.BLOCKED:
            self.admit(wi.work_item_id, advance)
        return advance

    def close_owned_region(self, scope_id: str, advance: Advance) -> bool:
        """Close a terminal agent's open region under its residual policy, adding the
        children it cancels to ``advance``; returns whether it was open."""
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is None or cap.status is not CapabilityStatus.OPEN:
            return False
        owner = self._ledger.scopes[scope_id].owner_operator_id
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.DRAIN) is ResidualPolicy.CANCEL:
            cap.status = CapabilityStatus.REVOKED
            self._ledger.emit(
                "child_init_revoked",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
            advance.cancelled.extend(self._cancel_residual_subtrees(scope_id))
        else:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_sealed",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
        return True

    def settle_child_wi(
        self,
        wi: WorkItem,
        activation: Activation,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
    ) -> Advance:
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        wi.value_ref = value_ref
        if outcome is PublicationOutcome.SUCCESS:
            self._ledger.succeeded_children[activation.scope_id] += 1
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        cap = self._scope_progress.require_capability(
            activation.scope_id, ProgressAxis.CHILD_INIT
        )
        cap.outstanding = max(0, cap.outstanding - 1)
        spawn_op = self._ledger.scopes[activation.scope_id].owner_operator_id or ""
        self._publication.publish_keyed(spawn_op, activation, outcome, value_ref)
        self._ledger.emit(
            "child_settled",
            operator_id=activation.operator_id,
            detail={"scope": activation.scope_id, "outcome": outcome.value},
        )
        return self.maybe_release_join(activation.scope_id)

    def pending_branch_reads(self) -> list[tuple[str, ValueRef]]:
        """Each branch occurrence awaiting a selector read, with the input it reads."""
        reads: list[tuple[str, ValueRef]] = []
        candidates = self._ledger.selection_candidates
        for key in list(candidates):
            state = self._ledger.control_states[key]
            op = self._topology.operators.get(self._ledger.occurrence(key).operator_id)
            if (
                not isinstance(op, BranchRegion)
                or state.status is not ControlStatus.PENDING
                or key in self._ledger.branch_decisions
            ):
                del candidates[key]
            elif (value := state.inputs.get(op.rule.input)) is not None:
                reads.append((key, value))
        return reads

    def accept_branch_selection(
        self, key: str, value: Any, *, error: str | None = None
    ) -> Advance:
        """Route a branch occurrence by the selector value read from its input.

        Selects the one port the rule names, releasing on it the value of the input
        the branch forwards, and resolves every other port dead, or fails the branch
        when the value selects nothing or ``error`` says its input could not be read.
        An occurrence already decided, or no longer pending, is left as it is.
        """
        advance = Advance()
        state = self._ledger.control_states.get(key)
        occurrence = self._ledger.occurrence(key)
        branch = self._topology.operators.get(occurrence.operator_id)
        if (
            state is None
            or state.status is not ControlStatus.PENDING
            or key in self._ledger.branch_decisions
            or not isinstance(branch, BranchRegion)
            or branch.rule.input not in state.inputs
            or branch.forward not in state.inputs
        ):
            return advance
        port, case, reason = (
            (None, None, error) if error else selected_port(branch, value)
        )
        if port is None:
            self.fail_control(
                key, f"BranchSelectionInvalid: {reason}", advance, fault=True
            )
            return advance
        forwarded = state.inputs[branch.forward]
        self._ledger.branch_decisions[key] = BranchDecision(
            occurrence=key,
            port=port,
            case=case,
            rule_version=branch.rule.version,
            input_ref=state.inputs[branch.rule.input],
        )
        state.status = ControlStatus.LIVE
        state.outputs = {**state.outputs, port: forwarded}
        self._ledger.emit(
            "branch_routed", operator_id=occurrence.operator_id, detail={"port": port}
        )
        self.propagate(key, advance, value=forwarded, port=port)
        return advance

    def cancel_one_scope(self, scope_id: str) -> Advance:
        self._ledger.emit("scope_cancelled", detail={"scope": scope_id})
        self._scope_progress.revoke_progress(scope_id)
        scope = self._ledger.scopes[scope_id]
        cancelled = self._apply_cancellation_residual(scope_id)
        for wi in self._ledger.scope_work_items(
            scope_id, kinds=("leaf", "agent", "occurrence")
        ):
            if wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                self._cancel_work_item(wi)
                cancelled.append(wi.legacy_task_id)
        for state in self._ledger.scope_control_states(scope_id):
            if state.status is ControlStatus.PENDING:
                state.status = ControlStatus.CANCELLED
        if self.contexts is not None:
            self.contexts.on_cancelled(scope_id)
        if scope.grant_id and scope.grant_id in self._authority.grants:
            grant = self._authority.grants[scope.grant_id]
            if not grant.revoked:
                self._authority.store_grant(grant.model_copy(update={"revoked": True}))
                self._ledger.emit(
                    "grant_revoked",
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )
        advance = self._resolve_cancelled_outputs(scope_id)
        advance.cancelled.extend(cancelled)
        return advance

    def _apply_cancellation_residual(self, scope_id: str) -> list[str]:
        """Apply a cancelled scope's join residual policy to its materialized children;
        returns the task ids it cancelled.

        A declared ``drain``/``continue`` leaves materialized children to settle; the
        default (``cancel``, and any scope without a declared policy) cancels every not-
        yet-settled child.
        """
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.CANCEL) is ResidualPolicy.CANCEL:
            return [
                wi.legacy_task_id
                for wi in self._cancel_residual_children(scope_id)
                if wi.legacy_task_id
            ]
        return []

    def _resolve_cancelled_outputs(self, scope_id: str) -> Advance:
        """Release a cancelled scope's join as its cancellation outcome, at the join
        occurrence its spawn opened it under."""
        if scope_id in self._ledger.released_scopes:
            return Advance()
        join = self._join_of_scope(scope_id)
        if join is None:
            return Advance()
        outcome = self._no_winner_outcome(join)
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        self._ledger.emit(
            "join_released",
            operator_id=join.operator_id,
            detail={"outcome": outcome.value, "cancelled": "true"},
        )
        self._scope_progress.frontier_closed(scope_id)
        advance = Advance()
        if (join_key := self._join_key(scope_id, join.operator_id)) is None:
            return advance
        empty = ValueRef(kind="empty")
        state = self._ledger.control_state(join_key)
        if state.status is ControlStatus.PENDING:
            state.status = ControlStatus.LIVE
            state.outputs = _port_outputs(join, empty)
        self._publish_control(join_key, outcome, empty)
        self.propagate(join_key, advance, value=empty)
        return advance

    def _gates_open(self, join_key: str) -> bool:
        """Whether every route a join occurrence runs only on resolved live."""
        return all(
            i.state in _LIVE_STATES
            for i in self.edges.incoming(join_key)
            if i.use is not DependencyUse.ORDER_ONLY
        )

    def _join_scope(self, occurrence: Occurrence, join: JoinRegion) -> str | None:
        """The child-init scope a join occurrence collects: its spawn's, in its own
        context and time."""
        spawn_op = next(
            (
                e.from_op
                for e in self._topology.bundle.template.edges
                if e.to_op == join.operator_id
                and self._topology.kind(e.from_op) is OperatorKind.SPAWN
            ),
            None,
        )
        if spawn_op is None:
            return None
        spawn = self._ledger.occurrence(self.edges.sibling(occurrence, spawn_op))
        opener = spawn.activation_id or self._ledger.control_activation(spawn_op)
        return self._ledger.scope_by_activation.get(opener)

    def _join_key(self, scope_id: str, join_op: str) -> str | None:
        """The join occurrence a child-init scope releases into: the one in the context
        and time of the occurrence that opened the scope, or None for a nested level of
        an agent's recursive region, which delivers nothing."""
        if (spawn_key := self._ledger.scope_occurrence.get(scope_id)) is not None:
            return self.edges.sibling(self._ledger.occurrence(spawn_key), join_op)
        return join_op if self._ledger.root_level(scope_id) else None

    def _cancel_work_item(self, wi: WorkItem) -> None:
        # A cancelled in-flight external effect is not compensated here; compensation on
        # cancel rides with the deferred effect-commit machinery.
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        wi.status = WorkItemStatus.CANCELLED
        self._ledger.private_state.release(wi.activation_id)
        self._publication.publish(
            wi.operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
        )
        # Recorded before the emitter reads the trace: a cancelled item with no attempt
        # has no other event carrying its work_item_id, so this is the sole source the
        # work-item span's "or latest matching event" end-time rule can find.
        self._ledger.emit(
            "work_item_cancelled",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
        )
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)

    def propagate(
        self,
        key: str,
        advance: Advance,
        *,
        value: ValueRef | None = None,
        port: str | None = None,
    ) -> None:
        """Resolve a settled occurrence's outgoing routes and return bindings, and
        reconsider each occurrence they lead to."""
        occurrence = self._ledger.occurrence(key)
        if value is not None or port is not None:
            self._ledger.records.append(
                Record(
                    operator_id=occurrence.operator_id,
                    activation_id=occurrence.activation_id
                    or self._ledger.control_activation(occurrence.operator_id),
                    scope_id=occurrence.scope_id,
                    value_ref=value,
                    occurrence=key,
                    source_port=port,
                )
            )
            self._ledger.emit("record_delivered", operator_id=occurrence.operator_id)
        targets = sorted(
            {e.to_op for e in self._topology.outgoing.get(occurrence.operator_id, ())}
        )
        for target in targets:
            self._release(
                self.edges.sibling(occurrence, target), occurrence.operator_id, advance
            )
        if self._topology.returns_from.get(occurrence.operator_id) and self.contexts:
            self.contexts.on_return(key, advance)

    def _release(self, target: str, from_op: str, advance: Advance) -> None:
        """Note one resolved input of an occurrence; reconsider it once all have."""
        operator_id = self._ledger.occurrence(target).operator_id
        if self._topology.is_control(operator_id):
            cont = self._ledger.continuations.get(control_key(target))
            root = target not in self._ledger.occurrences
            if (
                cont is None
                or self._ledger.control_terminal(target)
                or (root and self._failures.region_failed(operator_id))
            ):
                return
        else:
            wi_id = self._ledger.wi_by_occurrence.get(target)
            cont = self._ledger.continuations.get(wi_id) if wi_id else None
            if cont is None:
                return
        cont.waiting_on = cont.waiting_on - {from_op}
        if not cont.waiting_on:
            self._ledger.offer_inputs(cont)
            self.evaluate(target, advance)

    def evaluate(self, key: str, advance: Advance) -> None:
        """Apply an occurrence's input contract once every input has resolved.

        A leaf or agent runs, or is inactive when a value or route it requires is
        dead; a control operator applies its own combination, selection, entry or
        child-region rule.
        """
        occurrence = self._ledger.occurrence(key)
        op = self._topology.operators[occurrence.operator_id]
        inputs = self.edges.incoming(key)
        if any(i.state is EdgeState.PENDING for i in inputs):
            return
        if not self._topology.is_control(op.operator_id):
            wi_id = self._ledger.wi_by_occurrence.get(key)
            wi = self._ledger.work_items.get(wi_id) if wi_id else None
            if wi is None or wi.status is not WorkItemStatus.BLOCKED:
                return
            required = [i for i in inputs if i.use is not DependencyUse.ORDER_ONLY]
            if failed := [i for i in required if i.state is EdgeState.FAILED]:
                # A value or route that resolved as a failure delivers nothing to run
                # on.
                wi.failure_reason = dependency_failed(failed[0].edge.from_op)
                advance.extend(self.settle_failed_wi(wi))
                return
            if any(i.state is EdgeState.CANCELLED for i in required):
                self._cancel_work_item(wi)
                advance.cancelled.append(wi.legacy_task_id)
                return
            if self._inactive(occurrence, op, inputs):
                self.mark_dead(key, advance)
            else:
                self._record_entry_inputs(occurrence, wi, inputs)
                self.admit(wi.work_item_id, advance)
            return
        if self._ledger.control_terminal(key):
            return
        required = [i for i in inputs if i.use is not DependencyUse.ORDER_ONLY]
        match op:
            case MergeRegion():
                self._combine(key, op, inputs, advance)
            case BranchRegion():
                self._await_selection(key, op, inputs, advance)
            case SpawnRegion():
                if any(i.state is EdgeState.DEAD for i in required):
                    self.mark_dead(key, advance)
                else:
                    self._open_spawn(key, occurrence, inputs)
            case _ if op.kind is OperatorKind.LOOP_CONTEXT:
                if any(i.state is EdgeState.DEAD for i in required):
                    self.mark_dead(key, advance)
                elif self.contexts is not None:
                    self.contexts.ingress(key, inputs, advance)
            case JoinRegion():
                # A join releases on its spawn's scope closure once every route it runs
                # only on is live; a dead or failed one settles it instead.
                if any(i.state is EdgeState.DEAD for i in required):
                    self.mark_dead(key, advance)
                elif broken := [i for i in required if i.state in _BROKEN_STATES]:
                    self.fail_control(
                        key, broken_input_reason("join", broken[0]), advance
                    )
                elif (scope_id := self._join_scope(occurrence, op)) is not None:
                    advance.extend(self.maybe_release_join(scope_id))
            case _:
                pass

    def _inactive(
        self, occurrence: Occurrence, op: Any, inputs: list[Incoming]
    ) -> bool:
        """Whether a leaf or agent has no live route to run on.

        A dead required value or route makes it inactive; with only ordering inputs, it
        runs when any of them is live.
        """
        required = [i for i in inputs if i.use is not DependencyUse.ORDER_ONLY]
        if any(i.state is EdgeState.DEAD for i in required):
            return True
        ordering = [i for i in inputs if i.use is DependencyUse.ORDER_ONLY]
        return (
            not required
            and bool(ordering)
            and all(i.state is EdgeState.DEAD for i in ordering)
        )

    def _record_entry_inputs(
        self, occurrence: Occurrence, wi: WorkItem, inputs: list[Incoming]
    ) -> None:
        """Record the accepted inputs an agent occurrence takes on declared ports."""
        self._inputs.record_routed_inputs(
            occurrence,
            wi,
            [
                (
                    i.edge if i.edge.is_forward else None,
                    i.port or "",
                    (
                        self.edges.sibling(occurrence, i.edge.from_op)
                        if i.edge.is_forward
                        else None
                    ),
                    i.value,
                )
                for i in inputs
                if i.state in _LIVE_STATES
            ],
        )

    def _combine(
        self, key: str, op: MergeRegion, inputs: list[Incoming], advance: Advance
    ) -> None:
        """Combine a merge's live inputs: forward its one live record under one_live,
        or freeze every live record in input order."""
        live = [i for i in inputs if i.state in _LIVE_STATES]
        if not live:
            self.mark_dead(key, advance)
            return
        if op.combination is MergeCombination.ONE_LIVE and len(live) > 1:
            self.fail_control(
                key,
                f"one_live merge {op.operator_id} received {len(live)} live inputs",
                advance,
                fault=True,
            )
            return
        value = _merged_value(op, inputs)
        state = self._ledger.control_state(key)
        state.status = ControlStatus.LIVE
        state.outputs = _port_outputs(op, value)
        self._ledger.emit("merge_combined", operator_id=op.operator_id)
        empty = op.combination is MergeCombination.ONE_LIVE and (
            live[0].state is EdgeState.EMPTY or value.kind == "empty"
        )
        self._publish_control(
            key,
            (
                PublicationOutcome.EXPLICIT_EMPTY
                if empty
                else PublicationOutcome.SUCCESS
            ),
            value,
        )
        self.propagate(key, advance, value=value)

    def _await_selection(
        self, key: str, op: BranchRegion, inputs: list[Incoming], advance: Advance
    ) -> None:
        """Hold a branch whose inputs resolved for the read of its selector.

        Its selection and forwarded inputs both take part: one that failed or was
        cancelled fails the branch, and a dead one leaves it inactive. The values
        both bind are fixed here, so the decision reads and releases exactly these.
        """
        ports = [i.port for i in inputs]
        participating = {
            index
            for name in (op.rule.input, op.forward)
            if (index := op.input_index(ports, name)) is not None
        }
        required = [
            item
            for index, item in enumerate(inputs)
            if index in participating or item.use is not DependencyUse.ORDER_ONLY
        ]
        if broken := [i for i in required if i.state in _BROKEN_STATES]:
            self.fail_control(key, broken_input_reason("branch", broken[0]), advance)
            return
        if any(i.state is EdgeState.DEAD for i in required):
            self.mark_dead(key, advance)
            return
        index = op.input_index(ports, op.rule.input)
        selected = inputs[index] if index is not None else None
        index = op.input_index(ports, op.forward)
        forwarded = inputs[index] if index is not None else None
        if (
            selected is None
            or selected.state is EdgeState.EMPTY
            or selected.value is None
        ):
            self.fail_control(
                key,
                "BranchSelectionInvalid: its selection input is empty",
                advance,
                fault=True,
            )
            return
        state = self._ledger.control_state(key)
        state.inputs = {
            **state.inputs,
            op.rule.input: selected.value,
            op.forward: (
                forwarded.value
                if forwarded is not None and forwarded.value is not None
                else ValueRef(kind="empty")
            ),
        }
        self._ledger.selection_candidates[key] = None
        self._ledger.emit("branch_awaiting_selection", operator_id=op.operator_id)

    def _open_spawn(
        self, key: str, occurrence: Occurrence, inputs: list[Incoming]
    ) -> None:
        """Enter a spawn whose inputs are live: open its child-init scope, holding the
        values its children are entered with."""
        state = self._ledger.control_state(key)
        state.status = ControlStatus.LIVE
        state.inputs = {
            **state.inputs,
            **{
                item.port or "": item.value for item in inputs if item.value is not None
            },
        }
        opener = occurrence.activation_id or self._ledger.control_activation(
            occurrence.operator_id
        )
        scope_id = self._scope_progress.open_child_init_scope(
            opener,
            parent_scope_id=(
                occurrence.scope_id
                if occurrence.context_id or occurrence.time
                else None
            ),
        )
        self._ledger.bind_scope_occurrence(scope_id, key)

    def awaiting_fanouts(self) -> list[tuple[str, ValueRef]]:
        """Each live spawn occurrence still admitting the children its input fans
        out to, with that input; an agent's child region is filled by its agent's
        requests instead."""
        awaiting: list[tuple[str, ValueRef]] = []
        candidates = self._ledger.fanout_candidates
        for scope_id in list(candidates):
            key = self._ledger.scope_occurrence[scope_id]
            operator_id = self._ledger.occurrence(key).operator_id
            cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
            if (
                (cap is not None and cap.status is not CapabilityStatus.OPEN)
                or self._topology.kind(operator_id) is not OperatorKind.SPAWN
                or operator_id in self._topology.agent_region_spawns
                or (
                    self._ledger.control_terminal(key)
                    and self._ledger.control_states[key].status
                    is not ControlStatus.LIVE
                )
            ):
                del candidates[scope_id]
                continue
            if cap is not None and (value := self.spawn_input(key)) is not None:
                awaiting.append((key, value))
        return sorted(awaiting, key=lambda item: item[0])

    def spawn_input(self, key: str) -> ValueRef | None:
        """The value a live spawn occurrence fans out over."""
        state = self._ledger.control_states.get(key)
        if state is None or state.status is not ControlStatus.LIVE:
            return None
        return next(
            (
                value
                for port, value in sorted(state.inputs.items())
                if is_spawn_fanout_port(port)
            ),
            None,
        )

    def mark_dead(self, key: str, advance: Advance) -> None:
        """Settle an occurrence no record can reach, and resolve its routes dead.

        A leaf or agent settles without running, publishing its declared output
        empty, and a dead agent's child regions are dead with it; a dead spawn creates
        no child and its join is dead with it.
        """
        occurrence = self._ledger.occurrence(key)
        operator_id = occurrence.operator_id
        root = not occurrence.context_id and not occurrence.time
        if not self._topology.is_control(operator_id):
            wi_id = self._ledger.wi_by_occurrence.get(key)
            wi = self._ledger.work_items.get(wi_id) if wi_id else None
            if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
                return
            wi.status = WorkItemStatus.SKIPPED
            wi.outcome = PublicationOutcome.EXPLICIT_EMPTY
            wi.value_ref = ValueRef(kind="empty")
            self._ledger.emit(
                "work_item_skipped",
                work_item_id=wi.work_item_id,
                operator_id=operator_id,
            )
            self._ledger.emitter.emit_work_item(wi)
            self._ledger.emitter.emit_activation(wi.activation_id)
            self._ledger.private_state.release(wi.activation_id)
            if isinstance(
                op := self._topology.operators.get(operator_id), AgentOperator
            ):
                for ref in op.child_region_refs:
                    if (
                        join := self._topology.join_for_spawn(ref.spawn_ref)
                    ) is not None:
                        self.mark_dead(self.edges.sibling(occurrence, join), advance)
            if root:
                self._publication.publish(
                    operator_id,
                    PublicationOutcome.EXPLICIT_EMPTY,
                    ValueRef(kind="empty"),
                )
                advance.skipped.append(wi.legacy_task_id)
        else:
            state = self._ledger.control_state(key)
            if state.status is not ControlStatus.PENDING:
                return
            state.status = ControlStatus.DEAD
            self._ledger.emit("region_skipped", operator_id=operator_id)
            self._publish_control(
                key, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
            )
            if (join := self._topology.join_for_spawn(operator_id)) is not None and (
                self._topology.kind(operator_id) is OperatorKind.SPAWN
            ):
                self.mark_dead(self.edges.sibling(occurrence, join), advance)
        self.propagate(key, advance)

    def fail_control(
        self, key: str, reason: str, advance: Advance, fault: bool = False
    ) -> None:
        """Settle a control occurrence as a declared failure; a ``fault`` is one of
        the control's own, not of an input, and the first one is the instance's
        control failure.

        At the root it fails everything downstream of it; inside a region definition
        it fails the loop or child running it.
        """
        occurrence = self._ledger.occurrence(key)
        state = self._ledger.control_state(key)
        if state.status is not ControlStatus.PENDING:
            return
        state.status = ControlStatus.FAILED
        state.reason = reason
        if fault:
            self._failures.note_control_failure(reason)
        if occurrence.context_id or occurrence.time:
            self._ledger.emit(
                "region_failed",
                operator_id=occurrence.operator_id,
                detail={"reason": reason},
            )
            if self.contexts is not None:
                self.contexts.on_failure(occurrence, reason, advance)
            return
        cascade = Advance()
        visited: set[str] = set()
        self._fail_region(occurrence.operator_id, cascade, visited)
        self._failures.name_failures(cascade.failed, reason)
        advance.failed.extend(cascade.failed)
        advance.cancelled.extend(cascade.cancelled)

    def _fail_occurrence_wi(self, wi: WorkItem) -> Advance:
        """Fail a work item inside a region definition and the loop or child running
        it."""
        advance = Advance()
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return advance
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = PublicationOutcome.DECLARED_FAILURE
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        advance.failed.append(wi.legacy_task_id)
        self._failures.name_failures(
            [wi.legacy_task_id], wi.failure_reason or DECLARED_FAILURE_REASON
        )
        occurrence = self._ledger.occurrence(self._ledger.occurrence_of_work_item(wi))
        if self.contexts is not None:
            self.contexts.on_failure(
                occurrence, dependency_failed(wi.legacy_task_id), advance
            )
        return advance

    def _publish_control(
        self,
        key: str,
        outcome: PublicationOutcome,
        value: ValueRef | None,
        members: tuple[ValueMember, ...] | None = None,
    ) -> None:
        """Publish a root control occurrence's declared outputs: a singleton takes the
        value, a keyed collection one member per aggregate member, or one empty or
        failed member when the region produced no aggregate.

        ``members`` names the keyed members when they are not the value's own, as an
        early join's released qualifiers are."""
        occurrence = self._ledger.occurrence(key)
        if occurrence.context_id or occurrence.time:
            return
        self._publication.publish(occurrence.operator_id, outcome, value)
        if members is None and value is not None and value.kind == "aggregate":
            members = value.members
        if members:
            self._publication.publish_members(occurrence.operator_id, members)
        elif outcome is not PublicationOutcome.SUCCESS:
            self._publication.publish_keyed(
                occurrence.operator_id, None, outcome, value
            )

    def maybe_release_join(self, scope_id: str) -> Advance:
        scope = self._ledger.scopes.get(scope_id)
        if scope is None or scope.owner_operator_id is None:
            return Advance()
        join_op = self._topology.join_for_spawn(scope.owner_operator_id)
        if self._failures.scope_failed(scope_id):
            self._ledger.emit_scope_owner(scope_id)
            return Advance()
        # A join failed at the root blocks only the root-level scope: a nested level
        # still releases, delivering nothing.
        if (
            join_op is None
            or scope_id in self._ledger.released_scopes
            or (
                self._failures.region_failed(join_op)
                and self._ledger.root_level(scope_id)
            )
        ):
            return Advance()
        join_key = self._join_key(scope_id, join_op)
        if join_key is not None and (
            self._ledger.control_terminal(join_key) or not self._gates_open(join_key)
        ):
            return Advance()
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is None:
            return Advance()
        if (early := self._maybe_early_release(join_op, scope_id, cap)) is not None:
            return early
        if not cap.closed:
            return Advance()
        return self._release_join(join_op, scope_id)

    def _maybe_early_release(
        self, join_op: str, scope_id: str, cap: ProgressCapability
    ) -> Advance | None:
        """Release an early join once its qualifier threshold is met, per its rule.

        A monotone rule (``any``/``first_k``/a monotone predicate) releases on the first
        witness; a non-monotone predicate never releases early, waiting for closure.
        """
        join = self._topology.operators[join_op]
        if not isinstance(join, JoinRegion) or join.completion not in _EARLY_JOINS:
            return None
        threshold, monotone = self._early_rule(join)
        if not monotone or self._ledger.succeeded_children[scope_id] < threshold:
            return None
        return self._release_join(join_op, scope_id)

    def _release_join(self, join_op: str, scope_id: str) -> Advance:
        """Release a scope's join by its completion rule.

        The join occurrence the scope's spawn opened it under freezes the aggregate,
        publishes it at the root, and delivers its record downstream. A nested level of
        an agent's recursive region shares the join operator with its sibling levels,
        so its release is local to its level.
        """
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        join = self._topology.operators[join_op]
        assert isinstance(join, JoinRegion)
        outcome, value_ref = self._join_result(join, scope_id)
        children = self._ledger.scope_children_ordered(scope_id)
        if (frozen_at := self._join_key(scope_id, join_op)) is not None:
            self._freeze_region_aggregate(join, frozen_at, scope_id)
        if outcome is PublicationOutcome.DECLARED_FAILURE:
            self._scope_progress.frontier_closed(scope_id)
            cancelled = self._apply_residual_policy(join, scope_id)
            advance = self._fail_resolved_join(join_op, scope_id, children)
            advance.cancelled.extend(cancelled)
            return advance
        self._ledger.emit(
            "join_released",
            operator_id=join_op,
            detail={"outcome": outcome.value, "children": str(len(children))},
        )
        self._scope_progress.frontier_closed(scope_id)
        cancelled = self._apply_residual_policy(join, scope_id)
        if (join_key := self._join_key(scope_id, join_op)) is None:
            return Advance(cancelled=cancelled)
        value_ref = self._delivered_join_value(join_key, value_ref)
        state = self._ledger.control_state(join_key)
        state.status = ControlStatus.LIVE
        state.outputs = _port_outputs(join, value_ref or ValueRef(kind="empty"))
        self._publish_control(
            join_key, outcome, value_ref, self._released_members(join_key)
        )
        advance = Advance()
        self.propagate(join_key, advance, value=value_ref)
        advance.cancelled.extend(cancelled)
        return advance

    def _delivered_join_value(
        self, join_key: str, value_ref: ValueRef | None
    ) -> ValueRef | None:
        """The value a released join delivers downstream: its frozen aggregate's
        members in place of a full-closure result."""
        members = self._released_members(join_key)
        if value_ref is None or value_ref.kind != "join_result" or not members:
            return value_ref
        return ValueRef(kind="aggregate", members=members)

    def _released_members(self, join_key: str) -> tuple[ValueMember, ...]:
        """The members a join froze at its release, keyed by child."""
        aggregate = self._ledger.aggregate_by_join.get(join_key)
        return tuple(
            ValueMember(
                key=member.child_key,
                outcome=member.outcome,
                value_ref=member.value_ref,
            )
            for member in (aggregate.members if aggregate else ())
        )

    def _fail_resolved_join(
        self, join_op: str, scope_id: str, children: list[Activation]
    ) -> Advance:
        """Settle a join that resolved as a declared failure as a failed region.

        It delivers no record, and everything downstream of it fails as the dependent
        of its first failed child, or of the join itself when no child failed. A join
        of a scope nested under a spawned child fails only that scope, since sibling
        scopes share its operator.
        """
        failed_child = next(
            (
                wi
                for child in children
                if (
                    wi := self._ledger.work_items[
                        self._ledger.wi_by_activation[child.activation_id]
                    ]
                ).outcome
                is PublicationOutcome.DECLARED_FAILURE
            ),
            None,
        )
        join_key = self._join_key(scope_id, join_op)
        if join_key is not None and join_key != join_op:
            self._failures.mark_scope_failed(scope_id)
            advance = Advance()
            self.fail_control(
                join_key,
                f"join {join_op} resolved a failure",
                advance,
                fault=failed_child is None,
            )
            return advance
        if join_key is None:
            self._failures.mark_scope_failed(scope_id)
            self._ledger.emit(
                "region_failed", operator_id=join_op, detail={"scope": scope_id}
            )
            return Advance()
        cascade = Advance()
        self._settle_region_failed(join_op)
        self._fail_downstream(join_op, cascade, {join_op})
        if failed_child is not None and failed_child.legacy_task_id:
            self._failures.name_failures(
                cascade.failed, dependency_failed(failed_child.legacy_task_id)
            )
            cascade.failed.insert(0, failed_child.legacy_task_id)
            return cascade
        reason = f"join {join_op} resolved no winner"
        self._failures.note_control_failure(reason)
        self._failures.name_failures(cascade.failed, reason)
        return cascade

    def _freeze_region_aggregate(
        self, join: JoinRegion, join_key: str, scope_id: str
    ) -> None:
        """Capture an immutable region-join aggregate at release, ordered by child key.

        A full-closure join freezes every settled child; an early join freezes only its
        selected qualifiers. Membership is fixed here so a residual child never mutates
        the emitted aggregate and a restart replays the same members.
        """
        selected = (
            self._qualifiers(scope_id)
            if join.completion in _EARLY_JOINS
            else self._ledger.scope_children_ordered(scope_id)
        )
        members = tuple(
            RegionAggregateMember(
                child_activation_id=child.activation_id,
                child_key=str(
                    child.child_index if child.child_index is not None else 0
                ),
                source_port="out",
                outcome=self._ledger.work_items[
                    self._ledger.wi_by_activation[child.activation_id]
                ].outcome
                or PublicationOutcome.SUCCESS,
                value_ref=self._ledger.work_items[
                    self._ledger.wi_by_activation[child.activation_id]
                ].value_ref,
            )
            for child in selected
        )
        aggregate = RegionJoinAggregate(
            join_operator_id=join.operator_id,
            activation_id=self._ledger.control_activation(join.operator_id),
            members=members,
            occurrence=join_key,
        )
        self._ledger.region_aggregates.append(aggregate)
        self._ledger.aggregate_by_join[join_key] = aggregate

    def _join_result(
        self, join: JoinRegion, scope_id: str
    ) -> tuple[PublicationOutcome, ValueRef | None]:
        """A join's published outcome and value: a winner's value, or a no-winner mark.

        An early join publishes its lowest-``child_index`` qualifier's value, or its
        declared no-winner outcome; a full-closure join aggregates over its children.
        """
        if join.completion in _EARLY_JOINS:
            qualifiers = self._qualifiers(scope_id)
            threshold, _ = self._early_rule(join)
            if len(qualifiers) >= threshold:
                winner = self._ledger.work_items[
                    self._ledger.wi_by_activation[qualifiers[0].activation_id]
                ]
                return PublicationOutcome.SUCCESS, (
                    winner.value_ref or ValueRef(kind="join_result")
                )
            return self._no_winner_outcome(join), ValueRef(kind="empty")
        outcomes = [
            self._ledger.work_items[
                self._ledger.wi_by_activation[c.activation_id]
            ].outcome
            for c in self._ledger.scope_children_ordered(scope_id)
        ]
        return (
            self._join_outcome(join, [o for o in outcomes if o is not None]),
            ValueRef(kind="join_result"),
        )

    def _no_winner_outcome(self, join: JoinRegion) -> PublicationOutcome:
        return (
            PublicationOutcome.DECLARED_FAILURE
            if join.no_winner_failure
            else PublicationOutcome.EXPLICIT_EMPTY
        )

    def _join_outcome(
        self, join: JoinRegion, outcomes: list[PublicationOutcome]
    ) -> PublicationOutcome:
        if not outcomes:
            return PublicationOutcome.EXPLICIT_EMPTY
        if (
            join.completion is JoinCompletion.ALL_SUCCEED
            and PublicationOutcome.DECLARED_FAILURE in outcomes
        ):
            return PublicationOutcome.DECLARED_FAILURE
        return PublicationOutcome.SUCCESS

    def _early_rule(self, join: JoinRegion) -> tuple[int, bool]:
        """The (qualifier threshold, monotone) of an early join's release rule."""
        if join.completion is JoinCompletion.ANY:
            return 1, True
        if join.completion is JoinCompletion.FIRST_K:
            return join.first_k or 1, True
        pred = join.predicate
        return (pred.min_qualifiers if pred else 1), (pred.monotone if pred else True)

    def _qualifiers(self, scope_id: str) -> list[Activation]:
        """A scope's settled children that succeeded, ordered by ``child_index``."""
        return [
            child
            for child in self._ledger.scope_children_ordered(scope_id)
            if self._ledger.work_items[
                self._ledger.wi_by_activation[child.activation_id]
            ].outcome
            is PublicationOutcome.SUCCESS
        ]

    def _residual_policy(
        self, join: JoinRegion | None, default: ResidualPolicy
    ) -> ResidualPolicy:
        if join is not None and join.residual_policy:
            return ResidualPolicy(join.residual_policy)
        return default

    def _join_of_scope(self, scope_id: str) -> JoinRegion | None:
        join_op = self._topology.join_for_spawn(
            self._ledger.scopes[scope_id].owner_operator_id or ""
        )
        join = self._topology.operators.get(join_op) if join_op else None
        return join if isinstance(join, JoinRegion) else None

    def _cancel_residual_children(self, scope_id: str) -> list[WorkItem]:
        """Cancel a scope's unsettled children; returns their work items."""
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        cancelled: list[WorkItem] = []
        for child in self._ledger.scope_children_ordered(scope_id):
            wi = self._ledger.work_items[
                self._ledger.wi_by_activation[child.activation_id]
            ]
            if wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                self._cancel_work_item(wi)
                cancelled.append(wi)
                if cap is not None:
                    cap.outstanding = max(0, cap.outstanding - 1)
        return cancelled

    def _cancel_residual_subtrees(self, scope_id: str) -> list[str]:
        """Cancel a scope's unsettled children, each with every scope it runs or
        entered and everything under them, as a cancel does; returns the task ids
        cancelled."""
        cancelled: list[str] = []
        for wi in self._cancel_residual_children(scope_id):
            if wi.legacy_task_id:
                cancelled.append(wi.legacy_task_id)
            for owned in self._child_scopes(wi.activation_id):
                for sid in self._ledger.scope_subtree(owned):
                    cancelled.extend(self.cancel_one_scope(sid).cancelled)
        return cancelled

    def _child_scopes(self, child_activation: str) -> list[str]:
        """The scopes a child owns: the context a region-definition child runs its
        members in, or the regions an agent child entered."""
        if (context := self._ledger.child_contexts.get(child_activation)) is not None:
            return [context.scope_id]
        return [
            self._ledger.scope_by_activation[opener]
            for (activation_id, _), opener in self._ledger.region_openers.items()
            if activation_id == child_activation
            and opener in self._ledger.scope_by_activation
        ]

    def _apply_residual_policy(self, join: JoinRegion, scope_id: str) -> list[str]:
        """Govern a released early join's not-yet-settled materialized children;
        returns the ones it cancelled.

        ``continue`` leaves child-init open; ``drain`` seals it; ``cancel`` revokes it
        and cancels the residual children. Residual settlements stay ledger-visible but
        never become the join's implicit winner output.
        """
        if join.completion not in _EARLY_JOINS:
            return []
        policy = self._residual_policy(join, ResidualPolicy.CONTINUE)
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if policy is ResidualPolicy.CONTINUE or cap is None:
            return []
        cancel = policy is ResidualPolicy.CANCEL
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED if cancel else CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_revoked" if cancel else "child_init_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id, "residual": policy.value},
            )
        return self._cancel_residual_subtrees(scope_id) if cancel else []

    def open_roots(self) -> Advance:
        """Admit each root occurrence whose inputs are already resolved."""
        advance = Advance()
        for wi in list(self._ledger.work_items.values()):
            cont = self._ledger.continuations.get(wi.work_item_id)
            if (
                cont is not None
                and not cont.waiting_on
                and wi.legacy_task_id
                and wi.operator_id not in self._topology.definition_of
                and not self._ledger.is_dynamic_activation(wi.activation_id)
            ):
                self.evaluate(self._ledger.occurrence_of_work_item(wi), advance)
        for op_id in self._topology.operators:
            # A child-template region opens only when its parent spawn materializes it,
            # an agent-selected region only when the agent requests it, and a region
            # definition's member only when its loop or child enters it, so none fires
            # at the root.
            if (
                not self._topology.is_control(op_id)
                or op_id in self._topology.child_templates
                or op_id in self._topology.agent_region_spawns
                or op_id in self._topology.definition_of
            ):
                continue
            cont = self._ledger.continuations.get(control_key(op_id))
            if cont is not None and not cont.waiting_on:
                self.evaluate(op_id, advance)
        return advance

    def admit(self, work_item_id: str, advance: Advance) -> None:
        """Move a work item whose inputs resolved to READY, gating on authority.

        A declared-input agent is admissible only once its accepted-input manifest is
        satisfied — every required port carries a recorded accepted input — so a fan-out
        child or a downstream merge agent never dispatches before its input is durable.
        A leaf's external effect is a fixed template contract, decided against the root
        grant wherever its occurrence runs; a work item inside a region definition is
        materialized as it becomes ready.
        """
        wi = self._ledger.work_items[work_item_id]
        if wi.status is not WorkItemStatus.BLOCKED:
            return
        cont = self._ledger.continuations.get(work_item_id)
        if cont is not None and cont.required_ports:
            have = {
                a.target_port
                for a in self._inputs.accepted_inputs_for(wi.activation_id)
            }
            if not cont.required_ports <= have:
                return
        interface = self._topology.requested_interface(wi.operator_id)
        grant_id = self._ledger.root_grant.grant_id
        if interface is not None and interface not in self._ledger.root_grant.invoke:
            reason = f"interface {interface!r} outside root grant invoke face"
            self._authority.record_decision(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.DENIED,
                    denial_kind=DenialKind.AUTHORITY,
                    reason=reason,
                )
            )
            wi.failure_reason = f"authority denied: {reason}"
            self._ledger.emit(
                "authority_denied",
                work_item_id=work_item_id,
                operator_id=wi.operator_id,
            )
            advance.extend(self.settle_failed_wi(wi))
            return
        if interface is not None:
            self._authority.record_decision(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.GRANTED,
                )
            )
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "work_item_ready", work_item_id=work_item_id, operator_id=wi.operator_id
        )
        advance.ready.append(wi.legacy_task_id)

    def _settle_failure(self, work_item_id: str) -> Advance:
        """Settle a work item as a declared failure and fail everything downstream of
        it."""
        cascade = Advance()
        self._fail_work_item(work_item_id, cascade, set())
        self._declare_failures(self._ledger.work_items[work_item_id], cascade.failed)
        return cascade

    def _declare_failures(self, primary: WorkItem, failed: list[str]) -> None:
        """Name why a failed work item and each task its failure cascaded into failed:
        the work item by its own reason, the rest as its dependents. A task an earlier
        failure named keeps that reason."""
        if not failed:
            return
        self._failures.name_failures(
            [primary.legacy_task_id],
            primary.failure_reason or DECLARED_FAILURE_REASON,
        )
        self._failures.name_failures(failed, dependency_failed(primary.legacy_task_id))

    def _fail_work_item(
        self, work_item_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        wi = self._ledger.work_items[work_item_id]
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = PublicationOutcome.DECLARED_FAILURE
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        self._publication.publish(
            wi.operator_id, PublicationOutcome.DECLARED_FAILURE, None
        )
        cascade.failed.append(wi.legacy_task_id)
        self._fail_agent_regions(wi, cascade, visited)
        self._fail_downstream(wi.operator_id, cascade, visited)

    def _fail_agent_regions(
        self, wi: WorkItem, cascade: Advance, visited: set[str]
    ) -> None:
        """Fail every child region a failed agent declares.

        A region the agent never entered fails as a spawn whose input failed does, and
        opens no scope. An entered region's children follow its residual policy, and
        its join fails unless it already released. A spawned agent instance shares its
        regions' operators with its siblings, so only its own scopes fail.
        """
        op = self._topology.operators.get(wi.operator_id)
        if not isinstance(op, AgentOperator):
            return
        instance = self._ledger.activations[wi.activation_id].kind == "child"
        for ref in op.child_region_refs:
            opener = self._ledger.region_openers.get((wi.activation_id, ref.spawn_ref))
            scope_id = self._ledger.scope_by_activation.get(opener) if opener else None
            if scope_id is None:
                if not instance:
                    self._fail_region(ref.spawn_ref, cascade, visited)
                continue
            if (
                scope_id not in self._ledger.released_scopes
                and not self._failures.scope_failed(scope_id)
            ):
                self._fail_entered_region(
                    ref.spawn_ref, scope_id, instance, cascade, visited
                )
            self.close_owned_region(scope_id, cascade)
            self._ledger.emit_scope_owner(scope_id)

    def _fail_entered_region(
        self,
        spawn_op: str,
        scope_id: str,
        instance: bool,
        cascade: Advance,
        visited: set[str],
    ) -> None:
        """Fail the join of one scope a failed agent opened, which has not released.

        The scope is named directly: a recursive agent's levels share the join
        operator, and another level's scope may already have released.
        """
        self._failures.mark_scope_failed(scope_id)
        join_op = self._topology.join_for_spawn(spawn_op)
        if instance or join_op is None:
            self._ledger.emit(
                "region_failed", operator_id=join_op, detail={"scope": scope_id}
            )
        elif join_op not in visited and not self._failures.region_failed(join_op):
            visited.add(join_op)
            self._settle_region_failed(join_op)
            self._fail_downstream(join_op, cascade, visited)

    def _fail_downstream(
        self, operator_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            if self._topology.is_control(successor):
                self._fail_region(successor, cascade, visited)
            elif succ_wi := self._ledger.wi_by_occurrence.get(successor):
                self._fail_work_item(succ_wi, cascade, visited)

    def _fail_region(
        self, operator_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        """Settle a control operator a failed input feeds as a declared failure.

        The region never fires. A failed spawn opens no scope and creates no child: it
        fails its child template and its join, and each collection it publishes holds
        one failed member. A join fails whatever its completion rule, unless its scope
        already released.
        """
        if operator_id in visited or self._failures.region_failed(operator_id):
            return
        visited.add(operator_id)
        kind = self._topology.kind(operator_id)
        if kind is OperatorKind.JOIN and self._ledger.region_closed(operator_id):
            return
        self._settle_region_failed(operator_id)
        if kind is OperatorKind.SPAWN:
            self._fail_spawn_template(operator_id)
            if (join_op := self._topology.join_for_spawn(operator_id)) is not None:
                self._fail_region(join_op, cascade, visited)
        self._fail_downstream(operator_id, cascade, visited)

    def _settle_region_failed(self, operator_id: str) -> None:
        self._failures.mark_region_failed(operator_id)
        if (state := self._ledger.control_states.get(operator_id)) is not None:
            state.status = ControlStatus.FAILED
        self._ledger.emit("region_failed", operator_id=operator_id)
        self._publication.publish(
            operator_id, PublicationOutcome.DECLARED_FAILURE, None
        )
        self._publication.publish_keyed(
            operator_id, None, PublicationOutcome.DECLARED_FAILURE, None
        )

    def _fail_spawn_template(self, spawn_op: str) -> None:
        """Publish a failed spawn's child template, and the templates nested under it,
        as failed once no live spawn instantiates them; a template runs as no task of
        its own."""
        template = self._topology.child_template_of(spawn_op)
        if template is None:
            return
        for failed in self._ledger.template_closure(
            template, self._instantiable_by_live
        ):
            self._publication.publish(failed, PublicationOutcome.DECLARED_FAILURE, None)

    def _instantiable_by_live(self, template: str, dead: list[str]) -> bool:
        """Whether a spawn that neither failed nor belongs to a dead template can still
        instantiate ``template``."""
        return any(
            isinstance(op, SpawnRegion)
            and op.child_template_ref == template
            and not self._failures.region_failed(op.operator_id)
            and self._topology.region_owner(op.operator_id) not in dead
            for op in self._topology.operators.values()
        )

    def reconcile_failure(self, task_id: str) -> list[str]:
        """Fail what a task's settled failure left standing downstream of it."""
        wi = self._ledger.work_item_for_task(task_id)
        if (
            wi is None
            or wi.outcome is not PublicationOutcome.DECLARED_FAILURE
            or self._ledger.is_dynamic_activation(wi.activation_id)
        ):
            return []
        cascade = Advance()
        visited: set[str] = set()
        self._fail_agent_regions(wi, cascade, visited)
        self._fail_downstream(wi.operator_id, cascade, visited)
        self._failures.name_failures(cascade.failed, dependency_failed(task_id))
        return cascade.failed

    def fail_undeliverable_region_inputs(self) -> list[str]:
        """Fail each task reading a region of an agent that runs only as a spawned
        child, and everything downstream of it."""
        cascade = Advance()
        visited: set[str] = set()
        owners = spawned_only_region_owners(self._topology.operators.values())
        for spawn_op, owner in sorted(owners.items()):
            join_op = self._topology.join_for_spawn(spawn_op)
            for successor in sorted(self._topology.forward.get(join_op or "", ())):
                if (wi_id := self._ledger.wi_by_occurrence.get(successor)) is None:
                    continue
                start = len(cascade.failed)
                self._fail_work_item(wi_id, cascade, visited)
                if failed := cascade.failed[start:]:
                    self._failures.name_failures(
                        failed[:1],
                        f"region of spawned-only agent {owner} delivers nothing",
                    )
                    self._failures.name_failures(
                        failed[1:], dependency_failed(failed[0])
                    )
        return cascade.failed
