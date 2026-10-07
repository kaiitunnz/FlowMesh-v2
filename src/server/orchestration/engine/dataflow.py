"""Record flow, admission, join release and failure propagation through regions."""

from ...task.v2.representations.operators import (
    AgentOperator,
    BranchRegion,
    JoinCompletion,
    JoinRegion,
    OperatorKind,
    ResidualPolicy,
    SpawnRegion,
    spawned_only_region_owners,
)
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    Activation,
    AuthorityDecision,
    AuthorityDecisionKind,
    CapabilityStatus,
    DenialKind,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    Record,
    RegionAggregateMember,
    RegionJoinAggregate,
    ValueRef,
    WorkItem,
    WorkItemStatus,
)
from .advance import Advance, RegionError, dependency_failed
from .authority import AuthorityLedger
from .failures import DECLARED_FAILURE_REASON, FailureLedger
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger, control_key
from .publications import PublicationLedger
from .scopes import ScopeProgress
from .topology import CHILD_INIT_OPENERS, PlanTopology

_EARLY_JOINS = frozenset(
    {JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE}
)


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

    def settle_failed_wi(self, wi: WorkItem) -> Advance:
        """Settle a work item as a declared failure: a child drains its scope, anything
        else cascades over its static successors. A failed agent fails the regions it
        declares either way."""
        activation = self._ledger.activations[wi.activation_id]
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

    def route_branch(self, branch_op: str, selected_port: str) -> Advance:
        """Route a branch record to the selected port; settle the other ports empty."""
        if self._topology.kind(branch_op) is not OperatorKind.BRANCH:
            raise RegionError(f"{branch_op!r} is not a branch region")
        advance = Advance()
        skipped: set[str] = set()
        for successor in sorted(self._topology.forward.get(branch_op, ())):
            from_port = self._topology.edge_from_port(branch_op, successor)
            if from_port in (selected_port, None):
                self._release_one(successor, branch_op, advance)
            else:
                self._settle_empty_successor(successor, skipped)
        self._ledger.emit(
            "branch_routed", operator_id=branch_op, detail={"port": selected_port}
        )
        return advance

    def cancel_one_scope(self, scope_id: str) -> Advance:
        self._ledger.emit("scope_cancelled", detail={"scope": scope_id})
        self._scope_progress.revoke_progress(scope_id)
        scope = self._ledger.scopes[scope_id]
        cancelled = self._apply_cancellation_residual(scope_id)
        for wi in self._ledger.scope_work_items(scope_id, kinds=("leaf", "agent")):
            if wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                self._cancel_work_item(wi)
                cancelled.append(wi.legacy_task_id)
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
                wi.legacy_task_id for wi in self._cancel_residual_children(scope_id)
            ]
        return []

    def _resolve_cancelled_outputs(self, scope_id: str) -> Advance:
        """Release a cancelled scope's join as its cancellation outcome.

        Only a root-level scope publishes it and delivers its record.
        """
        if scope_id in self._ledger.released_scopes:
            return Advance()
        owner_op = self._ledger.scopes[scope_id].owner_operator_id or ""
        join = self._join_of_scope(scope_id)
        if join is not None:
            outcome = self._no_winner_outcome(join)
            release_op = join.operator_id
        elif self._topology.kind(owner_op) is OperatorKind.LOOP_CONTEXT:
            outcome = PublicationOutcome.EXPLICIT_EMPTY
            release_op = owner_op
        else:
            return Advance()
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        self._ledger.emit(
            "join_released" if join is not None else "loop_egress",
            operator_id=release_op,
            detail={"outcome": outcome.value, "cancelled": "true"},
        )
        self._scope_progress.frontier_closed(scope_id)
        if not self._ledger.root_level(scope_id):
            return Advance()
        self._publication.publish(release_op, outcome, ValueRef(kind="empty"))
        return self.deliver_record(
            release_op,
            self._ledger.control_activation(release_op),
            ValueRef(kind="empty"),
        )

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

    def deliver_record(
        self, operator_id: str, activation_id: str, value_ref: ValueRef | None
    ) -> Advance:
        self._ledger.records.append(
            Record(
                operator_id=operator_id,
                activation_id=activation_id,
                scope_id=self._ledger.root_scope.scope_id,
                value_ref=value_ref,
            )
        )
        self._ledger.emit("record_delivered", operator_id=operator_id)
        advance = Advance()
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            self._release_one(successor, operator_id, advance)
        return advance

    def _release_one(self, successor: str, from_op: str, advance: Advance) -> None:
        """Deliver a record to one successor: fire a control op, or admit a leaf."""
        if self._topology.is_control(successor):
            cont = self._ledger.continuations.get(control_key(successor))
            if cont is None or self._failures.region_failed(successor):
                return
            cont.waiting_on.discard(from_op)
            if not cont.waiting_on:
                self._fire_control(successor, advance)
            return
        wi_id = self._ledger.wi_by_operator.get(successor)
        cont = self._ledger.continuations.get(wi_id) if wi_id else None
        if cont is None:
            return
        cont.waiting_on.discard(from_op)
        if not cont.waiting_on:
            self.admit(cont.work_item_id, advance)

    def _fire_control(self, operator_id: str, advance: Advance) -> None:
        kind = self._topology.kind(operator_id)
        if kind in CHILD_INIT_OPENERS:
            self._scope_progress.open_child_init_scope(
                self._ledger.control_activation(operator_id)
            )
        elif kind is OperatorKind.LOOP_CONTEXT:
            self._scope_progress.open_loop(self._ledger.control_activation(operator_id))
        elif kind is OperatorKind.MERGE:
            self._ledger.emit("merge_combined", operator_id=operator_id)
            advance.extend(
                self.deliver_record(
                    operator_id, self._ledger.control_activation(operator_id), None
                )
            )
        elif kind is OperatorKind.BRANCH:
            branch = self._topology.operators[operator_id]
            selection = branch.selection if isinstance(branch, BranchRegion) else None
            if selection and any(
                self._topology.edge_from_port(operator_id, s) == selection
                for s in self._topology.forward.get(operator_id, ())
            ):
                advance.extend(self.route_branch(operator_id, selection))
        # JOIN is released by scope closure, never by an input record.

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
        if not monotone or len(self._qualifiers(scope_id)) < threshold:
            return None
        return self._release_join(join_op, scope_id)

    def _release_join(self, join_op: str, scope_id: str) -> Advance:
        """Release a scope's join by its completion rule.

        Only a root-level scope freezes the join's aggregate, publishes it, and
        delivers its record downstream. A scope nested under a spawned child shares the
        join operator with its sibling levels, so its release is local to its level.
        """
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        join = self._topology.operators[join_op]
        assert isinstance(join, JoinRegion)
        outcome, value_ref = self._join_result(join, scope_id)
        children = self._materialized_children(scope_id)
        nested = not self._ledger.root_level(scope_id)
        if not nested:
            self._freeze_region_aggregate(join, join_op, scope_id)
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
        if nested:
            return Advance(cancelled=cancelled)
        self._publication.publish(join_op, outcome, value_ref)
        advance = self.deliver_record(
            join_op, self._ledger.control_activation(join_op), value_ref
        )
        advance.cancelled.extend(cancelled)
        return advance

    def _fail_resolved_join(
        self, join_op: str, scope_id: str, children: list[Activation]
    ) -> Advance:
        """Settle a join that resolved as a declared failure as a failed region.

        It delivers no record, and everything downstream of it fails as the dependent
        of its first failed child, or of the join itself when no child failed. A join
        of a scope nested under a spawned child fails only that scope, since sibling
        scopes share its operator.
        """
        if not self._ledger.root_level(scope_id):
            self._failures.mark_scope_failed(scope_id)
            self._ledger.emit(
                "region_failed", operator_id=join_op, detail={"scope": scope_id}
            )
            return Advance()
        cascade = Advance()
        self._settle_region_failed(join_op)
        self._fail_downstream(join_op, cascade, {join_op})
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
        if failed_child is not None and failed_child.legacy_task_id:
            self._failures.name_failures(
                cascade.failed, dependency_failed(failed_child.legacy_task_id)
            )
            cascade.failed.insert(0, failed_child.legacy_task_id)
            return cascade
        self._failures.name_failures(
            cascade.failed, f"join {join_op} resolved no winner"
        )
        return cascade

    def _freeze_region_aggregate(
        self, join: JoinRegion, join_op: str, scope_id: str
    ) -> None:
        """Capture an immutable region-join aggregate at release, ordered by child key.

        A full-closure join freezes every settled child; an early join freezes only its
        selected qualifiers. Membership is fixed here so a residual child never mutates
        the emitted aggregate and a restart replays the same members.
        """
        selected = (
            self._qualifiers(scope_id)
            if join.completion in _EARLY_JOINS
            else self._materialized_children(scope_id)
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
            join_operator_id=join_op,
            activation_id=self._ledger.control_activation(join_op),
            members=members,
        )
        self._ledger.region_aggregates.append(aggregate)
        self._ledger.aggregate_by_join[join_op] = aggregate

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
            for c in self._materialized_children(scope_id)
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
            for child in self._materialized_children(scope_id)
            if self._ledger.work_items[
                self._ledger.wi_by_activation[child.activation_id]
            ].outcome
            is PublicationOutcome.SUCCESS
        ]

    def _materialized_children(self, scope_id: str) -> list[Activation]:
        return sorted(
            (
                a
                for a in self._ledger.activations.values()
                if a.scope_id == scope_id and a.kind == "child"
            ),
            key=lambda a: a.child_index if a.child_index is not None else 0,
        )

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
        for child in self._materialized_children(scope_id):
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
        """Cancel a scope's unsettled children, each with the regions it entered and
        everything under them, as a cancel does; returns the task ids cancelled."""
        cancelled: list[str] = []
        for wi in self._cancel_residual_children(scope_id):
            cancelled.append(wi.legacy_task_id)
            for opener in self._entered_region_openers(wi.activation_id):
                for sid in self._ledger.scope_subtree(
                    self._ledger.scope_by_activation[opener]
                ):
                    cancelled.extend(self.cancel_one_scope(sid).cancelled)
        return cancelled

    def _entered_region_openers(self, agent_activation: str) -> list[str]:
        return [
            opener
            for (activation_id, _), opener in self._ledger.region_openers.items()
            if activation_id == agent_activation
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
        advance = Advance()
        for wi in list(self._ledger.work_items.values()):
            cont = self._ledger.continuations.get(wi.work_item_id)
            if cont is not None and not cont.waiting_on and wi.legacy_task_id:
                self.admit(wi.work_item_id, advance)
        for op_id in self._topology.operators:
            # A child-template region opens only when its parent spawn materializes it,
            # and an agent-selected region only when the agent requests it, so neither
            # fires at the root.
            if (
                not self._topology.is_control(op_id)
                or op_id in self._topology.child_templates
                or op_id in self._topology.agent_region_spawns
            ):
                continue
            cont = self._ledger.continuations.get(control_key(op_id))
            if cont is not None and not cont.waiting_on:
                self._fire_control(op_id, advance)
        return advance

    def admit(self, work_item_id: str, advance: Advance) -> None:
        """Move a work item whose predecessors settled to READY, gating on authority.

        A declared-input agent is admissible only once its accepted-input manifest is
        satisfied — every required port carries a recorded accepted input — so a fan-out
        child or a downstream merge agent never dispatches before its input is durable.
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
        if interface is not None and interface not in self._ledger.root_grant.invoke:
            reason = f"interface {interface!r} outside root grant invoke face"
            self._authority.record_decision(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=self._ledger.root_grant.grant_id,
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
                    grant_id=self._ledger.root_grant.grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.GRANTED,
                )
            )
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "work_item_ready", work_item_id=work_item_id, operator_id=wi.operator_id
        )
        advance.ready.append(wi.legacy_task_id)

    def _settle_empty_successor(
        self, operator_id: str, visited: set[str] | None = None
    ) -> None:
        """Skip a non-selected branch successor and its whole subtree.

        A leaf resolves empty; a control successor clears its pending inputs and is
        skipped so a downstream join never waits on an untaken path. A join itself is
        left to its own scope closure. Two non-selected ports can share a downstream
        operator, so the walk is idempotent per operator.
        """
        visited = visited if visited is not None else set()
        if operator_id in visited:
            return
        visited.add(operator_id)
        if self._topology.is_control(operator_id):
            if self._topology.kind(operator_id) is OperatorKind.JOIN:
                return
            if (
                cont := self._ledger.continuations.get(control_key(operator_id))
            ) is not None:
                cont.waiting_on.clear()
            self._ledger.emit("region_skipped", operator_id=operator_id)
        else:
            wi_id = self._ledger.wi_by_operator.get(operator_id)
            if wi_id is None:
                return
            wi = self._ledger.work_items[wi_id]
            if wi.status in TERMINAL_WORK_ITEM_STATUSES:
                return
            wi.status = WorkItemStatus.SETTLED
            wi.outcome = PublicationOutcome.EXPLICIT_EMPTY
            self._ledger.emitter.emit_work_item(wi)
            self._ledger.emitter.emit_activation(wi.activation_id)
            self._ledger.private_state.release(wi.activation_id)
            self._publication.publish(
                operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
            )
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            self._settle_empty_successor(successor, visited)

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
            elif succ_wi := self._ledger.wi_by_operator.get(successor):
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
            self._fail_spawn_template(operator_id, cascade)
            self._publication.publish_keyed(
                operator_id, None, PublicationOutcome.DECLARED_FAILURE, None
            )
            if (join_op := self._topology.join_for_spawn(operator_id)) is not None:
                self._fail_region(join_op, cascade, visited)
        self._fail_downstream(operator_id, cascade, visited)

    def _settle_region_failed(self, operator_id: str) -> None:
        self._failures.mark_region_failed(operator_id)
        self._ledger.emit("region_failed", operator_id=operator_id)
        self._publication.publish(
            operator_id, PublicationOutcome.DECLARED_FAILURE, None
        )

    def _fail_spawn_template(self, spawn_op: str, cascade: Advance) -> None:
        """Fail a failed spawn's child template, and the templates nested under it,
        once no live spawn instantiates them."""
        template = self._topology.child_template_of(spawn_op)
        if template is None:
            return
        for failed in self._ledger.template_closure(
            template, self._instantiable_by_live
        ):
            self._publication.publish(failed, PublicationOutcome.DECLARED_FAILURE, None)
            cascade.failed.append(failed)

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
                if (wi_id := self._ledger.wi_by_operator.get(successor)) is None:
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
