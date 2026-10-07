"""Boundaries an episode yields."""

from collections.abc import Iterable

from shared.outcome import OutcomeManifest
from shared.tools.contract import MediatedOperationPermit
from shared.utils import new_idempotency_key, new_invocation_id, new_mediated_permit_id

from ...task.v2.representations.operators import (
    AgentOperator,
    BoundaryEventKind,
    LogicalOperator,
)
from ...utils.time import now_iso
from ..guardrails import ScopeBudget
from ..outcomes import is_compensable, is_replayable, next_on_terminal
from ..state import (
    TERMINAL_INVOCATION_STATES,
    TERMINAL_WORK_ITEM_STATUSES,
    AttemptStatus,
    AuthorityDecision,
    AuthorityDecisionKind,
    BoundaryEvent,
    DenialKind,
    Invocation,
    InvocationState,
    ValueRef,
    WorkItem,
    WorkItemStatus,
)
from ..tool_dispatch import MODEL_INTERFACE, FacadeTurnGroup, ToolOutcomeStatus
from .advance import Advance, RegionError
from .authority import AuthorityLedger
from .boundaries import BoundaryLedger
from .ledger import OrchestrationLedger
from .spawns import SpawnRegions
from .topology import PlanTopology

# Boundary kinds whose exactly-once rests on the durable correlation key, so a mediated
# one must carry a call correlation or it could duplicate a target effect on re-drive.
_DEDUP_CAPABLE = frozenset(
    {
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.EXTERNAL_EFFECT,
    }
)


class EpisodeBoundaryRouter:
    """Routes each boundary an episode yields into the ledger, validating an agent's
    against its signature and authority, suspends the episode on it and settles its
    outcome."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        authority: AuthorityLedger,
        boundaries: BoundaryLedger,
        spawns: SpawnRegions,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._authority = authority
        self._boundaries = boundaries
        self._spawns = spawns
        self._budget = budget

    def route_boundary_event(self, task_id: str, event: BoundaryEvent) -> Advance:
        """Route an episode's boundary request back into the ledger, validated first."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        # A recorded call is a re-drive whether it was granted or denied: it maps to its
        # key and creates no new work, so re-validation and duplicate records are cut.
        if self._is_boundary_redrive(wi, event):
            return Advance()
        op = self._topology.operators.get(wi.operator_id)
        # An agent boundary is authority-checked against the operator's declared face; a
        # leaf boundary carries no authority ceiling and defers its invocation directly.
        if isinstance(op, AgentOperator):
            if (denial := self._validate_agent_boundary(op, wi, event)) is not None:
                return self._record_boundary_denial(wi, event, denial)
            if event.kind in _DEDUP_CAPABLE and event.call_correlation is None:
                raise RegionError(
                    f"mediated {event.kind.value} boundary requires a call correlation "
                    "for durable dedup"
                )
        match event.kind:
            case BoundaryEventKind.SPAWN:
                opener = self._agent_spawn_opener(op, wi, event)
                # Record only after the child materializes: a budget/depth rejection
                # raises and must leave no phantom-accepted envelope to re-drive. The
                # region selects the entry body; a raw child_ref never names a body. The
                # spawn payload becomes the child's inline child-init input.
                child_input = event.value_ref or (
                    ValueRef(kind="inline", literal=event.request_payload)
                    if event.request_payload is not None
                    else None
                )
                advance = self._spawns.materialize_child(opener, value_ref=child_input)
                self._record_boundary(wi, event)
                return advance
            case BoundaryEventKind.SPAWN_SEAL:
                opener = self._agent_spawn_opener(op, wi, event)
                advance = self._spawns.seal_spawn(opener)
                self._record_boundary(wi, event)
                return advance
            case BoundaryEventKind.INVOCATION:
                return self._suspend_on_request(wi, event, effect=False)
            case BoundaryEventKind.EXTERNAL_EFFECT:
                return self._suspend_on_request(wi, event, effect=True)
            case BoundaryEventKind.YIELD:
                wi.continuation_ref = event.continuation
                self._record_boundary(wi, event)
                self._suspend_work_item(wi, "episode_yielded")
                return Advance()
            case BoundaryEventKind.STATE_ACCESS:
                self._ledger.emit(
                    "state_access",
                    work_item_id=wi.work_item_id,
                    operator_id=wi.operator_id,
                    detail={"state_ref": event.state_ref or ""},
                )
                return Advance()
        return Advance()

    def route_facade_turn_group(self, task_id: str, group: FacadeTurnGroup) -> Advance:
        """Record a model turn's facade group and route each member
        kind-specifically."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        op = self._topology.operators.get(wi.operator_id)
        advance = Advance()
        per_region: dict[str, int] = {}
        for member in group.members:
            if (
                wi.activation_id,
                member.call_correlation,
            ) in self._boundaries.boundary_events:
                continue  # a re-driven group reuses its recorded members
            event = self._boundaries.facade_member_event(group.group_id, member)
            if member.kind is BoundaryEventKind.SPAWN:
                self._route_group_spawn(op, wi, event, per_region, advance)
            else:
                self._route_group_invocation(op, wi, event)
        members = self._boundaries.group_members(wi.activation_id, group.group_id)
        if self._boundaries.group_awaits_unresolved(members):
            self._suspend_work_item(wi, "episode_group_suspended")
            return advance
        # No await-outcome member is pending: the group settled at admission. Stage its
        # ordered acceptance vector for the next step; the runtime re-enqueues the lane
        # at once (closing this turn's attempt) rather than holding a worker.
        wi.pending_outcome_group = group.group_id
        wi.pending_outcome_call = None
        return advance

    def _route_group_invocation(
        self, op: LogicalOperator | None, wi: WorkItem, event: BoundaryEvent
    ) -> None:
        # Only an agent operator originates a facade; a non-agent operator is a fabric
        # misconfiguration, denied fail-closed rather than minting an unvalidated call.
        denial = (
            self._validate_agent_boundary(op, wi, event)
            if isinstance(op, AgentOperator)
            else DenialKind.AUTHORITY
        )
        if denial is not None:
            self._record_boundary(wi, event, denial=denial)
            return
        invocation = Invocation(
            invocation_id=new_invocation_id(),
            work_item_id=wi.work_item_id,
            state=InvocationState.ISSUED,
            replayable=True,
        )
        self._ledger.invocations[invocation.invocation_id] = invocation
        self._record_boundary(wi, event, invocation_id=invocation.invocation_id)

    def _route_group_spawn(
        self,
        op: LogicalOperator | None,
        wi: WorkItem,
        event: BoundaryEvent,
        per_region: dict[str, int],
        advance: Advance,
    ) -> None:
        if not isinstance(op, AgentOperator):
            self._record_boundary(wi, event, denial=DenialKind.AUTHORITY)
            return
        if (denial := self._validate_agent_boundary(op, wi, event)) is not None:
            self._record_boundary(wi, event, denial=denial)
            return
        region = event.child_region_ref or ""
        # This turn's admitted spawns already register in the child-init scope, so the
        # region count reflects them; per_region only sums the per-turn budget across
        # regions and must not be re-added here (that would trip the region cap at 2x).
        turn_total = sum(per_region.values())
        region_total = self._region_child_count(wi.activation_id, region)
        if (
            turn_total >= self._budget.max_spawns_per_turn
            or region_total >= self._budget.max_spawns_per_region
        ):
            # A budget overflow is a typed quota outcome, never a denial and never a
            # sibling-poisoning failure: the member acks quota and creates no child.
            self._record_boundary(wi, event)
            self._boundaries.set_member_outcome(
                wi,
                event.call_correlation,
                f"{ToolOutcomeStatus.QUOTA.value}: the {region!r} spawn budget was "
                "reached this turn; no child was created",
            )
            return
        child_input = (
            ValueRef(kind="inline", literal=event.request_payload)
            if event.request_payload is not None
            else None
        )
        try:
            opener = self._agent_spawn_opener(op, wi, event)
            child_advance = self._spawns.materialize_child(
                opener, value_ref=child_input
            )
        except RegionError as exc:
            self._record_boundary(wi, event)
            self._boundaries.set_member_outcome(
                wi,
                event.call_correlation,
                f"{ToolOutcomeStatus.UNAVAILABLE.value}: the {region!r} region cannot "
                f"accept a child ({exc})",
            )
            return
        advance.extend(child_advance)
        self._record_boundary(wi, event)
        self._boundaries.set_member_outcome(
            wi,
            event.call_correlation,
            f"{ToolOutcomeStatus.SUCCESS.value}: spawned a {region!r} reviewer child",
        )
        per_region[region] = per_region.get(region, 0) + 1

    def _region_child_count(self, agent_activation: str, region: str) -> int:
        """The children already materialized under an agent's named spawn region."""
        act = self._ledger.activations.get(agent_activation)
        op = self._topology.operators.get(act.operator_id) if act else None
        if not isinstance(op, AgentOperator):
            return 0
        region_op = self._topology.agent_region_op(op, region)
        if region_op is None:
            return 0
        opener = self._ledger.region_openers.get((agent_activation, region_op))
        if opener is None or (scope_id := self._ledger.scope_id_for(opener)) is None:
            return 0
        return self._ledger.scope_children[scope_id]

    def deliver_boundary_outcome(self, task_id: str, call_correlation: str) -> Advance:
        """Re-ready a boundary-suspended work item once its outcome is durable."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return Advance()
        if (wi.activation_id, call_correlation) not in self._boundaries.boundary_events:
            return Advance()
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "episode_resumed",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"call": call_correlation},
        )
        return Advance(ready=[wi.legacy_task_id])

    def mark_pending_outcome(self, task_id: str, call_correlation: str | None) -> None:
        """Record (or clear) the settled boundary whose outcome the next resume
        injects."""
        if (wi := self._ledger.work_item_for_task(task_id)) is not None:
            wi.pending_outcome_call = call_correlation
            if call_correlation is None:
                wi.pending_outcome_group = None

    def mint_operation_permit(
        self,
        task_id: str,
        call_correlation: str,
        *,
        target_id: str,
        target_generation: int,
        max_results: int,
        timeout_sec: float,
        result_char_cap: int,
        deadline_epoch: float,
        credential: str | None = None,
        deployment_credential: bool = False,
    ) -> MediatedOperationPermit | None:
        """A one-use permit for a recorded worker-originated boundary, or None."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self._boundaries.boundary_events.get((wi.activation_id, call_correlation))
        if env is None or env.invocation_id is None or env.request_digest is None:
            return None
        interface = env.interface or ""
        epoch = 0
        if (act := self._ledger.activations.get(wi.activation_id)) is not None:
            epoch = self._authority.grant_for_scope(act.scope_id).epoch
        return MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=wi.legacy_task_id,
            call_correlation=call_correlation,
            interface=interface,
            subject=interface,
            invocation_id=env.invocation_id,
            idempotency_key=env.idempotency_key,
            request_digest=env.request_digest,
            target_id=target_id,
            target_generation=target_generation,
            policy_epoch=epoch,
            deadline_epoch=deadline_epoch,
            max_results=max_results,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
            credential=credential,
            deployment_credential=deployment_credential,
        )

    def authorize_model_turn(
        self,
        task_id: str,
        call_correlation: str,
        request_digest: str,
        *,
        target_id: str,
        target_generation: int,
        timeout_sec: float,
        result_char_cap: int,
        deadline_epoch: float,
        credential: str | None = None,
        deployment_credential: bool = False,
    ) -> MediatedOperationPermit | None:
        """A one-use permit for a held agent's in-turn model egress, or None on
        denial."""
        wi = self._ledger.work_item_for_task(task_id)
        act = self._ledger.activations.get(wi.activation_id) if wi is not None else None
        op = self._topology.operators.get(act.operator_id) if act is not None else None
        if wi is None or act is None or not isinstance(op, AgentOperator):
            return None
        invoke, _ = self._authority.agent_faces(op, wi)
        if MODEL_INTERFACE not in invoke:
            return None
        return MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=wi.legacy_task_id,
            call_correlation=call_correlation,
            interface=MODEL_INTERFACE,
            subject=MODEL_INTERFACE,
            invocation_id=new_invocation_id(),
            idempotency_key=new_idempotency_key(),
            request_digest=request_digest,
            target_id=target_id,
            target_generation=target_generation,
            policy_epoch=self._authority.grant_for_scope(act.scope_id).epoch,
            deadline_epoch=deadline_epoch,
            max_results=1,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
            credential=credential,
            deployment_credential=deployment_credential,
        )

    def settle_boundary_outcome(
        self,
        task_id: str,
        call_correlation: str,
        *,
        value: str | None = None,
        ref: OutcomeManifest | None = None,
    ) -> Advance:
        """Persist a mediated outcome and re-ready the suspended episode."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return Advance()
        corr = (wi.activation_id, call_correlation)
        env = self._boundaries.boundary_events.get(corr)
        resolved = wi.status is not WorkItemStatus.BLOCKED or (
            env is not None and self._boundaries.boundary_resolved(env)
        )
        if resolved:
            # A duplicate/late settle of an already-resolved member, or any settle for a
            # no-longer-BLOCKED (cancelled or terminal) work item, is an idempotent
            # no-op: it never re-runs the deliver path, so it cannot re-ready or stamp a
            # stale outcome on a resolved or cancelled boundary.
            return Advance()
        if env is not None and (value is not None or ref is not None):
            env = env.model_copy(update={"outcome_value": value, "outcome_ref": ref})
            self._boundaries.store_event(corr, env)
        if env is not None and env.group_id is not None:
            # A group member settled: hold the resume until every await-outcome member
            # is resolved, then re-ready exactly once with the full ordered vector.
            members = self._boundaries.group_members(wi.activation_id, env.group_id)
            if self._boundaries.group_awaits_unresolved(members):
                return Advance()
            wi.pending_outcome_group = env.group_id
            wi.pending_outcome_call = None
            return self.deliver_boundary_outcome(task_id, call_correlation)
        self.mark_pending_outcome(task_id, call_correlation)
        return self.deliver_boundary_outcome(task_id, call_correlation)

    def terminalize_boundary_invocation(
        self, task_id: str, call_correlation: str
    ) -> str | None:
        """Record a settled mediated boundary's invocation as terminal in the ledger."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self._boundaries.boundary_events.get((wi.activation_id, call_correlation))
        if env is None or env.invocation_id is None:
            return None
        invocation = self._ledger.invocations.get(env.invocation_id)
        if invocation is not None:
            invocation.state = next_on_terminal(invocation.state)
            self._ledger.emitter.emit_boundary(invocation)
        return env.invocation_id

    def terminalize_unsettled_invocations(
        self, task_ids: Iterable[str] | None = None
    ) -> list[str]:
        """Terminalize the unsettled mediated boundary invocations of the given tasks'
        activations, or of every activation; one already terminal is left as it is."""
        activations = (
            None
            if task_ids is None
            else {
                wi.activation_id
                for task_id in task_ids
                if (wi := self._ledger.work_item_for_task(task_id)) is not None
            }
        )
        ids: list[str] = []
        for (
            activation,
            invocation_id,
        ) in self._boundaries.unsettled_invocation_boundaries():
            if activations is not None and activation not in activations:
                continue
            invocation = self._ledger.invocations.get(invocation_id)
            if invocation is not None:
                if invocation.state in TERMINAL_INVOCATION_STATES:
                    continue
                invocation.state = next_on_terminal(invocation.state)
                self._ledger.emitter.emit_boundary(invocation)
            ids.append(invocation_id)
        return ids

    def _is_boundary_redrive(self, wi: WorkItem, event: BoundaryEvent) -> bool:
        """Whether a boundary reissues a recorded facade call under its stable id."""
        if event.call_correlation is None:
            return False
        if (
            wi.activation_id,
            event.call_correlation,
        ) not in self._boundaries.boundary_events:
            return False
        self._ledger.emit(
            "boundary_redriven",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"call": event.call_correlation},
        )
        return True

    def _validate_agent_boundary(
        self, op: AgentOperator, wi: WorkItem, event: BoundaryEvent
    ) -> DenialKind | None:
        """Check an agent boundary against its signature and effective authority.

        A kind outside the declared signature, a tool/model interface outside the
        effective invoke face, or a spawn/seal that names no declared child region — a
        raw operator id or an undeclared role — is a definitive denial: authority when
        the request is undeclared, policy when the pinned envelope blocks a declared
        one. None means admissible.
        """
        if event.kind not in op.boundary.events:
            return DenialKind.AUTHORITY
        if event.kind in (
            BoundaryEventKind.INVOCATION,
            BoundaryEventKind.EXTERNAL_EFFECT,
        ):
            invoke, _ = self._authority.agent_faces(op, wi)
            if event.interface is not None and event.interface not in invoke:
                return (
                    DenialKind.POLICY
                    if event.interface in op.authority.invoke
                    else DenialKind.AUTHORITY
                )
        elif event.kind in (BoundaryEventKind.SPAWN, BoundaryEventKind.SPAWN_SEAL):
            if self._topology.agent_region_op(op, event.child_region_ref) is None:
                return DenialKind.AUTHORITY
        return None

    def _agent_spawn_opener(
        self, op: LogicalOperator | None, wi: WorkItem, event: BoundaryEvent
    ) -> str:
        """The opener activation for the agent's selected region, minted on first use.

        Validation admitted the role, so it resolves to a declared spawn region; the
        opener owns that region's child-init scope keyed by (agent activation, region).
        """
        if not isinstance(op, AgentOperator):
            raise RegionError(
                f"{wi.operator_id!r} is not an agent; it cannot yield a spawn boundary"
            )
        region_op = self._topology.agent_region_op(op, event.child_region_ref)
        assert region_op is not None  # admitted by _validate_agent_boundary
        return self._spawns.region_opener(wi.activation_id, region_op)

    def _record_boundary(
        self,
        wi: WorkItem,
        event: BoundaryEvent,
        *,
        invocation_id: str | None = None,
        denial: DenialKind | None = None,
    ) -> str | None:
        """Persist the durable correlation envelope; return its idempotency key.

        Keyed on (activation, call correlation): a re-driven facade call reuses the
        recorded key rather than minting a new one, so a fresh harness call id never
        duplicates a dedupe-capable target effect. A boundary without a call correlation
        carries no durable dedupe handle and records nothing.
        """
        if event.call_correlation is None:
            return None
        corr = (wi.activation_id, event.call_correlation)
        existing = self._boundaries.boundary_events.get(corr)
        key = existing.idempotency_key if existing else new_idempotency_key()
        self._boundaries.store_event(
            corr,
            event.model_copy(
                update={
                    "activation": wi.activation_id,
                    "idempotency_key": key,
                    "invocation_id": invocation_id,
                    "denial": denial,
                }
            ),
        )
        detail = {"idempotency_key": key or "", "call": event.call_correlation}
        if denial is not None:
            detail["denial"] = denial.value
        self._ledger.emit(
            "boundary_recorded",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            invocation_id=invocation_id,
            detail=detail,
        )
        return key

    def _record_boundary_denial(
        self, wi: WorkItem, event: BoundaryEvent, denial: DenialKind
    ) -> Advance:
        """Settle a denied agent boundary as a durable typed continuation outcome.

        The denial is recorded against the operator and injected back as the boundary's
        outcome; it creates neither an invocation nor a child, and the episode suspends
        to receive it rather than silently proceeding.
        """
        subject = event.interface or event.child_region_ref or event.child_ref or ""
        self._authority.record_decision(
            AuthorityDecision(
                grant_id=self._ledger.root_grant.grant_id,
                interface=subject,
                kind=AuthorityDecisionKind.DENIED,
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
                denial_kind=denial,
                reason=f"boundary {event.kind.value} denied",
            )
        )
        self._record_boundary(wi, event, denial=denial)
        self._ledger.emit(
            "authority_denied" if denial is DenialKind.AUTHORITY else "policy_denied",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"interface": subject},
        )
        self._suspend_work_item(wi, "episode_suspended")
        return Advance()

    def _suspend_on_request(
        self, wi: WorkItem, event: BoundaryEvent, *, effect: bool
    ) -> Advance:
        invocation = Invocation(
            invocation_id=new_invocation_id(),
            work_item_id=wi.work_item_id,
            state=InvocationState.ISSUED,
            replayable=(
                is_replayable(wi.effect_class, wi.replay_contract) if effect else True
            ),
            compensable=(
                is_compensable(wi.effect_class, wi.replay_contract) if effect else False
            ),
        )
        self._ledger.invocations[invocation.invocation_id] = invocation
        self._record_boundary(wi, event, invocation_id=invocation.invocation_id)
        self._ledger.emit(
            "effect_requested" if effect else "invocation_issued",
            work_item_id=wi.work_item_id,
            invocation_id=invocation.invocation_id,
            operator_id=wi.operator_id,
            detail={"interface": event.interface or ""},
        )
        self._suspend_work_item(wi, "episode_suspended")
        return Advance()

    def _suspend_work_item(self, wi: WorkItem, kind: str) -> None:
        """Release the worker and suspend a work item awaiting a boundary reply."""
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        wi.status = WorkItemStatus.BLOCKED
        self._ledger.emit(
            kind, work_item_id=wi.work_item_id, operator_id=wi.operator_id
        )
