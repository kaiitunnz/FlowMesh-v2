"""Recorded mediated boundaries of one workflow instance."""

from collections.abc import Sequence

from shared.harness import DeliveredOutcome, OutcomeKind

from ...task.v2.representations.operators import (
    BoundaryEventKind,
)
from ..state import (
    TERMINAL_INVOCATION_STATES,
    BoundaryEvent,
    WorkItem,
    WorkItemStatus,
)
from ..tool_dispatch import (
    MODEL_INTERFACE,
    FacadeCallMember,
    FacadeCompletionMode,
    ToolInvocationEnvelope,
)
from .authority import AuthorityLedger
from .ledger import OrchestrationLedger

# Boundary kinds an off-lane handler settles while their episode is suspended.
_MEDIATED_BOUNDARY_KINDS = frozenset(
    {BoundaryEventKind.INVOCATION, BoundaryEventKind.EXTERNAL_EFFECT}
)


class BoundaryLedger:
    """Holds each mediated boundary an episode recorded, keyed by its activation and
    call correlation, and answers what is pending or settled on them."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        authority: AuthorityLedger,
    ) -> None:
        self._ledger = ledger
        self._authority = authority
        self.boundary_events: dict[tuple[str, str], BoundaryEvent] = {}

    def store_event(self, key: tuple[str, str], event: BoundaryEvent) -> None:
        self.boundary_events[key] = event

    def facade_member_event(
        self, group_id: str, member: FacadeCallMember
    ) -> BoundaryEvent:
        is_spawn = member.kind is BoundaryEventKind.SPAWN
        return BoundaryEvent(
            kind=member.kind,
            interface=None if is_spawn else member.interface_or_region,
            child_region_ref=member.interface_or_region if is_spawn else None,
            call_correlation=member.call_correlation,
            request_payload=member.request_payload,
            request_digest=member.request_digest,
            injection_target=member.harness_call_id,
            injection_tool=member.tool_name,
            group_id=group_id,
            group_ordinal=member.ordinal,
            completion_mode=member.completion_mode.value,
        )

    def set_member_outcome(
        self, wi: WorkItem, call_correlation: str | None, value: str
    ) -> None:
        """Settle a group member's outcome in place without re-readying the lane."""
        if call_correlation is None:
            return
        corr = (wi.activation_id, call_correlation)
        if (env := self.boundary_events.get(corr)) is not None:
            self.boundary_events[corr] = env.model_copy(update={"outcome_value": value})

    def group_members(self, activation_id: str, group_id: str) -> list[BoundaryEvent]:
        members = [
            env
            for (act, _), env in self.boundary_events.items()
            if act == activation_id and env.group_id == group_id
        ]
        members.sort(key=lambda e: e.group_ordinal or 0)
        return members

    @staticmethod
    def boundary_resolved(env: BoundaryEvent) -> bool:
        """Whether a boundary is settled: an inline value, a reference, or a denial."""
        return (
            env.outcome_value is not None
            or env.outcome_ref is not None
            or env.denial is not None
        )

    def awaits_worker_held_boundary(self, task_id: str) -> bool:
        """Whether the task is suspended on an unsettled boundary whose raw request
        only its capturing worker holds."""
        wi = self._ledger.work_item_for_task(task_id)
        return (
            wi is not None
            and wi.status is WorkItemStatus.BLOCKED
            and self.has_pending_local_boundary(wi)
        )

    def has_pending_local_boundary(self, wi: WorkItem) -> bool:
        """Whether the work item awaits an unsettled worker-originated boundary.

        A recorded request digest marks a boundary whose raw request lives only on the
        capturing worker; if that worker is lost the boundary cannot be recovered here.
        """
        return any(
            act == wi.activation_id
            and env.request_digest is not None
            and not self.boundary_resolved(env)
            for (act, _), env in self.boundary_events.items()
        )

    @classmethod
    def group_awaits_unresolved(cls, members: Sequence[BoundaryEvent]) -> bool:
        """Whether the group holds an await-outcome member with no settled outcome."""
        return any(
            e.completion_mode == FacadeCompletionMode.AWAIT_OUTCOME.value
            and not cls.boundary_resolved(e)
            for e in members
        )

    def has_open_facade_group(self, task_id: str) -> bool:
        """Whether a recorded facade group for this episode still holds the resume gate.

        A group is open only while an await-outcome member is unsettled; a spawn-only
        group closes at admission, so the fence lets the next turn issue another group.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return False
        return self.group_awaits_unresolved(
            [
                env
                for (act, _), env in self.boundary_events.items()
                if act == wi.activation_id and env.group_id is not None
            ]
        )

    def group_dispatch_envelopes(
        self, task_id: str, group_id: str
    ) -> list[ToolInvocationEnvelope]:
        """The dispatch envelopes for a group's still-unresolved invocation members."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return []
        out: list[ToolInvocationEnvelope] = []
        for env in self.group_members(wi.activation_id, group_id):
            if env.kind is not BoundaryEventKind.INVOCATION:
                continue
            if self.boundary_resolved(env):
                continue
            if (envelope := self._envelope_from(wi, env)) is not None:
                out.append(envelope)
        return out

    def awaits_mediated_outcome(self, wi: WorkItem) -> bool:
        """Whether the work item's activation awaits a mediated boundary with no
        outcome."""
        return any(
            act == wi.activation_id
            and env.kind in _MEDIATED_BOUNDARY_KINDS
            and not self.boundary_resolved(env)
            for (act, _), env in self.boundary_events.items()
        )

    def pending_tool_dispatches(self) -> list[ToolInvocationEnvelope]:
        """Mediated boundaries suspended with no durable outcome, for a restart.

        A model or tool boundary suspends off-lane while its handler settles it; a crash
        before that settle leaves the work item blocked with an issued invocation and no
        recorded outcome. Returns each as a full dispatch envelope so the runtime routes
        it back to its handler by (kind, interface) — a search to the broker, a model to
        the gateway — never misrouting on the recovered kind.
        """
        pending: list[ToolInvocationEnvelope] = []
        for (activation, corr), env in self.boundary_events.items():
            if env.kind not in _MEDIATED_BOUNDARY_KINDS:
                continue
            if self.boundary_resolved(env):
                continue
            wi = self._ledger.wi_by_activation.get(activation)
            work_item = self._ledger.work_items.get(wi) if wi else None
            if work_item is None or work_item.status is not WorkItemStatus.BLOCKED:
                continue
            if (envelope := self._envelope_from(work_item, env)) is not None:
                pending.append(envelope)
        return pending

    def tool_dispatch_envelope(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a recorded mediated boundary, or None if absent."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self.boundary_events.get((wi.activation_id, call_correlation))
        if env is None:
            return None
        return self._envelope_from(wi, env)

    def pending_tool_dispatch(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a still-pending mediated boundary, or None.

        Returns an envelope only while the boundary is suspended with no durable outcome
        and its work item is blocked, so a held re-drive re-issues exactly the off-lane
        dispatch a restart would; a settled, terminalized, or cancelled boundary yields
        nothing.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return None
        env = self.boundary_events.get((wi.activation_id, call_correlation))
        if env is None or self.boundary_resolved(env):
            return None
        return self._envelope_from(wi, env)

    def _envelope_from(
        self, wi: WorkItem, env: BoundaryEvent
    ) -> ToolInvocationEnvelope | None:
        if env.invocation_id is None or env.call_correlation is None:
            return None
        return ToolInvocationEnvelope(
            kind=env.kind,
            interface=env.interface or MODEL_INTERFACE,
            invocation_id=env.invocation_id,
            task_id=wi.legacy_task_id,
            activation_id=wi.activation_id,
            call_correlation=env.call_correlation,
            idempotency_key=env.idempotency_key,
            request_payload=env.request_payload,
            request_digest=env.request_digest,
            grant_snapshot=self._authority.grant_snapshot_for(wi),
        )

    def boundary_settleable(self, task_id: str, call_correlation: str) -> bool:
        """Whether a mediated settle for this exact boundary is still legal.

        A settle is legal only while the boundary's work item is still BLOCKED — its
        episode suspended on the mediated call — and the boundary itself is unresolved.
        A cancelled or otherwise non-blocked work item, or an already-settled or denied
        boundary, is absorbing: a late or duplicate model, tool, or resident delivery is
        audit evidence only. It must not stamp an outcome, re-ready the episode, replace
        the durable correlation, or release a resident credit a second time. Gate on the
        BLOCKED work item, not a task record's status, since cancellation can leave the
        record CANCELLING while the work item is already CANCELLED.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return False
        env = self.boundary_events.get((wi.activation_id, call_correlation))
        return env is not None and not self.boundary_resolved(env)

    def boundary_invocation_completed(self, invocation_id: str) -> bool | None:
        """Whether a terminal boundary invocation completed with an outcome; None while
        it is unknown or not terminal."""
        invocation = self._ledger.invocations.get(invocation_id)
        if invocation is None or invocation.state not in TERMINAL_INVOCATION_STATES:
            return None
        return any(
            env.invocation_id == invocation_id
            and (env.outcome_value is not None or env.outcome_ref is not None)
            for env in self.boundary_events.values()
        )

    def suspended_boundary_tasks(self) -> list[str]:
        """Tasks suspended at an unsettled mediated boundary.

        Such a task's worker released the lane, so it holds no dispatch and returns no
        terminal; a task mid-step is not among them.
        """
        tasks: list[str] = []
        for activation, _ in self.unsettled_invocation_boundaries():
            wi_id = self._ledger.wi_by_activation.get(activation)
            if (wi := self._ledger.work_items.get(wi_id) if wi_id else None) is None:
                continue
            if wi.legacy_task_id not in tasks:
                tasks.append(wi.legacy_task_id)
        return tasks

    def unsettled_invocation_boundaries(self) -> list[tuple[str, str]]:
        """Each unsettled mediated-boundary invocation id and its owning activation."""
        return [
            (activation, env.invocation_id)
            for (activation, _), env in self.boundary_events.items()
            if env.invocation_id is not None and not self.boundary_resolved(env)
        ]

    def episode_context(
        self, task_id: str
    ) -> tuple[str | None, tuple[DeliveredOutcome, ...]]:
        """The durable capsule and pending injected outcome for an agent's next step.

        Rebuilt from the ledger, never in-memory episode state: the capsule is the work
        item's continuation, and the one pending outcome is reconstructed from its
        settled boundary envelope, so a re-dispatch after a restart carries it again.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None, ()
        outcomes: tuple[DeliveredOutcome, ...] = ()
        if wi.pending_outcome_group is not None:
            members = self.group_members(wi.activation_id, wi.pending_outcome_group)
            outcomes = tuple(self._delivered_outcome(env) for env in members)
        elif wi.pending_outcome_call is not None:
            env = self.boundary_events.get((wi.activation_id, wi.pending_outcome_call))
            if env is not None:
                outcomes = (self._delivered_outcome(env),)
        return wi.continuation_ref, outcomes

    @staticmethod
    def _delivered_outcome(env: BoundaryEvent) -> DeliveredOutcome:
        corr = env.call_correlation or ""
        if env.denial is not None:
            return DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=env.idempotency_key,
                kind=OutcomeKind.DENIED,
                denial=env.denial,
                injection_target=env.injection_target,
                injection_tool=env.injection_tool,
                injection_arguments=env.request_payload,
            )
        return DeliveredOutcome(
            call_correlation=corr,
            idempotency_key=env.idempotency_key,
            kind=OutcomeKind.RESULT,
            value=env.outcome_value,
            outcome_ref=env.outcome_ref,
            injection_target=env.injection_target,
            injection_tool=env.injection_tool,
        )

    def boundary_envelope(
        self, activation_id: str, call_correlation: str
    ) -> BoundaryEvent | None:
        """The durable envelope recorded for one mediated facade call, if any.

        Carries the fabric-assigned idempotency key, the causal invocation id, and the
        outcome (or denial) the continuation resumes with.
        """
        return self.boundary_events.get((activation_id, call_correlation))
