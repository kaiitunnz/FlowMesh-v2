"""The Admission controller: the fast admission loop over durable control facts.

It fair-orders claims from the DemandLedger, selects a compatible replica from the
derived CapacityPools and Replica directory, atomically records the fenced
``ServiceClaim.RESERVED`` credit, and advances the claim FSM as engine acknowledgements
and fenced ``DS`` terminal outcomes arrive. It writes only claim facts; capacity reports
are evidence it reads, never credit it mints. Every authoritative mutation is flushed
to the stores' durability before the step returns, so the credit survives a restart.
Once a consumed invocation's cleanup is done it retires into its durable record.
"""

import logging
from collections.abc import Callable

from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.utils.ids import new_admission_handoff_token

from ..utils.time import now_iso
from .capacity import default_credit
from .claim import (
    ClaimTransitionError,
    accept,
    begin_stream,
    mark_uncertain,
    new_claim,
    release_on_terminal,
    reserve,
    settle_terminal,
)
from .durability import CLAIM, REQUEST, live_field
from .selection import SelectionStrategy, build_selection_strategy
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    DemandEntry,
    InvocationRequest,
    InvocationSubject,
    ReplicaIncarnation,
    ReplicaState,
    RetiredInvocation,
    ServiceClaim,
)
from .stores import ResidentStores

_logger = logging.getLogger("resident-capacity")


class AdmissionController:
    """Orders claims, selects replicas, and advances the claim FSM and its credits."""

    def __init__(self, stores: ResidentStores) -> None:
        self._stores = stores
        self._strategies: dict[str, SelectionStrategy] = {}
        self._on_release: Callable[[str], None] = lambda _replica_id: None
        # The replicas of each invocation whose release a terminal began and a failed
        # persist or release hook left unfinished.
        self._unfinished: dict[str, list[str | None]] = {}

    def set_release_hook(self, on_release: Callable[[str], None]) -> None:
        """Call ``on_release`` with a replica's id when a claim on it releases."""
        self._on_release = on_release

    def _build_handoff(
        self,
        claim: ServiceClaim,
        replica: ReplicaIncarnation,
        *,
        idempotency_key: str | None,
        tenant: str | None,
        deadline_at: str | None,
        adapter_name: str | None = None,
        adapter_source: str | None = None,
        serve_task_id: str | None = None,
        binding_generation: int | None = None,
        descriptor_digest: str | None = None,
    ) -> AdmissionHandoff:
        """A single-use fence handoff for a reserved claim's replica incarnation.

        It carries no route or origin: the network path resolves and attaches those. It
        never carries the raw engine endpoint. For an adapter-bound claim it names the
        adapter the replica loads into a slot and the request selects. A task-addressed
        external claim also binds its authorized serve task, residency-binding
        generation, and the bounded canonical request digest the gate recomputes.
        """
        return AdmissionHandoff(
            token=new_admission_handoff_token(),
            claim_id=claim.claim_id,
            invocation_id=claim.invocation_id,
            idempotency_key=idempotency_key,
            family=claim.family,
            tenant=tenant,
            replica_id=replica.replica_id,
            incarnation=replica.incarnation,
            listener_generation=replica.listener_generation,
            expires_at=deadline_at,
            adapter_name=adapter_name,
            adapter_source=adapter_source,
            serve_task_id=serve_task_id,
            binding_generation=binding_generation,
            descriptor_digest=descriptor_digest,
        )

    def _strategy_for(self, family: str) -> SelectionStrategy:
        definition = self._stores.families.get(family)
        name = definition.selection_strategy if definition else None
        if family not in self._strategies:
            self._strategies[family] = build_selection_strategy(name)
        return self._strategies[family]

    def active_claim(self, invocation_id: str) -> ServiceClaim | None:
        """The non-terminal claim for an invocation, if one is in flight.

        A re-driven boundary attaches to this existing claim — a resume that reuses its
        held credit — rather than raising a successor and re-admitting, so a restart
        cannot double-submit or release the parked credit before the fenced terminal.
        """
        return next(
            (
                claim
                for claim in self._stores.claims.by_invocation(invocation_id)
                if claim.state is not ClaimState.TERMINAL
            ),
            None,
        )

    def raise_claim(
        self,
        *,
        invocation_id: str,
        subject: InvocationSubject,
        family: str,
        profile: AdmissionProfile,
        replayable: bool = True,
    ) -> ServiceClaim:
        """Record the durable invocation request and a fresh pending claim.

        Called only when no claim for the invocation is in flight. A permitted reissue
        is a successor: the admission epoch advances past every prior claim, live or
        counted by its retirement record, and no prior credit is released here —
        release happens only through the fenced terminal fact. The successor's
        request replaces one a consumed predecessor left, so it inherits no consumption.
        """
        live = self._stores.invocations.get(invocation_id)
        prior = self._stores.claims.by_invocation(invocation_id)
        if prior or live is not None:
            epoch = max((claim.admission_epoch + 1 for claim in prior), default=0)
        elif (record := self.retired_record(invocation_id)) is not None:
            if record.subject != subject or record.family != family:
                raise ClaimTransitionError(
                    f"invocation {invocation_id} was retired under another owner"
                )
            epoch = record.epoch_high + 1
        else:
            epoch = 0
        request = InvocationRequest(
            invocation_id=invocation_id,
            subject=subject,
            family=family,
            profile=profile,
            replayable=replayable,
        )
        if live is not None and live.consumed is not None:
            self._stores.invocations.replace(request)
        else:
            self._stores.invocations.put(request)
        claim = new_claim(
            invocation_id=invocation_id, family=family, admission_epoch=epoch
        )
        self._stores.claims.add(claim)
        self._stores.demand.enqueue(
            DemandEntry(
                claim_id=claim.claim_id,
                invocation_id=invocation_id,
                family=family,
                tenant=profile.tenant,
                deadline_at=profile.deadline_at,
            )
        )
        self._stores.flush()
        return claim

    def rebuild_handoff(
        self, claim: ServiceClaim, *, idempotency_key: str | None
    ) -> AdmissionHandoff | None:
        """A fresh claim-bound handoff for an in-flight claim's replica, if still live.

        Resuming a re-driven boundary reissues to the same fenced replica incarnation
        under the existing credit; if the replica is gone or superseded, no handoff is
        built and the caller settles the boundary so the fenced terminal releases the
        credit.
        """
        if claim.replica_id is None or claim.incarnation is None:
            return None
        replica = self._stores.directory.get(claim.replica_id)
        # A draining replica still finishes the work it admitted.
        if (
            replica is None
            or replica.endpoint is None
            or replica.incarnation != claim.incarnation
            or (
                replica.state not in SERVABLE_REPLICA_STATES
                and not (replica.state is ReplicaState.DRAINING and claim.holds_credit)
            )
        ):
            return None
        request = self._stores.invocations.get(claim.invocation_id)
        profile = request.profile if request is not None else None
        return self._build_handoff(
            claim,
            replica,
            idempotency_key=idempotency_key,
            tenant=profile.tenant if profile is not None else None,
            deadline_at=profile.deadline_at if profile is not None else None,
            adapter_name=profile.adapter_ref if profile is not None else None,
            adapter_source=profile.adapter_source if profile is not None else None,
            serve_task_id=profile.serve_task_id if profile is not None else None,
            binding_generation=(
                profile.binding_generation if profile is not None else None
            ),
            descriptor_digest=(
                profile.descriptor_digest if profile is not None else None
            ),
        )

    def admit(
        self,
        claim: ServiceClaim,
        profile: AdmissionProfile,
        *,
        idempotency_key: str | None,
        now_ts: float | None = None,
    ) -> AdmissionHandoff | None:
        """Select a feasible replica and record the fenced RESERVED credit, or defer.

        Returns the single-use claim-bound handoff on success; ``None`` leaves the claim
        pending without holding a replica or an episode lane when no replica is
        feasible.
        """
        candidates = self._stores.pools.feasible_candidates(
            claim.family, profile, now_ts=now_ts
        )
        credit = default_credit(profile)
        chosen = self._strategy_for(claim.family).select(candidates, credit)
        if chosen is None:
            return None
        replica = self._stores.directory.get(chosen.replica_id)
        if replica is None or replica.endpoint is None:
            return None
        reserve(
            claim,
            replica_id=replica.replica_id,
            incarnation=replica.incarnation,
            credit=credit,
        )
        replica.last_active_at = now_iso()
        self._stores.demand.mark_admitted(claim.claim_id)
        self._stores.flush()
        return self._build_handoff(
            claim,
            replica,
            idempotency_key=idempotency_key,
            tenant=profile.tenant,
            deadline_at=profile.deadline_at,
            adapter_name=profile.adapter_ref,
            adapter_source=profile.adapter_source,
            serve_task_id=profile.serve_task_id,
            binding_generation=profile.binding_generation,
            descriptor_digest=profile.descriptor_digest,
        )

    def accept_and_authorize(
        self,
        claim: ServiceClaim,
        *,
        idempotency_key: str | None,
        origin_id: str | None,
        deadline_at: str | None = None,
    ) -> RouteAuthorization:
        """Record ``ACCEPTED`` on an engine enqueue ack and issue the immutable fence.

        The fence is minted only after acknowledgement and is re-mintable from the
        durable claim on a resume.
        """
        accept(claim)
        self._stores.flush()
        return self._mint_authorization(
            claim,
            idempotency_key=idempotency_key,
            origin_id=origin_id,
            deadline_at=deadline_at,
        )

    def reauthorize(
        self,
        claim: ServiceClaim,
        *,
        idempotency_key: str | None,
        origin_id: str | None,
        deadline_at: str | None = None,
    ) -> RouteAuthorization:
        """Re-mint the fence for a resumed in-flight claim without an FSM transition.

        A re-driven boundary on a claim already past ``RESERVED`` reissues to the same
        held credit under a freshly minted fence for the same replica incarnation.
        """
        return self._mint_authorization(
            claim,
            idempotency_key=idempotency_key,
            origin_id=origin_id,
            deadline_at=deadline_at,
        )

    def _mint_authorization(
        self,
        claim: ServiceClaim,
        *,
        idempotency_key: str | None,
        origin_id: str | None,
        deadline_at: str | None,
    ) -> RouteAuthorization:
        assert claim.replica_id is not None and claim.incarnation is not None
        replica = self._stores.directory.get(claim.replica_id)
        request = self._stores.invocations.get(claim.invocation_id)
        profile = request.profile if request is not None else None
        return RouteAuthorization(
            claim_id=claim.claim_id,
            invocation_id=claim.invocation_id,
            idempotency_key=idempotency_key,
            tenant=profile.tenant if profile is not None else None,
            origin_id=origin_id,
            replica_id=claim.replica_id,
            incarnation=claim.incarnation,
            listener_generation=(
                replica.listener_generation if replica is not None else 0
            ),
            expires_at=deadline_at,
            serve_task_id=profile.serve_task_id if profile is not None else None,
            binding_generation=(
                profile.binding_generation if profile is not None else None
            ),
        )

    def on_stream_started(self, claim: ServiceClaim) -> None:
        """Record the start of the authorized response stream for an accepted claim."""
        begin_stream(claim)
        self._stores.flush_or_defer()

    def on_enqueue_failed(self, claim: ServiceClaim) -> None:
        """Release a reserved credit on a known pre-acceptance enqueue failure.

        A release whose write fails is left for the invocation's consumption to finish.
        """
        settle_terminal(claim, ClaimTerminalReason.ENQUEUE_FAILED)
        self._touch_replica(claim)
        self._unfinished.setdefault(claim.invocation_id, []).append(claim.replica_id)
        try:
            self._finish_release(claim.invocation_id)
        except Exception:
            _logger.warning(
                "releasing resident claim %s deferred", claim.claim_id, exc_info=True
            )

    def _touch_replica(self, claim: ServiceClaim) -> None:
        """Stamp the claim's replica idle-clock so a retain window starts at release."""
        if claim.replica_id is None:
            return
        if (replica := self._stores.directory.get(claim.replica_id)) is not None:
            replica.last_active_at = now_iso()

    def on_denied(self, claim: ServiceClaim) -> None:
        """Terminate a still-pending claim a policy denial refused before any credit."""
        settle_terminal(claim, ClaimTerminalReason.CANCELLED)
        self._stores.demand.remove(claim.claim_id)
        self._stores.flush()

    def on_expired(self, claim: ServiceClaim) -> None:
        """Terminate a still-pending claim whose cold-start budget elapsed."""
        settle_terminal(claim, ClaimTerminalReason.EXPIRED)
        self._stores.demand.remove(claim.claim_id)
        self._stores.flush()

    def on_route_loss(self, claim: ServiceClaim) -> None:
        """Hold the credit after a post-acceptance route loss, pending reconciliation.

        The credit is not released here: a lost route is uncertain, not evidence the
        invocation completed. It releases only when the fenced DS terminal arrives.
        """
        if claim.holds_credit and claim.state is not ClaimState.UNCERTAIN:
            mark_uncertain(claim)
            self._stores.flush_or_defer()

    def settle_invocation_terminal(
        self, invocation_id: str, reason: ClaimTerminalReason
    ) -> bool:
        """Settle every non-terminal claim of an invocation from a fenced terminal fact.

        This is the sole normal release path for an accepted credit. A workflow
        subject's fact is the orchestration engine's ``DS`` outcome; an external
        subject's is a durable external status-terminal fact. The controller consumes
        either by ``invocation_id``, tolerant of the claim's source state; it never
        assumes a ``DS`` record exists. The first consumption is recorded on the request
        in the same write that settles its claims, whether or not any credit releases,
        and a later one leaves that record as it is. Return whether any claim released,
        now or in the earlier consumption this one finishes.
        """
        released: list[str | None] = []
        for claim in self._stores.claims.by_invocation(invocation_id):
            if claim.state is not ClaimState.TERMINAL:
                release_on_terminal(claim, reason)
                self._stores.demand.remove(claim.claim_id)
                self._touch_replica(claim)
                released.append(claim.replica_id)
        if (
            request := self._stores.invocations.get(invocation_id)
        ) is not None and request.consumed is None:
            self._stores.invocations.replace(
                request.model_copy(update={"consumed": reason})
            )
        unfinished = self._unfinished.get(invocation_id)
        if not released and unfinished is None:
            self._stores.flush()
            return False
        # A terminal handed over again finishes a release an earlier one began, so the
        # claim is persisted and each replica hears of its release once.
        self._unfinished.setdefault(invocation_id, []).extend(released)
        self._finish_release(invocation_id)
        return True

    def _finish_release(self, invocation_id: str) -> None:
        """Make the released claims durable, then tell each of their replicas once."""
        pending = self._unfinished[invocation_id]
        self._stores.flush()
        while pending:
            if (replica_id := pending[0]) is not None:
                self._on_release(replica_id)
            pending.pop(0)
        del self._unfinished[invocation_id]

    def retired_record(self, invocation_id: str) -> RetiredInvocation | None:
        """Read the invocation's stored retirement record, raising on a store failure
        or an unreadable record."""
        [record] = self._stores.durability.retired_invocations([invocation_id])
        return record

    def retirable(self, invocation_id: str) -> bool:
        """Report whether a consumed invocation's cleanup is done."""
        request = self._stores.invocations.get(invocation_id)
        return (
            request is not None
            and request.consumed is not None
            and invocation_id not in self._unfinished
            and all(
                claim.state is ClaimState.TERMINAL
                for claim in self._stores.claims.by_invocation(invocation_id)
            )
        )

    def retire(self, invocation_id: str) -> bool:
        """Retire a consumed invocation whose cleanup is done into its durable record.

        Its pending changes are flushed, the record is published and the live fields
        deleted, and only then do its claims, demand and request leave the stores. A
        failure at any point raises and leaves the invocation live, so a later call
        retires it. Return whether it retired.
        """
        if not self.retirable(invocation_id):
            return False
        request = self._stores.invocations.get(invocation_id)
        assert request is not None and request.consumed is not None
        claims = self._stores.claims.by_invocation(invocation_id)
        self._stores.flush()
        record = RetiredInvocation(
            invocation_id=invocation_id,
            subject=request.subject,
            family=request.family,
            epoch_high=max((claim.admission_epoch for claim in claims), default=0),
            terminal_reason=request.consumed,
        )
        self._stores.durability.retire_invocation(
            record,
            [live_field(REQUEST, invocation_id)]
            + [live_field(CLAIM, claim.claim_id) for claim in claims],
        )
        self._stores.forget_invocation(invocation_id)
        return True
