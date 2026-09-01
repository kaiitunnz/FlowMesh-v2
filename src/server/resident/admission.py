"""The Admission controller: the fast admission loop over durable control facts.

It fair-orders claims from the DemandLedger, selects a compatible replica from the
derived CapacityPools and Replica directory, atomically records the fenced
``ServiceClaim.RESERVED`` credit, and advances the claim FSM as engine acknowledgements
and fenced ``DS`` terminal outcomes arrive. It writes only claim facts; capacity reports
are evidence it reads, never credit it mints. Every authoritative mutation is persisted
through the injected hook so the credit survives a restart.
"""

from collections.abc import Callable

from shared.utils.ids import new_admission_handoff_token

from ..utils.time import now_iso
from .capacity import default_credit
from .claim import (
    accept,
    begin_stream,
    mark_uncertain,
    new_claim,
    release_on_ds_terminal,
    reserve,
    settle_terminal,
)
from .selection import SelectionStrategy, build_selection_strategy
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionHandoff,
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    DemandEntry,
    InvocationRequest,
    ReplicaIncarnation,
    RouteAuthorization,
    ServiceClaim,
)
from .stores import ResidentStores


class AdmissionController:
    """Orders claims, selects replicas, and advances the claim FSM and its credits."""

    def __init__(
        self, stores: ResidentStores, persist: Callable[[], None] | None = None
    ) -> None:
        self._stores = stores
        self._persist = persist or (lambda: None)
        self._strategies: dict[str, SelectionStrategy] = {}
        self._route_auth_epochs: dict[str, int] = {}

    def _build_handoff(
        self,
        claim: ServiceClaim,
        replica: ReplicaIncarnation,
        *,
        idempotency_key: str | None,
        tenant: str | None,
        deadline_at: str | None,
    ) -> AdmissionHandoff:
        """A single-use fence handoff for a reserved claim's replica incarnation.

        It carries no route or origin: the network path resolves and attaches those, and
        the compatibility path reaches the replica endpoint from the directory. It never
        carries the raw engine endpoint.
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
            deadline_at=deadline_at,
            expires_at=deadline_at,
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
        workflow_id: str,
        family: str,
        profile: AdmissionProfile,
        replayable: bool = True,
    ) -> ServiceClaim:
        """Record the durable invocation request and a fresh pending claim.

        Called only when no claim for the invocation is in flight. A permitted reissue
        is a successor: the admission epoch advances past every prior terminal claim,
        and no prior credit is released here — release happens only through the fenced
        DS terminal.
        """
        self._stores.invocations.put(
            InvocationRequest(
                invocation_id=invocation_id,
                workflow_id=workflow_id,
                family=family,
                profile=profile,
                replayable=replayable,
            )
        )
        epoch = max(
            (
                prior.admission_epoch + 1
                for prior in self._stores.claims.by_invocation(invocation_id)
            ),
            default=0,
        )
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
        self._persist()
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
        if (
            replica is None
            or replica.endpoint is None
            or replica.incarnation != claim.incarnation
            or replica.state not in SERVABLE_REPLICA_STATES
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
        self._persist()
        return self._build_handoff(
            claim,
            replica,
            idempotency_key=idempotency_key,
            tenant=profile.tenant,
            deadline_at=profile.deadline_at,
        )

    def on_enqueue_ack(self, claim: ServiceClaim) -> None:
        """Record a single-shot enqueue acknowledgement, then the response stream.

        Used by the in-server compatibility path, where acknowledgement and completion
        arrive together, so there is no separate authorized stream to fence.
        """
        accept(claim)
        begin_stream(claim)
        self._persist()

    def accept_and_authorize(
        self,
        claim: ServiceClaim,
        *,
        idempotency_key: str | None,
        origin_id: str | None,
        operation: str,
        budget: int | None = None,
        deadline_at: str | None = None,
    ) -> RouteAuthorization:
        """Record ``ACCEPTED`` on an engine enqueue ack and issue the immutable fence.

        The fence is minted only after acknowledgement and is re-mintable from the
        durable claim on a resume; a fresh route-authorization epoch supersedes any
        prior mint for the same claim.
        """
        accept(claim)
        assert claim.replica_id is not None and claim.incarnation is not None
        epoch = self._route_auth_epochs.get(claim.claim_id, 0) + 1
        self._route_auth_epochs[claim.claim_id] = epoch
        self._persist()
        replica = self._stores.directory.get(claim.replica_id)
        request = self._stores.invocations.get(claim.invocation_id)
        return RouteAuthorization(
            claim_id=claim.claim_id,
            invocation_id=claim.invocation_id,
            idempotency_key=idempotency_key,
            family=claim.family,
            operation=operation,
            admission_epoch=claim.admission_epoch,
            route_auth_epoch=epoch,
            tenant=request.profile.tenant if request is not None else None,
            origin_id=origin_id,
            replica_id=claim.replica_id,
            incarnation=claim.incarnation,
            listener_generation=(
                replica.listener_generation if replica is not None else 0
            ),
            deadline_at=deadline_at,
            budget=budget,
            expires_at=deadline_at,
        )

    def on_stream_started(self, claim: ServiceClaim) -> None:
        """Record the start of the authorized response stream for an accepted claim."""
        begin_stream(claim)
        self._persist()

    def on_enqueue_failed(self, claim: ServiceClaim) -> None:
        """Release a reserved credit on a known pre-acceptance enqueue failure."""
        settle_terminal(claim, ClaimTerminalReason.ENQUEUE_FAILED)
        self._touch_replica(claim)
        self._persist()

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
        self._persist()

    def on_expired(self, claim: ServiceClaim) -> None:
        """Terminate a still-pending claim whose cold-start budget elapsed."""
        settle_terminal(claim, ClaimTerminalReason.EXPIRED)
        self._stores.demand.remove(claim.claim_id)
        self._persist()

    def on_route_loss(self, claim: ServiceClaim) -> None:
        """Hold the credit after a post-acceptance route loss, pending reconciliation.

        The credit is not released here: a lost route is uncertain, not evidence the
        invocation completed. It releases only when the fenced DS terminal arrives.
        """
        if claim.holds_credit and claim.state is not ClaimState.UNCERTAIN:
            mark_uncertain(claim)
            self._persist()

    def on_ds_terminal(self, invocation_id: str, reason: ClaimTerminalReason) -> None:
        """Settle every non-terminal claim of an invocation from a fenced DS outcome.

        This is the sole normal release path for an accepted credit: the orchestration
        engine records the terminal outcome and the controller consumes it by
        ``invocation_id``, tolerant of the claim's source state.
        """
        released = False
        for claim in self._stores.claims.by_invocation(invocation_id):
            if claim.state is not ClaimState.TERMINAL:
                release_on_ds_terminal(claim, reason)
                self._stores.demand.remove(claim.claim_id)
                self._touch_replica(claim)
                released = True
        if released:
            self._persist()
