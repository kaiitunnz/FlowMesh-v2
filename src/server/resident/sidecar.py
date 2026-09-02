"""The resident-facing claim gate: target-side fence validation and load evidence.

A resident allocation is fronted by a sidecar that admits data-plane traffic only after
validating the fence it carries against the sidecar's own replica incarnation and
listener generation. The pure validation here is that gate: it accepts a claim-bound
bootstrap handoff to open a session, then admits the authorized response stream only
under a matching immutable route authorization, rejecting a fence that is expired, names
another replica incarnation or a superseded listener generation, or does not continue
the session's tenant subject, claim, invocation, or request identity. It validates those
bindings and trusts that only the origin deputy reaches its per-replica route; it does
not track the handoff token to reject a replay, which the deferred credential handshake
would add.

The gate is the target-side authority: an intermediate relay may validate its own hop,
but never substitutes for this check before engine delivery. A rejection is an
authorization failure, not a network-path failure, so a caller reports it as
non-demoting route evidence. Every admitted operation carries claim-tagged load evidence
for control-plane accounting, distinguished from bulk transfer.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from ..utils.time import now_iso, parse_iso_ts
from .state import AdmissionHandoff, RouteAuthorization


class GateRejection(StrEnum):
    """Why the claim gate refused a fence, all authorization (never path) failures."""

    EXPIRED = "expired"
    WRONG_INCARNATION = "wrong_incarnation"
    STALE_LISTENER = "stale_listener"
    WRONG_REPLICA = "wrong_replica"
    WRONG_SUBJECT = "wrong_subject"
    WRONG_CLAIM = "wrong_claim"
    WRONG_INVOCATION = "wrong_invocation"
    WRONG_REQUEST = "wrong_request"
    NO_SESSION = "no_session"


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict; ``rejection`` is set only when ``admitted`` is false."""

    admitted: bool
    rejection: GateRejection | None = None

    @classmethod
    def ok(cls) -> "GateDecision":
        return cls(admitted=True)

    @classmethod
    def deny(cls, rejection: GateRejection) -> "GateDecision":
        return cls(admitted=False, rejection=rejection)


@dataclass(frozen=True)
class SidecarSession:
    """The accepted context a bootstrap opened, matched by later authorized traffic."""

    claim_id: str
    invocation_id: str
    idempotency_key: str | None
    tenant: str | None
    origin_id: str | None


class TrafficClass(StrEnum):
    """Latency-sensitive service traffic versus bulk snapshot/artifact transfer."""

    SERVICE = "service"
    BULK = "bulk"


@dataclass(frozen=True)
class LoadEvidence:
    """Claim-tagged evidence one admitted data-plane operation emits for accounting."""

    claim_id: str
    invocation_id: str
    replica_id: str
    incarnation: int
    operation: str
    traffic_class: TrafficClass = TrafficClass.SERVICE


class SidecarClaimGate:
    """Validates every handoff or route a resident allocation's sidecar is offered.

    Bound to the one replica incarnation and listener generation it fronts, so a fence
    minted for a superseded incarnation or a stale listener is refused before any engine
    delivery.
    """

    def __init__(
        self,
        *,
        replica_id: str,
        incarnation: int,
        listener_generation: int,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self._replica_id = replica_id
        self._incarnation = incarnation
        self._listener_generation = listener_generation
        self._clock = clock

    def _expired(self, expires_at: str | None) -> bool:
        return expires_at is not None and parse_iso_ts(self._clock()) >= parse_iso_ts(
            expires_at
        )

    def check_bootstrap(self, handoff: AdmissionHandoff) -> GateDecision:
        """Admit a claim-bound bootstrap delivery, or reject its fence."""
        if handoff.replica_id != self._replica_id:
            return GateDecision.deny(GateRejection.WRONG_REPLICA)
        if handoff.incarnation != self._incarnation:
            return GateDecision.deny(GateRejection.WRONG_INCARNATION)
        if handoff.listener_generation != self._listener_generation:
            return GateDecision.deny(GateRejection.STALE_LISTENER)
        if self._expired(handoff.expires_at):
            return GateDecision.deny(GateRejection.EXPIRED)
        return GateDecision.ok()

    def session_for(self, handoff: AdmissionHandoff) -> SidecarSession:
        """The accepted context an admitted bootstrap opens for its response stream."""
        return SidecarSession(
            claim_id=handoff.claim_id,
            invocation_id=handoff.invocation_id,
            idempotency_key=handoff.idempotency_key,
            tenant=handoff.tenant,
            origin_id=handoff.origin_id,
        )

    def check_stream(
        self, auth: RouteAuthorization, session: SidecarSession | None
    ) -> GateDecision:
        """Admit authorized stream traffic only under a fence continuing the session.

        The route authorization must name this replica incarnation and current listener
        generation, be unexpired, and continue the bootstrap session's tenant subject,
        claim, invocation, and request identity — so a refreshed route can never widen
        the fence onto another subject or incarnation.
        """
        if session is None:
            return GateDecision.deny(GateRejection.NO_SESSION)
        if auth.replica_id != self._replica_id:
            return GateDecision.deny(GateRejection.WRONG_REPLICA)
        if auth.incarnation != self._incarnation:
            return GateDecision.deny(GateRejection.WRONG_INCARNATION)
        if auth.listener_generation != self._listener_generation:
            return GateDecision.deny(GateRejection.STALE_LISTENER)
        if self._expired(auth.expires_at):
            return GateDecision.deny(GateRejection.EXPIRED)
        if auth.claim_id != session.claim_id:
            return GateDecision.deny(GateRejection.WRONG_CLAIM)
        if auth.invocation_id != session.invocation_id:
            return GateDecision.deny(GateRejection.WRONG_INVOCATION)
        if auth.idempotency_key != session.idempotency_key:
            return GateDecision.deny(GateRejection.WRONG_REQUEST)
        if auth.tenant != session.tenant or auth.origin_id != session.origin_id:
            return GateDecision.deny(GateRejection.WRONG_SUBJECT)
        return GateDecision.ok()

    def load_evidence(
        self,
        handoff_or_auth: AdmissionHandoff | RouteAuthorization,
        operation: str,
        *,
        traffic_class: TrafficClass = TrafficClass.SERVICE,
    ) -> LoadEvidence:
        """Claim-tagged evidence for one admitted operation on this replica."""
        return LoadEvidence(
            claim_id=handoff_or_auth.claim_id,
            invocation_id=handoff_or_auth.invocation_id,
            replica_id=self._replica_id,
            incarnation=self._incarnation,
            operation=operation,
            traffic_class=traffic_class,
        )
