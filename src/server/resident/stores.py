"""Control-state (``CS``) stores for resident-capacity control.

The Service-family registry, Replica directory, DemandLedger, Allocation leases, durable
invocation requests, and ``ServiceClaim`` facts are authoritative in their domains. Only
``ServiceClaim`` facts are authoritative for admission credits. The derived
``CapacityPools`` and Admission-credit ledger are computed from those authorities,
so they cannot diverge: no cached credit exists for a report or a provisioning decision
to overwrite or release. Capacity reports are evidence, fenced by incarnation and report
epoch, never a credit authority.
"""

from .capacity import is_feasible, outstanding_slots
from .selection import ReplicaCandidate
from .state import (
    AdmissionProfile,
    AllocationLease,
    ClaimState,
    DemandEntry,
    InvocationRequest,
    ReplicaCapacityReport,
    ReplicaIncarnation,
    ReplicaState,
    ResidentSnapshot,
    ServiceClaim,
    ServiceFamily,
)

_INERT_REPLICA_STATES: frozenset[ReplicaState] = frozenset(
    {ReplicaState.STOPPED, ReplicaState.PREEMPTED, ReplicaState.FAILED}
)


class ServiceFamilyRegistry:
    """Policy-approved, plan-derived families. Registration materializes none."""

    def __init__(self) -> None:
        self._families: dict[str, ServiceFamily] = {}

    def register(self, family: ServiceFamily) -> None:
        self._families.setdefault(family.family, family)

    def get(self, family: str) -> ServiceFamily | None:
        return self._families.get(family)

    def __contains__(self, family: str) -> bool:
        return family in self._families

    def all(self) -> list[ServiceFamily]:
        return list(self._families.values())


class ReplicaDirectory:
    """The source of truth for which replica incarnations exist and their fences."""

    def __init__(self) -> None:
        self._replicas: dict[str, ReplicaIncarnation] = {}

    def add(self, replica: ReplicaIncarnation) -> None:
        self._replicas[replica.replica_id] = replica

    def get(self, replica_id: str) -> ReplicaIncarnation | None:
        return self._replicas.get(replica_id)

    def by_family(self, family: str) -> list[ReplicaIncarnation]:
        return [r for r in self._replicas.values() if r.family == family]

    def live_by_family(self, family: str) -> list[ReplicaIncarnation]:
        return [
            r
            for r in self._replicas.values()
            if r.family == family and r.state not in _INERT_REPLICA_STATES
        ]

    def all(self) -> list[ReplicaIncarnation]:
        return list(self._replicas.values())


class DemandLedger:
    """Unadmitted-claim references the Admission controller fair-orders. Read only."""

    def __init__(self) -> None:
        self._entries: dict[str, DemandEntry] = {}

    def enqueue(self, entry: DemandEntry) -> None:
        self._entries[entry.claim_id] = entry

    def get(self, claim_id: str) -> DemandEntry | None:
        return self._entries.get(claim_id)

    def mark_admitted(self, claim_id: str) -> None:
        if (entry := self._entries.get(claim_id)) is not None:
            self._entries[claim_id] = entry.model_copy(update={"admitted": True})

    def remove(self, claim_id: str) -> None:
        self._entries.pop(claim_id, None)

    def unadmitted(self, family: str | None = None) -> list[DemandEntry]:
        """Fair order: earliest deadline first, then arrival order."""
        pending = [
            e
            for e in self._entries.values()
            if not e.admitted and (family is None or e.family == family)
        ]
        return sorted(
            pending, key=lambda e: (e.deadline_at or "~", e.created_at, e.claim_id)
        )

    def backlog(self, family: str) -> int:
        return len(self.unadmitted(family))


class LeaseStore:
    """Allocation-lease records and their lifecycle ownership."""

    def __init__(self) -> None:
        self._leases: dict[str, AllocationLease] = {}

    def add(self, lease: AllocationLease) -> None:
        self._leases[lease.lease_id] = lease

    def get(self, lease_id: str) -> AllocationLease | None:
        return self._leases.get(lease_id)

    def by_family(self, family: str) -> list[AllocationLease]:
        return [lease for lease in self._leases.values() if lease.family == family]

    def all(self) -> list[AllocationLease]:
        return list(self._leases.values())


class InvocationStore:
    """Durable ``CS`` request records keyed by ``invocation_id``."""

    def __init__(self) -> None:
        self._requests: dict[str, InvocationRequest] = {}

    def put(self, request: InvocationRequest) -> None:
        self._requests.setdefault(request.invocation_id, request)

    def get(self, invocation_id: str) -> InvocationRequest | None:
        return self._requests.get(invocation_id)


class ClaimStore:
    """Authoritative ``ServiceClaim`` facts — the sole authority for credits."""

    def __init__(self) -> None:
        self._claims: dict[str, ServiceClaim] = {}

    def add(self, claim: ServiceClaim) -> None:
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> ServiceClaim | None:
        return self._claims.get(claim_id)

    def all(self) -> list[ServiceClaim]:
        return list(self._claims.values())

    def by_invocation(self, invocation_id: str) -> list[ServiceClaim]:
        return [c for c in self._claims.values() if c.invocation_id == invocation_id]

    def credit_bearing_for_replica(self, replica_id: str) -> list[ServiceClaim]:
        return [
            c
            for c in self._claims.values()
            if c.replica_id == replica_id and c.holds_credit
        ]


class ReportStore:
    """Latest capacity report per replica, fenced by incarnation and report epoch.

    A report is evidence only. It is retained only when it does not regress the
    replica's incarnation or report epoch, so stale telemetry cannot re-open a
    superseded decision.
    """

    def __init__(self) -> None:
        self._latest: dict[str, ReplicaCapacityReport] = {}

    def ingest(self, report: ReplicaCapacityReport) -> bool:
        current = self._latest.get(report.replica_id)
        if current is not None and (
            report.incarnation < current.incarnation
            or (
                report.incarnation == current.incarnation
                and report.report_epoch < current.report_epoch
            )
        ):
            return False
        self._latest[report.replica_id] = report
        return True

    def latest(self, replica_id: str) -> ReplicaCapacityReport | None:
        return self._latest.get(replica_id)


class AdmissionCreditLedger:
    """Per-replica held-credit accounting derived from credit-bearing claims.

    Recomputed from the authoritative claims on every read; it never overwrites or
    releases a credit and cannot diverge from the facts it derives from.
    """

    def __init__(self, claims: ClaimStore) -> None:
        self._claims = claims

    def held(self, replica_id: str) -> int:
        return outstanding_slots(self._claims.credit_bearing_for_replica(replica_id))


class CapacityPools:
    """Safe-capacity grouping over the directory, reports, and outstanding credits.

    Derived and rebuildable: it selects the feasible replicas for a claim by
    intersecting live incarnations, incarnation-current reports, and headroom net of
    every outstanding credit-bearing claim. It promotes, releases, or overwrites none.
    """

    def __init__(
        self, directory: ReplicaDirectory, claims: ClaimStore, reports: ReportStore
    ) -> None:
        self._directory = directory
        self._claims = claims
        self._reports = reports

    def feasible_candidates(
        self, family: str, profile: AdmissionProfile, *, now_ts: float | None = None
    ) -> list[ReplicaCandidate]:
        candidates: list[ReplicaCandidate] = []
        for replica in self._directory.live_by_family(family):
            report = self._reports.latest(replica.replica_id)
            if report is None or report.incarnation != replica.incarnation:
                continue
            held = outstanding_slots(
                self._claims.credit_bearing_for_replica(replica.replica_id)
            )
            if is_feasible(report, profile, held, now_ts=now_ts):
                candidates.append(ReplicaCandidate(replica.replica_id, report, held))
        return candidates


class ResidentStores:
    """The bundle of authoritative ``CS`` stores and their derived views."""

    def __init__(self) -> None:
        self.families = ServiceFamilyRegistry()
        self.directory = ReplicaDirectory()
        self.demand = DemandLedger()
        self.leases = LeaseStore()
        self.invocations = InvocationStore()
        self.claims = ClaimStore()
        self.reports = ReportStore()
        self.credit_ledger = AdmissionCreditLedger(self.claims)
        self.pools = CapacityPools(self.directory, self.claims, self.reports)

    def to_snapshot(self) -> ResidentSnapshot:
        """Serialize only the authoritative ``CS`` facts for durable persistence."""
        return ResidentSnapshot(
            families=self.families.all(),
            replicas=self.directory.all(),
            leases=self.leases.all(),
            invocations=[
                request
                for claim in self.claims.all()
                if (request := self.invocations.get(claim.invocation_id)) is not None
            ],
            claims=self.claims.all(),
        )

    def load_snapshot(self, snapshot: ResidentSnapshot) -> None:
        """Rehydrate authoritative facts and rebuild the DemandLedger from pending
        claims.

        Derived views and capacity telemetry are not restored: they are recomputed on
        read
        and refreshed by live reports.
        """
        for family in snapshot.families:
            self.families.register(family)
        for replica in snapshot.replicas:
            self.directory.add(replica)
        for lease in snapshot.leases:
            self.leases.add(lease)
        for request in snapshot.invocations:
            self.invocations.put(request)
        for claim in snapshot.claims:
            self.claims.add(claim)
            if claim.state is ClaimState.PENDING:
                self._reenqueue_demand(claim)

    def _reenqueue_demand(self, claim: ServiceClaim) -> None:
        request = self.invocations.get(claim.invocation_id)
        profile = request.profile if request is not None else None
        self.demand.enqueue(
            DemandEntry(
                claim_id=claim.claim_id,
                invocation_id=claim.invocation_id,
                family=claim.family,
                tenant=profile.tenant if profile else None,
                deadline_at=profile.deadline_at if profile else None,
            )
        )
