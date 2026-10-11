"""Control-state (``CS``) stores for resident-capacity control.

The Service-family registry, Replica directory, DemandLedger, Allocation leases, durable
invocation requests, and ``ServiceClaim`` facts are authoritative in their domains. Only
``ServiceClaim`` facts are authoritative for admission credits. The derived
``CapacityPools`` and Admission-credit ledger are computed from those authorities, so
they cannot diverge: no cached credit exists for a report or a provisioning decision to
overwrite or release. Capacity reports are evidence, fenced by incarnation and report
epoch, never a credit authority.

The persisted collections journal every change, and ``ResidentStores.flush`` makes the
changed entities durable in one write: each actor flushes before anything acts on what
it changed. The stores hold the live working set; a retired invocation, replica or
family leaves them only once its durable record is in place.
"""

import logging
from collections.abc import Callable, Hashable
from typing import NamedTuple

from pydantic import BaseModel

from ..utils.journal import ChangeJournal, TrackedDict
from .capacity import is_feasible, outstanding_slots
from .durability import (
    CLAIM,
    FAMILY,
    LEASE,
    REPLICA,
    REQUEST,
    MemoryResidentDurability,
    ResidentDurability,
    live_field,
)
from .selection import ReplicaCandidate
from .state import (
    INERT_REPLICA_STATES,
    AdmissionProfile,
    AllocationLease,
    ClaimState,
    DemandEntry,
    InvocationRequest,
    ReplicaCapacityReport,
    ReplicaIncarnation,
    ResidentSnapshot,
    ServiceClaim,
    ServiceFamily,
)

_logger = logging.getLogger("resident-capacity")

type _Bucket = dict[str, dict[str, None]]


def _bucket_add(index: _Bucket, key: str | None, member: str) -> None:
    if key is not None:
        index.setdefault(key, {})[member] = None


def _bucket_remove(index: _Bucket, key: str | None, member: str) -> None:
    if key is None or (bucket := index.get(key)) is None:
        return
    bucket.pop(member, None)
    if not bucket:
        del index[key]


class ServiceFamilyRegistry:
    """Policy-approved, plan-derived families. Registration materializes none."""

    def __init__(self, journal: ChangeJournal) -> None:
        self._families: TrackedDict[str, ServiceFamily] = TrackedDict(journal, FAMILY)

    def register(self, family: ServiceFamily) -> None:
        """Record a family definition, refining only an existing one's warmth.

        A family's compatibility identity is immutable, so an incoming definition
        describing a different engine, model, interface, isolation domain, or serving
        size leaves the live one untouched. Warmth refines one way: a preference
        upgrades an unset one, and no later definition demotes it.
        """
        if (existing := self._families.get(family.family)) is None:
            self._families[family.family] = family
            return
        if existing.warmth or not family.warmth:
            return
        if (
            existing.engine_batch_key != family.engine_batch_key
            or existing.model_ref != family.model_ref
            or existing.interface != family.interface
            or existing.isolation != family.isolation
            or existing.serving_size != family.serving_size
        ):
            return
        self._families[family.family] = existing.model_copy(
            update={"warmth": family.warmth}
        )

    def get(self, family: str) -> ServiceFamily | None:
        return self._families.get(family)

    def __contains__(self, family: str) -> bool:
        return family in self._families

    def all(self) -> list[ServiceFamily]:
        return list(self._families.values())

    def forget(self, family: str) -> None:
        self._families.forget(family)

    def rewrite(self, family: str) -> None:
        """Journal a family again, so the next flush writes its live field."""
        self._families.touch(family)


class ReplicaDirectory:
    """The source of truth for which replica incarnations exist and their fences."""

    def __init__(self, journal: ChangeJournal) -> None:
        self._replicas: TrackedDict[str, ReplicaIncarnation] = TrackedDict(
            journal, REPLICA
        )

    def add(self, replica: ReplicaIncarnation) -> None:
        self._replicas[replica.replica_id] = replica

    def get(self, replica_id: str) -> ReplicaIncarnation | None:
        return self._replicas.get(replica_id)

    def by_family(self, family: str) -> list[ReplicaIncarnation]:
        return [r for r in self._replicas.values() if r.family == family]

    def by_serve_task(self, serve_task_id: str) -> list[ReplicaIncarnation]:
        return [r for r in self._replicas.values() if r.serve_task_id == serve_task_id]

    def live_by_family(self, family: str) -> list[ReplicaIncarnation]:
        return [
            r
            for r in self._replicas.values()
            if r.family == family and r.state not in INERT_REPLICA_STATES
        ]

    def all(self) -> list[ReplicaIncarnation]:
        return list(self._replicas.values())

    def forget(self, replica_id: str) -> None:
        self._replicas.forget(replica_id)


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


class LeaseStore:
    """Allocation-lease records and their lifecycle ownership."""

    def __init__(self, journal: ChangeJournal) -> None:
        self._leases: TrackedDict[str, AllocationLease] = TrackedDict(journal, LEASE)

    def add(self, lease: AllocationLease) -> None:
        self._leases[lease.lease_id] = lease

    def get(self, lease_id: str) -> AllocationLease | None:
        return self._leases.get(lease_id)

    def by_family(self, family: str) -> list[AllocationLease]:
        return [lease for lease in self._leases.values() if lease.family == family]

    def all(self) -> list[AllocationLease]:
        return list(self._leases.values())

    def forget(self, lease_id: str) -> None:
        if lease_id in self._leases:
            self._leases.forget(lease_id)


class InvocationStore:
    """Durable ``CS`` request records keyed by ``invocation_id``."""

    def __init__(self, journal: ChangeJournal) -> None:
        self._requests: TrackedDict[str, InvocationRequest] = TrackedDict(
            journal, REQUEST
        )

    def put(self, request: InvocationRequest) -> None:
        self._requests.setdefault(request.invocation_id, request)

    def replace(self, request: InvocationRequest) -> None:
        self._requests[request.invocation_id] = request

    def get(self, invocation_id: str) -> InvocationRequest | None:
        return self._requests.get(invocation_id)

    def all(self) -> list[InvocationRequest]:
        return list(self._requests.values())

    def any_for_family(self, family: str) -> bool:
        return any(request.family == family for request in self._requests.values())

    def forget(self, invocation_id: str) -> None:
        if invocation_id in self._requests:
            self._requests.forget(invocation_id)


class _Filing(NamedTuple):
    """The index keys a claim is filed under, ``None`` where it is in no bucket."""

    invocation: str
    replica: str | None
    credit_replica: str | None
    pending_family: str | None


def _filing(claim: ServiceClaim) -> _Filing:
    return _Filing(
        invocation=claim.invocation_id,
        replica=claim.replica_id,
        credit_replica=claim.replica_id if claim.holds_credit else None,
        pending_family=claim.family if claim.state is ClaimState.PENDING else None,
    )


class ClaimStore:
    """Authoritative ``ServiceClaim`` facts — the sole authority for credits.

    Each claim is indexed by invocation, by the replica it names, by the replica it
    holds credit on, and by the family it is pending for, so a lookup reads only the
    claims it asks about.
    """

    def __init__(self, journal: ChangeJournal) -> None:
        self._claims: TrackedDict[str, ServiceClaim] = TrackedDict(
            journal, CLAIM, on_change=self._reindex
        )
        self._by_invocation: _Bucket = {}
        self._by_replica: _Bucket = {}
        self._credit_by_replica: _Bucket = {}
        self._pending_by_family: _Bucket = {}
        # In the order of a filing's keys.
        self._indexes = (
            self._by_invocation,
            self._by_replica,
            self._credit_by_replica,
            self._pending_by_family,
        )
        self._filed: dict[str, _Filing] = {}

    def _reindex(self, key: Hashable) -> None:
        claim_id = str(key)
        if (old := self._filed.pop(claim_id, None)) is not None:
            for index, index_key in zip(self._indexes, old, strict=True):
                _bucket_remove(index, index_key, claim_id)
        if (claim := self._claims.get(claim_id)) is None:
            return
        self._filed[claim_id] = filing = _filing(claim)
        for index, index_key in zip(self._indexes, filing, strict=True):
            _bucket_add(index, index_key, claim_id)

    def _members(self, index: _Bucket, key: str) -> list[ServiceClaim]:
        return [self._claims[claim_id] for claim_id in index.get(key, ())]

    def add(self, claim: ServiceClaim) -> None:
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> ServiceClaim | None:
        return self._claims.get(claim_id)

    def all(self) -> list[ServiceClaim]:
        return list(self._claims.values())

    def by_invocation(self, invocation_id: str) -> list[ServiceClaim]:
        return self._members(self._by_invocation, invocation_id)

    def by_replica(self, replica_id: str) -> list[ServiceClaim]:
        """List every live claim naming the replica, terminal ones included."""
        return self._members(self._by_replica, replica_id)

    def credit_bearing_for_replica(self, replica_id: str) -> list[ServiceClaim]:
        return self._members(self._credit_by_replica, replica_id)

    def credit_bearing(self) -> list[ServiceClaim]:
        return [
            self._claims[claim_id]
            for bucket in self._credit_by_replica.values()
            for claim_id in bucket
        ]

    def pending_for_family(self, family: str) -> list[ServiceClaim]:
        return self._members(self._pending_by_family, family)

    def forget(self, claim_id: str) -> None:
        self._claims.forget(claim_id)


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

    def forget(self, replica_id: str) -> None:
        self._latest.pop(replica_id, None)


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
    """The bundle of authoritative ``CS`` stores, their derived views, and the
    durability their changes flush to."""

    def __init__(self, durability: ResidentDurability | None = None) -> None:
        self.journal = ChangeJournal()
        self.durability: ResidentDurability = (
            durability if durability is not None else MemoryResidentDurability()
        )
        # Whether the stores hold every live fact; false while a restart loads them.
        self.complete = True
        self.families = ServiceFamilyRegistry(self.journal)
        self.directory = ReplicaDirectory(self.journal)
        self.demand = DemandLedger()
        self.leases = LeaseStore(self.journal)
        self.invocations = InvocationStore(self.journal)
        self.claims = ClaimStore(self.journal)
        self.reports = ReportStore()
        self.credit_ledger = AdmissionCreditLedger(self.claims)
        self.pools = CapacityPools(self.directory, self.claims, self.reports)
        self._readers: dict[str, Callable[[str], BaseModel | None]] = {
            FAMILY: self.families.get,
            REPLICA: self.directory.get,
            LEASE: self.leases.get,
            REQUEST: self.invocations.get,
            CLAIM: self.claims.get,
        }

    def flush(self) -> None:
        """Make every change since the last flush durable in one write.

        Call before acting on a change. A failed write raises and leaves its changes
        pending for the next flush.
        """
        if not (captured := self.journal.captured()):
            return
        changes: dict[str, str | None] = {}
        for kind, key in captured:
            entity = self._readers[kind](str(key))
            changes[live_field(kind, str(key))] = (
                entity.model_dump_json() if entity is not None else None
            )
        self.durability.write(changes)
        self.journal.clear(captured)

    @property
    def durable(self) -> bool:
        """Report whether every change has been made durable."""
        return not self.journal.pending

    @property
    def invocations_durable(self) -> bool:
        """Report whether every request and claim change has been made durable."""
        return not any(kind in (REQUEST, CLAIM) for kind, _ in self.journal.pending)

    def live(self, invocation_id: str) -> bool:
        """Report whether the invocation has a request or a claim in the working set."""
        return self.invocations.get(invocation_id) is not None or bool(
            self.claims.by_invocation(invocation_id)
        )

    def forget_invocation(self, invocation_id: str) -> None:
        """Drop a retired invocation's claims, demand and request from the stores."""
        for claim in self.claims.by_invocation(invocation_id):
            self.demand.remove(claim.claim_id)
            self.claims.forget(claim.claim_id)
        self.invocations.forget(invocation_id)

    def forget_replica(self, replica: ReplicaIncarnation) -> None:
        """Drop a retired replica, its lease and its report from the stores."""
        if replica.lease_id is not None:
            self.leases.forget(replica.lease_id)
        self.directory.forget(replica.replica_id)
        self.reports.forget(replica.replica_id)

    def register_family(self, family: ServiceFamily) -> None:
        """Register a family definition, first restoring a retired standing family's
        stored one so it keeps its identity and creation time.

        A history that cannot be read registers the incoming definition.
        """
        if family.standing and family.family not in self.families:
            try:
                stored = self.durability.stored_family(family.family)
            except Exception:
                _logger.warning(
                    "reading resident family %s from history failed",
                    family.family,
                    exc_info=True,
                )
            else:
                if stored is not None:
                    self.families.register(stored)
        self.families.register(family)

    def load_snapshot(self, snapshot: ResidentSnapshot) -> None:
        """Install the stored live facts and rebuild the DemandLedger from pending
        claims.

        Derived views and capacity telemetry are recomputed on read and refreshed by
        live reports. What is loaded is already durable, so it leaves nothing to flush.
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
        self.journal.reset()

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
