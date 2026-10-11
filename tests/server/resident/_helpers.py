"""Shared builders for resident-capacity control tests."""

import asyncio

from server.resident import (
    AdmissionProfile,
    ReplicaCapacityReport,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    ResidentSnapshot,
    ResidentStores,
    SafeCapacityVector,
    ServiceFamily,
)
from server.resident.durability import MemoryResidentDurability, ResidentDurability

PROFILE = AdmissionProfile(engine_batch_key="fam")


def warm_stores(
    *,
    slots: int = 2,
    replica_id: str = "rpl-1",
    durability: ResidentDurability | None = None,
) -> ResidentStores:
    """Stores with one warm, endpoint-bound replica reporting ``slots`` safe slots."""
    stores = ResidentStores(durability)
    stores.families.register(
        ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")
    )
    stores.directory.add(
        ReplicaIncarnation(
            replica_id=replica_id,
            family="fam",
            incarnation=1,
            state=ReplicaState.WARM,
            healthy=True,
            endpoint=ReplicaEndpoint(base_url="http://replica", model="m"),
        )
    )
    stores.reports.ingest(
        ReplicaCapacityReport(
            replica_id=replica_id,
            incarnation=1,
            report_epoch=1,
            state=ReplicaState.WARM,
            healthy=True,
            safe=SafeCapacityVector(admission_slots=slots),
        )
    )
    return stores


def stored_replica(stores: ResidentStores, replica_id: str) -> ReplicaIncarnation:
    """The replica the stores moved to stored history."""
    stored = asyncio.run(stores.durability.stored_replicas_async())
    return next(e.replica for e in stored if e.replica.replica_id == replica_id)


def durability_holding(snapshot: ResidentSnapshot) -> MemoryResidentDurability:
    """A durability whose live facts are exactly ``snapshot``'s."""
    stores = ResidentStores(durability := MemoryResidentDurability())
    for family in snapshot.families:
        stores.families.register(family.model_copy())
    for replica in snapshot.replicas:
        stores.directory.add(replica.model_copy())
    for lease in snapshot.leases:
        stores.leases.add(lease.model_copy())
    for request in snapshot.invocations:
        stores.invocations.put(request.model_copy())
    for claim in snapshot.claims:
        stores.claims.add(claim.model_copy())
    stores.flush()
    return durability


def stored_snapshot(stores: ResidentStores) -> ResidentSnapshot:
    """Flush the stores and load their live facts back, as a restart does."""
    stores.flush()
    snapshot = asyncio.run(stores.durability.load_snapshot_async())
    assert snapshot is not None
    return snapshot
