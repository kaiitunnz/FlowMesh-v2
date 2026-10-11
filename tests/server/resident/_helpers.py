"""Shared builders for resident-capacity control tests."""

import asyncio

from server.resident import (
    AdmissionProfile,
    ReplicaCapacityReport,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    ResidentStores,
    SafeCapacityVector,
    ServiceFamily,
)
from server.resident.durability import ResidentDurability

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
