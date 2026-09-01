"""Shared builders for resident-capacity control tests."""

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

PROFILE = AdmissionProfile(engine_batch_key="fam")


def warm_stores(*, slots: int = 2, replica_id: str = "rpl-1") -> ResidentStores:
    """Stores with one warm, endpoint-bound replica reporting ``slots`` safe slots."""
    stores = ResidentStores()
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
