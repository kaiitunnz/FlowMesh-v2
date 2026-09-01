from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..resident.state import (
    ReplicaIncarnation,
    ServiceClaim,
    ServiceFamily,
)


class ResidentFamilyInfo(BaseModel):
    family: str
    engine_batch_key: str
    model_ref: str
    isolation: str | None = None
    selection_strategy: str
    warmth: str | None = None
    created_at: str

    @classmethod
    def project(cls, family: ServiceFamily) -> "ResidentFamilyInfo":
        return cls(
            family=family.family,
            engine_batch_key=family.engine_batch_key,
            model_ref=family.model_ref,
            isolation=family.isolation,
            selection_strategy=family.selection_strategy,
            warmth=family.warmth,
            created_at=family.created_at,
        )


class ResidentReplicaEndpointInfo(BaseModel):
    host: str | None = None
    port: int | None = None

    @classmethod
    def parse(cls, base_url: str) -> "ResidentReplicaEndpointInfo":
        parsed = urlparse(base_url)
        return cls(host=parsed.hostname, port=parsed.port)


class ResidentReplicaInfo(BaseModel):
    replica_id: str
    family: str
    incarnation: int
    state: str
    healthy: bool
    serve_task_id: str | None = None
    worker_id: str | None = None
    lease_id: str | None = None
    endpoint: ResidentReplicaEndpointInfo | None = None
    created_at: str
    updated_at: str
    last_active_at: str

    @classmethod
    def project(cls, replica: ReplicaIncarnation) -> "ResidentReplicaInfo":
        endpoint = (
            ResidentReplicaEndpointInfo.parse(replica.endpoint.base_url)
            if replica.endpoint is not None
            else None
        )
        return cls(
            replica_id=replica.replica_id,
            family=replica.family,
            incarnation=replica.incarnation,
            state=replica.state.value,
            healthy=replica.healthy,
            serve_task_id=replica.serve_task_id,
            worker_id=replica.worker_id,
            lease_id=replica.lease_id,
            endpoint=endpoint,
            created_at=replica.created_at,
            updated_at=replica.updated_at,
            last_active_at=replica.last_active_at,
        )


class ResidentClaimInfo(BaseModel):
    claim_id: str
    invocation_id: str
    family: str
    admission_epoch: int
    state: str
    replica_id: str | None = None
    incarnation: int | None = None

    @classmethod
    def project(cls, claim: ServiceClaim) -> "ResidentClaimInfo":
        return cls(
            claim_id=claim.claim_id,
            invocation_id=claim.invocation_id,
            family=claim.family,
            admission_epoch=claim.admission_epoch,
            state=claim.state.value,
            replica_id=claim.replica_id,
            incarnation=claim.incarnation,
        )


class ResidentReplicaCredit(BaseModel):
    replica_id: str
    held_slots: int


class ResidentClaimsView(BaseModel):
    claims: list[ResidentClaimInfo] = Field(default_factory=list)
    held_credit: list[ResidentReplicaCredit] = Field(default_factory=list)
