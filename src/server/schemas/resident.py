from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..resident.state import (
    ReplicaIncarnation,
    ServiceClaim,
    ServiceFamily,
)


class ResidentFamilyInfo(BaseModel):
    family: str = Field(description="Service-family identifier.")
    engine_batch_key: str = Field(
        description="Engine and batch key a compatible replica is admitted against."
    )
    model_ref: str = Field(description="Model reference served by the family.")
    isolation: str | None = Field(
        default=None, description="Isolation requirement, if any."
    )
    selection_strategy: str = Field(
        description="Per-family replica-selection strategy."
    )
    warmth: str | None = Field(default=None, description="Warmth policy, if any.")
    created_at: str = Field(description="Family registration timestamp.")

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
    host: str | None = Field(default=None, description="Replica endpoint host.")
    port: int | None = Field(default=None, description="Replica endpoint port.")

    @classmethod
    def parse(cls, base_url: str) -> "ResidentReplicaEndpointInfo":
        parsed = urlparse(base_url)
        return cls(host=parsed.hostname, port=parsed.port)


class ResidentReplicaInfo(BaseModel):
    replica_id: str = Field(description="Replica incarnation identifier.")
    family: str = Field(description="Owning service-family identifier.")
    incarnation: int = Field(description="Monotonic incarnation fence.")
    state: str = Field(description="Replica lifecycle state.")
    healthy: bool = Field(description="Whether the replica is reporting healthy.")
    serve_task_id: str | None = Field(
        default=None, description="Backing serve task identifier."
    )
    worker_id: str | None = Field(
        default=None, description="Worker hosting the replica."
    )
    lease_id: str | None = Field(
        default=None, description="Allocation lease identifier."
    )
    endpoint: ResidentReplicaEndpointInfo | None = Field(
        default=None, description="Reachable endpoint host and port, when known."
    )
    created_at: str = Field(description="Replica creation timestamp.")
    updated_at: str = Field(description="Last state-change timestamp.")
    last_active_at: str = Field(description="Last admission-activity timestamp.")

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
    claim_id: str = Field(description="Admission-claim identifier.")
    invocation_id: str = Field(description="Linked invocation identifier.")
    family: str = Field(description="Service-family identifier.")
    admission_epoch: int = Field(description="Claim admission epoch.")
    state: str = Field(description="Claim FSM state.")
    replica_id: str | None = Field(
        default=None, description="Reserved replica incarnation, if admitted."
    )
    incarnation: int | None = Field(
        default=None, description="Fenced replica incarnation, if admitted."
    )

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
    replica_id: str = Field(description="Replica incarnation identifier.")
    held_slots: int = Field(
        description="Admission slots held, recomputed from credit-bearing claims."
    )


class ResidentClaimsView(BaseModel):
    claims: list[ResidentClaimInfo] = Field(
        default_factory=list, description="Credit-bearing admission claims."
    )
    held_credit: list[ResidentReplicaCredit] = Field(
        default_factory=list, description="Per-replica held-credit rollup."
    )
