from pydantic import BaseModel, Field

from ..services.private_state_inventory import SealedGenerationEntry


class SealedComponentInfo(BaseModel):
    kind: str = Field(description="Registered component kind.")
    schema_version: int = Field(description="Component schema version.")
    content_digest: str = Field(description="Digest of the component's sealed tree.")
    size_bytes: int = Field(description="Total sealed bytes.")
    entry_count: int = Field(description="Files covered by the seal.")


class SealedGenerationInfo(BaseModel):
    """One sealed generation, its holder fence, and its policy decision.

    Describes a generation; it carries no state bytes and no credential.
    """

    reference_id: str = Field(description="Opaque private-state reference.")
    instance_id: str = Field(description="Workflow instance owning the lineage.")
    activation_id: str = Field(description="Activation owning the lineage.")
    owner_id: str = Field(description="Principal the workflow runs as.")
    org_id: str = Field(description="Organization the workflow runs under.")
    tenant: str | None = Field(default=None, description="Tenant of the lineage.")
    profile: str = Field(description="Bundle profile of the sealed components.")
    generation: int = Field(description="Sealed generation number.")
    owner_worker_id: str = Field(description="Holder that sealed the generation.")
    owner_incarnation: int = Field(description="Holder incarnation fence.")
    sealed_at: str | None = Field(default=None, description="Seal timestamp.")
    attached: bool = Field(description="Whether a holder currently writes the lineage.")
    resumable: bool = Field(description="Whether work could resume on it.")
    exportable: bool = Field(
        description="Whether every component may materialize off its holder."
    )
    components: list[SealedComponentInfo] = Field(
        default_factory=list, description="Sealed components by content."
    )
    decision: str | None = Field(
        default=None, description="Advisory state-control verb."
    )
    decision_reason: str | None = Field(
        default=None, description="Why the policy decided it."
    )

    @classmethod
    def project(cls, entry: SealedGenerationEntry) -> "SealedGenerationInfo":
        evidence = entry.evidence
        return cls(
            reference_id=evidence.reference_id,
            instance_id=evidence.instance_id,
            activation_id=evidence.activation_id,
            owner_id=evidence.owner_id,
            org_id=evidence.org_id,
            tenant=evidence.tenant,
            profile=evidence.profile.value,
            generation=evidence.generation,
            owner_worker_id=evidence.owner_worker_id,
            owner_incarnation=evidence.owner_incarnation,
            sealed_at=evidence.sealed_at,
            attached=evidence.attached,
            resumable=evidence.resumable,
            exportable=evidence.exportable,
            components=[
                SealedComponentInfo(
                    kind=component.kind.value,
                    schema_version=component.schema_version,
                    content_digest=component.content_digest,
                    size_bytes=component.size_bytes,
                    entry_count=component.entry_count,
                )
                for component in evidence.components
            ],
            decision=entry.decision.verb.value if entry.decision else None,
            decision_reason=entry.decision.reason if entry.decision else None,
        )
