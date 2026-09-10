"""Durable control facts and runtime objects for resident-capacity control.

These are the ``CS`` (control-state) stores and objects that back the admission and
lifecycle actors. They link to the semantic ledger (``DS``) only through the stable
``invocation_id`` and fenced terminal outcomes; neither side infers or overwrites the
other's facts. ``ServiceClaim`` facts are the sole authority for admission credits;
capacity pools and the credit ledger are derived, rebuildable views.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shared.resident.contracts import ReplicaEndpoint

from ..network.state import ReplicaListenerAdvertisement
from ..utils.time import now_iso


class ClaimState(StrEnum):
    """The causal state of one admission claim.

    ``PENDING`` and ``TERMINAL`` hold no credit; every other state is a credit-bearing
    nonterminal fact. ``UNCERTAIN`` retains the credit after a route/incarnation loss
    until a fenced terminal outcome settles it.
    """

    PENDING = "pending"
    RESERVED = "reserved"
    ACCEPTED = "accepted"
    STREAMING = "streaming"
    UNCERTAIN = "uncertain"
    TERMINAL = "terminal"


CREDIT_BEARING_CLAIM_STATES: frozenset[ClaimState] = frozenset(
    {
        ClaimState.RESERVED,
        ClaimState.ACCEPTED,
        ClaimState.STREAMING,
        ClaimState.UNCERTAIN,
    }
)


class ClaimTerminalReason(StrEnum):
    """Why a claim reached ``TERMINAL``.

    ``COMPLETED`` is the settled semantic outcome consumed from ``DS``; the rest are
    pre-acceptance exits or a settled loss. None reopens a terminal claim.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ENQUEUE_FAILED = "enqueue_failed"
    EXPIRED = "expired"


class ReplicaState(StrEnum):
    """The lifecycle state of one leased replica incarnation."""

    ABSENT = "absent"
    MATERIALIZING = "materializing"
    WARM = "warm"
    BUSY = "busy"
    DRAINING = "draining"
    STOPPED = "stopped"
    PREEMPTED = "preempted"
    FAILED = "failed"


SERVABLE_REPLICA_STATES: frozenset[ReplicaState] = frozenset(
    {ReplicaState.WARM, ReplicaState.BUSY}
)


class ProvisioningDenialReason(StrEnum):
    """A typed reason an auto-materialization or admission is refused by policy."""

    MODEL_NOT_ALLOWED = "model_not_allowed"
    ISOLATION_DENIED = "isolation_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    RESOURCE_CAP = "resource_cap"
    EGRESS_CAP = "egress_cap"
    COLD_START_LIMIT = "cold_start_limit"
    COLD_START_BUDGET = "cold_start_budget"
    ADAPTER_SLOT_CAP = "adapter_slot_cap"


class AdmissionProfile(BaseModel):
    """The compatibility and SLO shape a claim is admitted against.

    The engine/batch key identifies a compatible model runner and configuration; it is
    stricter than broad reuse. ``adapter_ref`` names an adapter delta (a LoRA) that
    constrains a compatible base-model replica's adapter slot rather than crossing base
    versions; ``adapter_source`` is the loadable source the replica loads for it.
    """

    model_config = ConfigDict(frozen=True)

    engine_batch_key: str
    tenant: str | None = None
    deadline_at: str | None = None
    max_output_tokens: int | None = None
    adapter_ref: str | None = None
    adapter_source: str | None = None
    serve_task_id: str | None = None
    binding_generation: int | None = None
    descriptor_digest: str | None = None


class ClaimCredit(BaseModel):
    """The conservative demand a reserved claim debits from a replica's safe capacity.

    It is an admission accounting estimate, not a claim on literal engine KV blocks: one
    admission slot plus a projected token demand.
    """

    model_config = ConfigDict(frozen=True)

    slots: int = 1
    projected_tokens: int | None = None


class SafeCapacityVector(BaseModel):
    """A replica's conservative safe-admission headroom.

    ``admission_slots`` is the calibrated conservative slot count the stock adapter
    reports — a per-family, per-hardware slot budget rather than a scalar running-set
    threshold. It bounds how many invocations may hold the replica at once, which is
    admission, not execution: a sandbox host's sessions each hold a slot while open and
    their commands still run one at a time on its worker.
    """

    model_config = ConfigDict(frozen=True)

    admission_slots: int


class ReplicaCapacityReport(BaseModel):
    """Engine-adapter-normalized evidence about one replica incarnation.

    It carries the replica incarnation and report epoch so a stale report cannot fence a
    newer decision. It may tighten the safe-capacity budget, but never creates,
    overwrites, or releases a claim credit. ``adapter_slots_free`` constrains an
    adapter-scoped claim's feasibility (see ``AdmissionProfile.adapter_ref``), and
    ``held_adapters`` names the adapters already resident so a claim for one of them
    shares its slot rather than being denied at exhaustion.
    """

    model_config = ConfigDict(frozen=True)

    replica_id: str
    incarnation: int
    report_epoch: int
    state: ReplicaState
    healthy: bool
    safe: SafeCapacityVector
    adapter_slots_free: int | None = None
    held_adapters: tuple[str, ...] = ()
    at: str = Field(default_factory=now_iso)


class ServiceFamilyKind(StrEnum):
    """The substrate a family's replicas are materialized as."""

    MODEL_SERVING = "model_serving"
    SANDBOX_HOST = "sandbox_host"


class ServiceFamily(BaseModel):
    """A policy-approved, plan-derived service-family definition.

    Registration alone materializes no capacity and carries no credit. It records the
    engine/batch, isolation, resource, and protocol requirements a later eligible claim
    is admitted against, and the per-family replica-selection strategy.
    """

    model_config = ConfigDict(frozen=True)

    family: str
    engine_batch_key: str
    service_ref: str
    kind: ServiceFamilyKind = ServiceFamilyKind.MODEL_SERVING
    interface: str = "chat"
    isolation: str | None = None
    selection_strategy: str = "batch-aware-best-fit"
    warmth: str | None = None
    created_at: str = Field(default_factory=now_iso)


class ReplicaIncarnation(BaseModel):
    """A live replica incarnation in the Replica directory.

    ``incarnation`` is the monotonic fence: a lost or recreated replica invalidates
    outstanding routes and admission decisions bound to an older incarnation. The
    backing serve task and worker locate the generically hosted allocation. ``listener``
    is the non-secret resident-facing route advertisement, fenced by ``incarnation`` and
    ``listener_generation``; it never names the raw engine listener or credential.
    """

    replica_id: str
    family: str
    incarnation: int
    state: ReplicaState = ReplicaState.MATERIALIZING
    endpoint: ReplicaEndpoint | None = None
    listener: ReplicaListenerAdvertisement | None = None
    listener_generation: int = 0
    healthy: bool = False
    serve_task_id: str | None = None
    binding_generation: int | None = None
    standing: bool = False
    worker_id: str | None = None
    lease_id: str | None = None
    report_epoch: int = 0
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    last_active_at: str = Field(default_factory=now_iso)


class MaterializedAllocation(BaseModel):
    """What one materialization produced for a replica.

    A model-serving replica names the serve task backing it; a sandbox host names the
    worker whose sandbox capacity it reserves.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str | None = None
    worker_id: str | None = None


class NoSandboxCapacity(Exception):
    """No worker can hold a sandbox-host reservation."""


class AllocationLease(BaseModel):
    """Lease record and lifecycle ownership for one resident replica."""

    lease_id: str
    family: str
    replica_id: str
    state: ReplicaState = ReplicaState.MATERIALIZING
    cold_start_deadline_at: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class DemandEntry(BaseModel):
    """A pending-claim reference the Admission controller fair-orders.

    The DemandLedger is read state for fair ordering and scale signals; it never
    promotes a claim itself.
    """

    claim_id: str
    invocation_id: str
    family: str
    tenant: str | None = None
    deadline_at: str | None = None
    admitted: bool = False
    created_at: str = Field(default_factory=now_iso)


class InvocationSubjectKind(StrEnum):
    """Who owns an invocation."""

    WORKFLOW = "workflow"
    EXTERNAL = "external"


class InvocationSubject(BaseModel):
    """The tenant-scoped owner of an invocation.

    A workflow subject links the request to its submitting workflow instance and
    settles its terminal in ``DS``; an external subject is an authenticated external
    principal reaching a task-addressed gated serve surface, which settles its terminal
    as a durable external status fact. The tenant scopes admission and the claim gate
    without requiring a workflow activation.
    """

    model_config = ConfigDict(frozen=True)

    kind: InvocationSubjectKind
    id: str
    tenant: str | None = None


class InvocationRequest(BaseModel):
    """The durable ``CS`` request record keyed by ``invocation_id``.

    It holds the admission profile, the tenant-scoped subject, and the request context.
    A workflow subject links to ``DS`` only by ``invocation_id`` and fenced outcomes; an
    external subject has no ``DS`` state. Retries reuse this identity.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    subject: InvocationSubject
    family: str
    profile: AdmissionProfile
    replayable: bool = True
    created_at: str = Field(default_factory=now_iso)


class ServiceClaim(BaseModel):
    """The authoritative per-admission fact and causal FSM for one credit.

    Linked to its invocation by ``invocation_id`` and carrying a fresh ``claim_id`` and
    ``admission_epoch``. Its credit-bearing nonterminal states are the sole authority
    for a replica's held admission credit; a permitted reissue is a fresh successor
    claim under the same ``invocation_id``, never a reopening of a terminal one.
    """

    claim_id: str
    invocation_id: str
    family: str
    admission_epoch: int
    state: ClaimState = ClaimState.PENDING
    credit: ClaimCredit | None = None
    replica_id: str | None = None
    incarnation: int | None = None
    terminal_reason: ClaimTerminalReason | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    @property
    def holds_credit(self) -> bool:
        return self.state in CREDIT_BEARING_CLAIM_STATES


class ResidentSnapshot(BaseModel):
    """The durable aggregate of the authoritative resident-capacity control facts.

    It persists exactly the authoritative ``CS`` stores; derived views and telemetry are
    rebuilt on load. The DemandLedger is reconstructed from the pending claims and their
    invocation requests, so no separate demand persistence is needed.
    """

    families: list[ServiceFamily] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_unreadable_families(cls, data: Any) -> Any:
        """Load the snapshot without any family definition it cannot read.

        A family definition is re-derived from the next eligible demand, so dropping one
        an older snapshot wrote costs a cold start; refusing the whole snapshot would
        instead strand the credit-bearing claims it carries.
        """
        if not isinstance(data, dict):
            return data
        if not isinstance(families := data.get("families"), list):
            return data
        readable = []
        for entry in families:
            try:
                ServiceFamily.model_validate(entry)
            except ValidationError:
                continue
            readable.append(entry)
        return {**data, "families": readable}

    replicas: list[ReplicaIncarnation] = Field(default_factory=list)
    leases: list[AllocationLease] = Field(default_factory=list)
    invocations: list[InvocationRequest] = Field(default_factory=list)
    claims: list[ServiceClaim] = Field(default_factory=list)
