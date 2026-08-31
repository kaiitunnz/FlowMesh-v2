"""Durable control facts and runtime objects for resident-capacity control.

These are the ``CS`` (control-state) stores and objects that back the admission and
lifecycle actors. They link to the semantic ledger (``DS``) only through the stable
``invocation_id`` and fenced terminal outcomes; neither side infers or overwrites the
other's facts. ``ServiceClaim`` facts are the sole authority for admission credits;
capacity pools and the credit ledger are derived, rebuildable views.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..utils.time import now_iso


class ClaimState(StrEnum):
    """The causal state of one admission claim.

    ``PENDING`` and ``TERMINAL`` hold no credit; every other state is a
    credit-bearing nonterminal fact. ``UNCERTAIN`` and ``RECONCILING`` retain the
    credit after a route/incarnation loss until a fenced terminal outcome settles it.
    """

    PENDING = "pending"
    RESERVED = "reserved"
    ACCEPTED = "accepted"
    STREAMING = "streaming"
    UNCERTAIN = "uncertain"
    RECONCILING = "reconciling"
    TERMINAL = "terminal"


CREDIT_BEARING_CLAIM_STATES: frozenset[ClaimState] = frozenset(
    {
        ClaimState.RESERVED,
        ClaimState.ACCEPTED,
        ClaimState.STREAMING,
        ClaimState.UNCERTAIN,
        ClaimState.RECONCILING,
    }
)


class ClaimTerminalReason(StrEnum):
    """Why a claim reached ``TERMINAL``.

    ``COMPLETED`` is the settled semantic outcome consumed from ``DS``; the rest are
    pre-acceptance exits or a reconciled loss. None reopens a terminal claim.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ENQUEUE_FAILED = "enqueue_failed"
    EXPIRED = "expired"
    RECONCILED = "reconciled"


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


class AdmissionProfile(BaseModel):
    """The compatibility and SLO shape a claim is admitted against.

    The engine/batch key identifies a compatible model runner and configuration; it is
    stricter than broad reuse. ``adapter_ref`` names an adapter delta (a LoRA) that
    constrains a compatible base-model replica's adapter slot rather than crossing base
    versions.
    """

    model_config = ConfigDict(frozen=True)

    engine_batch_key: str
    isolation_domain: str | None = None
    tenant: str | None = None
    slo_class: str | None = None
    deadline_at: str | None = None
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    adapter_ref: str | None = None
    cache_affinity_hint: str | None = None


class ClaimCredit(BaseModel):
    """The conservative demand a reserved claim debits from a replica's safe capacity.

    It is an admission accounting estimate, not a claim on literal engine KV blocks:
    one admission slot plus a projected token demand.
    """

    model_config = ConfigDict(frozen=True)

    slots: int = 1
    projected_tokens: int | None = None


class SafeCapacityVector(BaseModel):
    """A replica's conservative safe-admission headroom.

    It is per-family, per-hardware, and per-SLO-class, not a scalar running-set
    threshold. ``admission_slots`` is the calibrated conservative slot count the stock
    adapter reports; the token/sequence/KV fields tighten it where an engine exposes
    them.
    """

    model_config = ConfigDict(frozen=True)

    admission_slots: int
    running_sequences: int | None = None
    scheduled_tokens: int | None = None
    kv_headroom: float | None = None


class ReplicaEndpoint(BaseModel):
    """The reachable address of a materialized replica.

    ``api_key`` stays server-side; it is materialized into the admission handoff and
    never surfaced to a workflow. ``base_url`` is OpenAI-compatible for the inference
    family.
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key: str | None = None
    protocol: str = "openai"


class ReplicaCapacityReport(BaseModel):
    """Engine-adapter-normalized evidence about one replica incarnation.

    It carries the replica incarnation and report epoch so a stale report cannot fence
    a newer decision. It may tighten the safe-capacity budget and reconciliation
    evidence, but never creates, overwrites, or releases a claim credit.
    """

    model_config = ConfigDict(frozen=True)

    replica_id: str
    incarnation: int
    report_epoch: int
    state: ReplicaState
    healthy: bool
    running: int = 0
    waiting: int = 0
    token_backlog: int | None = None
    safe: SafeCapacityVector
    ttft_ms: float | None = None
    itl_ms: float | None = None
    preemption_pressure: float | None = None
    adapter_slots_free: int | None = None
    cache_affinity: dict[str, str] = Field(default_factory=dict)
    at: str = Field(default_factory=now_iso)


class ServiceFamily(BaseModel):
    """A policy-approved, plan-derived service-family definition.

    Registration alone materializes no capacity and carries no credit. It records the
    engine/batch, isolation, resource, and protocol requirements a later eligible claim
    is admitted against, and the per-family replica-selection strategy.
    """

    model_config = ConfigDict(frozen=True)

    family: str
    engine_batch_key: str
    model_ref: str
    isolation: str | None = None
    selection_strategy: str = "batch-aware-best-fit"
    warmth: str | None = None
    created_at: str = Field(default_factory=now_iso)


class ReplicaIncarnation(BaseModel):
    """A live replica incarnation in the Replica directory.

    ``incarnation`` is the monotonic fence: a lost or recreated replica invalidates
    outstanding routes and admission decisions bound to an older incarnation. The
    backing serve task and worker locate the generically hosted allocation.
    """

    replica_id: str
    family: str
    incarnation: int
    state: ReplicaState = ReplicaState.MATERIALIZING
    endpoint: ReplicaEndpoint | None = None
    healthy: bool = False
    serve_task_id: str | None = None
    worker_id: str | None = None
    lease_id: str | None = None
    report_epoch: int = 0
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


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


class InvocationRequest(BaseModel):
    """The durable ``CS`` request record keyed by ``invocation_id``.

    ``DS`` retains the same identity's causal linkage and terminal semantics; this
    record holds the admission profile and request context. Retries reuse this identity.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    workflow_id: str
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
    report_epoch: int | None = None
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
    replicas: list[ReplicaIncarnation] = Field(default_factory=list)
    leases: list[AllocationLease] = Field(default_factory=list)
    invocations: list[InvocationRequest] = Field(default_factory=list)
    claims: list[ServiceClaim] = Field(default_factory=list)


class AdmissionHandoff(BaseModel):
    """An opaque, short-lived, claim-bound execution handoff.

    A ``RESERVED`` claim authorizes exactly this: a trusted, claim-bound descriptor an
    engine adapter consumes to reach the selected replica incarnation and obtain an
    enqueue acknowledgement. It is locality-neutral — the same descriptor can be
    consumed in-server or by an authenticated worker-side deputy — and is neither a
    data-plane route nor a post-acceptance route authorization.
    """

    model_config = ConfigDict(frozen=True)

    token: str
    claim_id: str
    invocation_id: str
    family: str
    replica_id: str
    incarnation: int
    endpoint: ReplicaEndpoint
    deadline_at: str | None = None
