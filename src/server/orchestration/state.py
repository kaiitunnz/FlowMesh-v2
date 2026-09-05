"""Durable orchestration-ledger (`DS`) state objects.

These model the durable semantic hierarchy: a workflow instance owns result
slots and activations; an activation owns records, continuations, work items, and its
scope's progress capabilities; a work item owns an optional invocation and one or more
physical attempts. The ledger persists grant/policy identity and authorization
decisions, never bearer credentials, and links to the separate control facts (`CS`)
only through a stable ``invocation_id``.

Structured dynamic regions populate the lineage fields the acyclic subset leaves at
their defaults: a child ``Scope.parent_scope_id`` and ``depth``, an ``Activation``'s
``kind``/``loop_time``/``child_index``, a grant's ``delegate`` face and ``epoch``, and
per-scope ``ProgressCapability`` accounting on the child-init and loop-time axes.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from shared.harness.boundary import DenialKind
from shared.outcome import OutcomeManifest

from ..task.v2.representations.operators import (
    BoundaryEventKind,
    EffectClass,
    EffectReplayContract,
    ModelRef,
    RecoveryClass,
)
from ..utils.time import now_iso


class WorkItemStatus(StrEnum):
    """Lifecycle of a stable semantic work item."""

    BLOCKED = "blocked"  # predecessors not settled
    READY = "ready"  # admissible for a physical attempt
    DISPATCHED = "dispatched"  # an attempt is in flight
    SETTLED = "settled"  # terminal, carries a settled outcome
    CANCELLED = "cancelled"  # terminal, withdrawn by cancellation


class AttemptStatus(StrEnum):
    """Lifecycle of one physical execution of a work item."""

    ISSUED = "issued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LOST = "lost"  # worker/route loss without a terminal receipt


class InvocationState(StrEnum):
    """Durable outcome state of a work item's request identity.

    ``ACKNOWLEDGED`` is the generic engine/worker enqueue acknowledgement, distinct
    from a resident ``ServiceClaim.ACCEPTED`` (a `CS` control fact).
    """

    UNISSUED = "unissued"
    ISSUED = "issued"
    ACKNOWLEDGED = "acknowledged"
    TERMINAL = "terminal"
    UNCERTAIN = "uncertain"
    COMPENSATION_REQUIRED = "compensation_required"  # compensable effect left uncertain
    AMBIGUITY_TERMINAL = "ambiguity_terminal"


class PublicationOutcome(StrEnum):
    """Terminal outcome of a declared logical output."""

    SUCCESS = "success"
    EXPLICIT_EMPTY = "explicit_empty"
    DECLARED_FAILURE = "declared_failure"


class RecoveryDisposition(StrEnum):
    """Whether a settled work item may be recomputed or must be restored."""

    RECOMPUTE = "recompute"  # pure/hermetic deterministic op over a pinned cone
    RESTORE = "restore"  # sampled/unpinned/effectful — needs its recorded outcome


class AuthorityDecisionKind(StrEnum):
    """Result of validating a pinned grant before an invocation or child scope."""

    GRANTED = "granted"
    DENIED = "denied"


class ProgressAxis(StrEnum):
    """The axis of a scope's outstanding-capability account."""

    CHILD_INIT = "child_init"  # whether a spawn producer can still introduce a child
    LOOP_TIME = "loop_time"  # feedback that can still arrive at a loop coordinate


class CapabilityStatus(StrEnum):
    """Lifecycle of a progress capability."""

    OPEN = "open"  # more records may still arrive on this axis
    SEALED = "sealed"  # producer will emit no more; drains as outstanding settles
    REVOKED = "revoked"  # cancellation withdrew the capability


class ValueRef(BaseModel):
    """A durable reference to a logical output value, never the value itself.

    A loop-carried value may reference a versioned ``ModelRef`` so an iteration pins the
    model version it observes without pinning a physical replica. ``collection_key``
    selects one element of a producer's fan-out collection, so a spawned child's
    child-init input is a frozen reference into the producer result rather than a copy.
    """

    model_config = ConfigDict(frozen=True)

    kind: str  # "legacy_task_result" | "inline" | "empty" | "join_result" | "model_ref"
    legacy_task_id: str | None = None
    collection_key: str | None = None  # element selector into a producer collection
    literal: str | None = None  # a bounded, immutable inline child-init value
    model_ref: ModelRef | None = None


class BoundaryEvent(BaseModel):
    """A durable correlation envelope for one mediated run-to-yield boundary.

    An episode yields this at a boundary; the physical layer carries it across the
    episode boundary and the engine applies its semantics. Beyond the request itself
    (``kind`` plus ``interface``/``child_ref``/``child_region_ref``/``state_ref``/
    ``value_ref`` and the ``request_payload`` value or reference), it carries the
    correlation the fabric needs
    to keep a re-drive exactly-once: the ``activation`` it belongs to, the adapter-local
    ``call_correlation`` that stays stable across a re-drive, the fabric-assigned
    ``idempotency_key`` (the sole dedupe authority for a mediated effect, distinct from
    ``invocation_id`` and any harness/cell-local id), the causal ``invocation_id``,
    the ``injection_target`` the outcome returns at, and the opaque ``continuation``
    capsule for the next resume. ``denial`` records a definitive authority/policy denial
    and ``outcome_value`` the settled result payload — the durable outcome the fabric
    injects back into the continuation at re-dispatch and rehydrates after a restart.
    """

    model_config = ConfigDict(frozen=True)

    kind: BoundaryEventKind
    activation: str | None = None
    call_correlation: str | None = None
    idempotency_key: str | None = None
    interface: str | None = None
    child_ref: str | None = None
    child_region_ref: str | None = None  # role a spawn/seal selects; never a raw op id
    request_payload: str | None = None
    # Worker-originated tool path: the canonical request digest and a bounded policy
    # descriptor stand in for the raw request, which stays worker-private. A set
    # ``request_digest`` marks a boundary the worker originated and runs off-lane.
    request_digest: str | None = None
    policy_descriptor: str | None = None
    injection_target: str | None = None  # the harness call id the outcome returns at
    injection_tool: str | None = None  # facade tool name, to rebuild the call
    continuation: str | None = None
    state_ref: str | None = None
    value_ref: ValueRef | None = None
    invocation_id: str | None = None
    # A turn-scoped facade group: members share one group id and one continuation,
    # ordered by source ordinal. An await-outcome member holds the resume gate; an
    # admit-and-close member (a spawn) settles at admission and never holds it.
    group_id: str | None = None
    group_ordinal: int | None = None
    completion_mode: str | None = None
    denial: DenialKind | None = None
    # A settled outcome is exactly one of these: ``outcome_value`` an inline, bounded
    # control datum, or ``outcome_ref`` a manifest for materialized content the resumed
    # worker hydrates. Both are injected at re-dispatch and rehydrated after a restart.
    outcome_value: str | None = None
    outcome_ref: OutcomeManifest | None = None


class AuthorityGrant(BaseModel):
    """Effective policy-bounded invoke/delegate rights; never bearer credentials."""

    model_config = ConfigDict(frozen=True)

    grant_id: str
    instance_id: str
    policy_id: str  # policy identity, not a secret
    invoke: tuple[str, ...] = ()
    delegate: tuple[str, ...] = ()
    epoch: int = 0


class DelegatedAuthorityGrant(BaseModel):
    """A child scope's grant, minted at a spawn site and attenuated by its parent.

    Both faces are bounded by the parent delegate face, the child region's ceiling, the
    spawn-site restriction, and the policy envelope. Snapshotted at child creation with
    its own epoch — grant minting/revocation is distinct from child-init sealing.
    """

    model_config = ConfigDict(frozen=True)

    grant_id: str
    instance_id: str
    scope_id: str
    parent_grant_id: str
    policy_id: str
    invoke: tuple[str, ...] = ()
    delegate: tuple[str, ...] = ()
    epoch: int = 0
    revoked: bool = False


class AuthorityDecision(BaseModel):
    """A durable grant-check fact recorded before or denying an invocation or spawn."""

    model_config = ConfigDict(frozen=True)

    grant_id: str
    interface: str
    kind: AuthorityDecisionKind
    work_item_id: str | None = None
    operator_id: str | None = None  # the spawn/agent site, for a dynamic denial
    scope_id: str | None = None
    denial_kind: DenialKind | None = None
    reason: str | None = None
    at: str = Field(default_factory=now_iso)


class WorkflowInstance(BaseModel):
    """Submission-level semantic identity for one workflow run."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    owner_id: str
    org_id: str = ""
    template_version: str
    plan_version: str
    policy_envelope: str | None = None
    root_grant_id: str


class Scope(BaseModel):
    """Ownership, cancellation, and progress namespace for a lineage region."""

    model_config = ConfigDict(frozen=True)

    scope_id: str
    instance_id: str
    parent_scope_id: str | None = None  # None for the root scope
    owner_operator_id: str | None = None  # spawn/call/loop that opened it; None at root
    owner_activation_id: str | None = None  # the opener activation; separates recursion
    grant_id: str | None = None  # the scope's effective (delegated) grant
    depth: int = 0  # bounds recursion


class Activation(BaseModel):
    """A lineage-level semantic instance of an operator or region."""

    model_config = ConfigDict(frozen=True)

    activation_id: str
    instance_id: str
    scope_id: str  # with loop_time/child_index, separates child vs iteration vs call
    operator_id: str
    kind: str = "leaf"
    loop_time: int = 0  # orders loop-body re-materializations
    child_index: int | None = None  # distinguishes spawned siblings
    parent_activation_id: str | None = None  # agent that owns a "region" opener


class Record(BaseModel):
    """A tagged delivery of a settled operator output to its successors."""

    model_config = ConfigDict(frozen=True)

    operator_id: str  # static template location
    activation_id: str
    scope_id: str  # scope-progress key
    loop_time: int = 0
    value_ref: ValueRef | None = None


class RegionAggregateMember(BaseModel):
    """One frozen member of a region-join aggregate, captured at join release.

    Retains the settled child's activation, its stable key (child index, never arrival
    order), the selected source output port, its terminal outcome, and an immutable
    value reference. Membership is frozen at release so a residual child never mutates
    the emitted aggregate and a restart replays the same members.
    """

    model_config = ConfigDict(frozen=True)

    child_activation_id: str
    child_key: str
    source_port: str | None = None
    outcome: PublicationOutcome
    value_ref: ValueRef | None = None


class RegionJoinAggregate(BaseModel):
    """The immutable aggregate a join publishes at its region-output endpoint.

    Materialized once when the join's release condition is met, ordered by stable child
    key. A downstream edge delivers it like any other record, creating the consumer's
    accepted input; the parent agent never receives it and never resumes for it.
    """

    model_config = ConfigDict(frozen=True)

    join_operator_id: str
    activation_id: str
    members: tuple[RegionAggregateMember, ...] = ()


class AcceptedInputMember(BaseModel):
    """One durable member of an accepted input on an agent's target port.

    Preserves its source operator/output port, source activation and child index, its
    terminal outcome, an immutable ``value_ref``, and a canonical ordinal. A merge/join
    aggregate carries one member per source child; a single producer binding carries
    exactly one.
    """

    model_config = ConfigDict(frozen=True)

    source_operator_id: str
    source_output_port: str | None = None
    source_activation_id: str
    child_index: int | None = None
    outcome: PublicationOutcome
    value_ref: ValueRef | None = None
    ordinal: int = 0


class AcceptedInput(BaseModel):
    """A durable delivery of a settled value to one agent input port.

    Keyed by (activation, target_port, occurrence_key); members are ordered by the
    declared producer/merge contract, never by arrival. Part of the agent input cone: a
    work item is ready only once each required port carries an accepted input, and a
    restart replays exactly these value references, terminal outcomes, and ordering.
    Minting an accepted input is authority-neutral — it delivers data, never an invoke
    right.
    """

    model_config = ConfigDict(frozen=True)

    activation_id: str
    target_port: str
    occurrence_key: str = "0"
    provenance: str = "producer"
    members: tuple[AcceptedInputMember, ...] = ()
    ordinal: int = 0


class Continuation(BaseModel):
    """Suspended logical progress waiting on predecessor records."""

    work_item_id: str
    waiting_on: set[str] = Field(default_factory=set)  # operator ids not settled
    # Declared input ports that must each carry an accepted input before the work item
    # is admissible; empty for an operator with no declared dataflow inputs.
    required_ports: set[str] = Field(default_factory=set)


class ProgressCapability(BaseModel):
    """One scope's outstanding-capability account on a single progress axis."""

    scope_id: str
    axis: ProgressAxis
    coordinate: int | None = None  # loop_time for the LOOP_TIME axis
    outstanding: int = 0  # records that can still arrive on this axis
    status: CapabilityStatus = CapabilityStatus.OPEN

    # Closure needs a sealed or revoked capability drained to zero, never an empty set.
    @property
    def closed(self) -> bool:
        return (
            self.status in (CapabilityStatus.SEALED, CapabilityStatus.REVOKED)
            and self.outstanding == 0
        )


class WorkItem(BaseModel):
    """Stable semantic identity for one bounded ready episode.

    Retries reuse this identity and its ``invocation_id``; only a new physical
    attempt is created. The slot a work item publishes to is keyed logically, never
    by this id or an attempt id.
    """

    work_item_id: str
    activation_id: str
    operator_id: str
    legacy_task_id: str
    status: WorkItemStatus = WorkItemStatus.BLOCKED
    outcome: PublicationOutcome | None = None
    value_ref: ValueRef | None = None  # a settled child's value, for a join winner
    invocation_id: str | None = None
    effect_class: EffectClass = EffectClass.PURE
    recovery: RecoveryClass = RecoveryClass.RECOMPUTE
    replay_contract: EffectReplayContract | None = None
    # opaque durable continuation of a yielded episode
    continuation_ref: str | None = None
    # the settled boundary call whose outcome the next episode resume injects
    pending_outcome_call: str | None = None
    # the settled facade group whose ordered outcome vector the next resume injects
    pending_outcome_group: str | None = None
    attempt_ids: list[str] = Field(default_factory=list)


class Invocation(BaseModel):
    """Causal `invocation_id` linkage and terminal state held by `DS`.

    The request/admission record itself lives in the separate control facts (`CS`);
    `DS` retains only this stable identity and its terminal semantics.
    """

    invocation_id: str
    work_item_id: str
    state: InvocationState = InvocationState.ISSUED
    replayable: bool = True
    compensable: bool = False


class Attempt(BaseModel):
    """Placement- and lease-specific physical execution history."""

    attempt_id: str
    work_item_id: str
    invocation_id: str | None
    attempt_no: int
    status: AttemptStatus = AttemptStatus.ISSUED
    worker_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class EffectReceipt(BaseModel):
    """Idempotent record of an externally visible or terminal action."""

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    work_item_id: str
    outcome: PublicationOutcome
    at: str = Field(default_factory=now_iso)


class ResultSlot(BaseModel):
    """Pending or terminal holder for one declared logical output key.

    Identified by the workflow instance, declared output, and logical key/scope/
    sequence — deliberately no physical-plan node, work-item, or attempt id.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str
    output_id: str
    source_operator_id: str
    scope_id: str | None = None
    logical_key: str | None = None  # collection key for a keyed-collection output
    sequence: int | None = None
    published: bool = False

    @property
    def slot_key(self) -> str:
        key = "" if self.logical_key is None else f":{self.logical_key}"
        seq = "" if self.sequence is None else f"#{self.sequence}"
        return f"{self.instance_id}:{self.output_id}{key}{seq}"


class ResultPublication(BaseModel):
    """Idempotent terminal publication of a declared logical output."""

    model_config = ConfigDict(frozen=True)

    slot_key: str
    output_id: str
    outcome: PublicationOutcome
    value_ref: ValueRef | None = None
    at: str = Field(default_factory=now_iso)


class OrchestrationEvent(BaseModel):
    """One entry of the compact contract-relevant trace."""

    model_config = ConfigDict(frozen=True)

    seq: int
    kind: str
    at: str = Field(default_factory=now_iso)
    operator_id: str | None = None
    work_item_id: str | None = None
    attempt_id: str | None = None
    invocation_id: str | None = None
    slot_key: str | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class LedgerSnapshot(BaseModel):
    """The durable aggregate of one workflow instance's orchestration ledger."""

    instance: WorkflowInstance
    root_scope: Scope
    root_grant: AuthorityGrant
    scopes: list[Scope] = Field(default_factory=list)
    activations: list[Activation] = Field(default_factory=list)
    work_items: list[WorkItem] = Field(default_factory=list)
    continuations: list[Continuation] = Field(default_factory=list)
    records: list[Record] = Field(default_factory=list)
    accepted_inputs: list[AcceptedInput] = Field(default_factory=list)
    region_aggregates: list[RegionJoinAggregate] = Field(default_factory=list)
    invocations: list[Invocation] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    boundary_events: list[BoundaryEvent] = Field(default_factory=list)
    effect_receipts: list[EffectReceipt] = Field(default_factory=list)
    authority_decisions: list[AuthorityDecision] = Field(default_factory=list)
    delegated_grants: list[DelegatedAuthorityGrant] = Field(default_factory=list)
    progress_capabilities: list[ProgressCapability] = Field(default_factory=list)
    result_slots: list[ResultSlot] = Field(default_factory=list)
    result_publications: list[ResultPublication] = Field(default_factory=list)
    trace: list[OrchestrationEvent] = Field(default_factory=list)
    released_scopes: list[str] = Field(default_factory=list)
    next_seq: int = 0
