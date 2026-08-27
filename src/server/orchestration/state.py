"""Durable orchestration-ledger (`DS`) state objects.

These model the semantic hierarchy of note 21 §7: a workflow instance owns result
slots and activations; an activation owns records, continuations, and work items; a
work item owns an optional invocation and one or more physical attempts. The ledger
persists grant/policy identity and authorization decisions, never bearer credentials,
and links to the separate control facts (`CS`) only through a stable
``invocation_id``.

A few fields of the note-21 object set — a grant's ``delegate`` face and ``epoch``, a
``Scope.parent_scope_id``, an ``Activation.kind`` beyond ``leaf``, and a
``Record.loop_time`` beyond zero — carry no meaning in the acyclic subset; they are
part of the durable object set but the acyclic path never sets them off their default.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..task.v2.representations.operators import (
    EffectClass,
    EffectReplayContract,
    RecoveryClass,
)
from ..utils.time import now_iso


class WorkItemStatus(StrEnum):
    """Lifecycle of a stable semantic work item."""

    BLOCKED = "blocked"  # predecessors not settled
    READY = "ready"  # admissible for a physical attempt
    DISPATCHED = "dispatched"  # an attempt is in flight
    SETTLED = "settled"  # terminal, carries a settled outcome


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
    """Result of validating a pinned grant before an invocation."""

    GRANTED = "granted"
    DENIED = "denied"


class ValueRef(BaseModel):
    """A durable reference to a logical output value, never the value itself."""

    model_config = ConfigDict(frozen=True)

    kind: str  # "legacy_task_result" | "empty"
    legacy_task_id: str | None = None


class AuthorityGrant(BaseModel):
    """Effective policy-bounded invoke/delegate rights; never bearer credentials."""

    model_config = ConfigDict(frozen=True)

    grant_id: str
    instance_id: str
    policy_id: str  # policy identity, not a secret
    invoke: tuple[str, ...] = ()
    delegate: tuple[str, ...] = ()
    epoch: int = 0


class AuthorityDecision(BaseModel):
    """A durable grant-check fact recorded before (or denying) an invocation."""

    model_config = ConfigDict(frozen=True)

    work_item_id: str
    grant_id: str
    interface: str
    kind: AuthorityDecisionKind
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
    parent_scope_id: str | None = None


class Activation(BaseModel):
    """A lineage-level semantic instance of an operator or region."""

    model_config = ConfigDict(frozen=True)

    activation_id: str
    instance_id: str
    scope_id: str
    operator_id: str
    kind: str = "leaf"


class Record(BaseModel):
    """A tagged delivery of a settled operator output to its successors."""

    model_config = ConfigDict(frozen=True)

    operator_id: str  # static template location
    activation_id: str
    scope_id: str  # scope-progress key
    loop_time: int = 0
    value_ref: ValueRef | None = None


class Continuation(BaseModel):
    """Suspended logical progress waiting on predecessor records."""

    work_item_id: str
    waiting_on: set[str] = Field(default_factory=set)  # operator ids not settled


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
    invocation_id: str | None = None
    effect_class: EffectClass = EffectClass.PURE
    recovery: RecoveryClass = RecoveryClass.RECOMPUTE
    replay_contract: EffectReplayContract | None = None
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
    sequence: int | None = None
    published: bool = False

    @property
    def slot_key(self) -> str:
        seq = "" if self.sequence is None else f"#{self.sequence}"
        return f"{self.instance_id}:{self.output_id}{seq}"


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
    activations: list[Activation] = Field(default_factory=list)
    work_items: list[WorkItem] = Field(default_factory=list)
    continuations: list[Continuation] = Field(default_factory=list)
    records: list[Record] = Field(default_factory=list)
    invocations: list[Invocation] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    effect_receipts: list[EffectReceipt] = Field(default_factory=list)
    authority_decisions: list[AuthorityDecision] = Field(default_factory=list)
    result_slots: list[ResultSlot] = Field(default_factory=list)
    result_publications: list[ResultPublication] = Field(default_factory=list)
    trace: list[OrchestrationEvent] = Field(default_factory=list)
    next_seq: int = 0
