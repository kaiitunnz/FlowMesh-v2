import time
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, computed_field

from shared.content import ContentReference
from shared.tasks import TaskEnvelopeTemplate, TaskType
from shared.tasks.worker_message import HardwareUsage

from ..orchestration.tool_dispatch import FacadeTurnGroup
from ..utils.time import now_iso

TRAINING_TASK_TYPES = {
    "sft",
    "lora_sft",
    "ppo",
    "dpo",
    "training",
    "image_classification_training",
}


def categorize_task_type(task_type: str | None) -> str:
    if not task_type:
        return "other"
    normalized = task_type.strip().lower()
    if normalized == "inference":
        return "inference"
    if normalized in TRAINING_TASK_TYPES:
        return "training"
    return "other"


class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    CANCELLING = "CANCELLING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DONE}
)

# A settling task has reached a terminal or is on its way to one (CANCELLING, awaiting
# its worker's terminal); the status writers refuse to regress one to an active state.
SETTLING_TASK_STATUSES = TERMINAL_TASK_STATUSES | {TaskStatus.CANCELLING}


class TerminalStatusReverted(RuntimeError):
    """A write that would move a terminal task back to an active status."""


# Task types that run a model server for the life of the task.
SERVE_TASK_TYPES = frozenset({TaskType.SERVE, TaskType.DEV_MODEL})


def serve_engine_reported(serve: Any) -> bool:
    """Report whether a serve task's ``serve`` update names its running engine.

    The engine's worker-private socket ("_"-prefixed so task metadata never discloses
    it) marks an engine the worker's sidecar can bind to.
    """
    return isinstance(serve, dict) and bool(serve.get("_socket"))


class WorkflowSettlement(NamedTuple):
    """Whether every task of a workflow has settled, and the last of their
    finishes."""

    settled: bool
    finished_ts: float | None


class TaskUsage(BaseModel):
    started_at: str = Field(description="Start timestamp.")
    finished_at: str = Field(description="Finish timestamp.")
    runtime_sec: float = Field(description="Runtime in seconds.")
    hardware: HardwareUsage = Field(description="Hardware usage details.")
    cost_per_hour: float = Field(description="Cost per hour in USD.")
    total_cost: float = Field(description="Total cost in USD.")
    status: str = Field(description="Task status at completion.")

    @classmethod
    def from_payload(cls, payload: dict[str, Any], status: str) -> "TaskUsage | None":
        try:
            return cls(
                started_at=payload["started_at"],
                finished_at=payload["finished_at"],
                runtime_sec=payload["runtime_sec"],
                hardware=HardwareUsage.model_validate(payload["hardware"]),
                cost_per_hour=payload["cost_per_hour"],
                total_cost=payload["total_cost"],
                status=status,
            )
        except (KeyError, TypeError, ValueError):
            return None


class EventEffect(StrEnum):
    """What a worker's task event did to its task."""

    STALE = "stale"
    SETTLED = "settled"
    APPLIED = "applied"
    RETURNED = "returned"
    FAILED = "failed"


class DispatchEnd(StrEnum):
    """What ending a dispatch without a result did to its task."""

    STALE = "stale"
    SETTLED = "settled"
    RETURNED = "returned"
    EXHAUSTED = "exhausted"
    MERGE_RETURNED = "merge_returned"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PublishGate(StrEnum):
    """Whether a dispatch may be published."""

    PUBLISH = "publish"
    """Its task is pending and everything the dispatch carries is durable."""
    NOT_PENDING = "not_pending"
    """Its task is not waiting for a dispatch."""
    NOT_DURABLE = "not_durable"
    """What the dispatch carries is still owed, or its task's last report waits to be
    handled again."""
    REPORTING = "reporting"
    """A report of its task is being handled; the task is queued again once it is."""


class SettleOutcome(NamedTuple):
    """What a worker's success or cancellation report did to its task; ``spent``
    when the task returned to the queue spending an attempt."""

    effect: EventEffect
    status: str | None
    usages: list[tuple[str, TaskUsage]]
    merged_children: list[str]
    impacted: tuple[tuple[str, str], ...] = ()
    spent: bool = False


class LossOutcome(NamedTuple):
    """What the loss of a v2 task's worker did to the task: it returned to the queue,
    spending an attempt when ``spent``, or failed along with the dependents in
    ``impacted``."""

    task_id: str
    end: DispatchEnd
    impacted: tuple[tuple[str, str], ...]
    spent: bool = False


class WorkerRecovery(NamedTuple):
    """The tasks a departed worker held.

    ``lost`` are the v1 tasks for the caller to return or settle; ``resolved`` are the
    v2 tasks already resolved as their worker's loss.
    """

    lost: list[str]
    resolved: list[LossOutcome]


class FailureOutcome(NamedTuple):
    """What a worker's failure report did to its task."""

    end: DispatchEnd
    attempts: int
    impacted: list[tuple[str, str]]
    usages: list[tuple[str, TaskUsage]]


class TaskRecord(BaseModel):
    task_id: str = Field(description="Task identifier.")
    workflow_id: str = Field(description="Workflow identifier.")
    owner_id: str = Field(description="Owner principal identifier.")
    org_id: str = Field(default="", description="Owner organization identifier.")
    supplier_id: str = Field(default="", description="Supplier identifier.")
    raw_yaml: str = Field(
        description="Submitted workflow source, with credentials redacted."
    )
    task: TaskEnvelopeTemplate = Field(description="Task template.")
    status: str = Field(default=TaskStatus.PENDING, description="Task status.")
    task_type: str | None = Field(default=None, description="Task type.")
    category: str | None = Field(default=None, description="Task category.")
    resident: bool = Field(
        default=False,
        description="Server-internal: backs resident capacity; not user-settable.",
    )
    residual_cancel: bool = Field(
        default=False,
        description="Server-internal: cancelled by its region's residual policy.",
    )
    assigned_worker: str | None = Field(
        default=None, description="Assigned worker identifier."
    )
    dispatch_id: str | None = Field(
        default=None, description="Dispatch holding the task.", exclude=True
    )
    topic: str | None = Field(default=None, description="Dispatch topic.")
    submitted_at: str = Field(
        default_factory=now_iso, description="Submission timestamp."
    )
    submitted_ts: float = Field(
        default_factory=time.time, description="Submission epoch seconds."
    )
    last_queue_ts: float = Field(
        default_factory=time.time, description="Last queue timestamp (epoch seconds)."
    )
    dispatched_ts: float | None = Field(
        default=None, description="Dispatch timestamp (epoch seconds)."
    )
    started_ts: float | None = Field(
        default=None, description="Start timestamp (epoch seconds)."
    )
    first_started_ts: float | None = Field(
        default=None,
        description="First start timestamp across re-runs (epoch seconds).",
        exclude=True,
    )
    finished_ts: float | None = Field(
        default=None, description="Finish timestamp (epoch seconds)."
    )
    usages: list[TaskUsage] = Field(
        default_factory=list, description="Resource usage records."
    )
    error: str | None = Field(default=None, description="Error message, if any.")
    attempts: int = Field(default=0, description="Attempt count.")
    max_attempts: int = Field(default=3, description="Max retry count.")
    parent_task_id: str | None = Field(
        default=None, description="Parent task identifier."
    )
    shard_index: int | None = Field(default=None, description="Shard index.")
    shard_total: int | None = Field(default=None, description="Total shard count.")
    next_retry_at: str | None = Field(default=None, description="Next retry timestamp.")
    failed_workers: list[str] = Field(
        default_factory=list,
        description="Distinct workers that have failed this task.",
        exclude=True,
    )
    last_error: str | None = Field(
        default=None, description="Most recent executor error message."
    )
    no_eligible_since: float | None = Field(
        default=None,
        description="Epoch seconds when no eligible worker was first observed.",
        exclude=True,
    )
    no_dispatch_since: float | None = Field(
        default=None,
        description="Epoch seconds when a selected worker was first found "
        "undeliverable.",
        exclude=True,
    )
    local_name: str | None = Field(default=None, description="Workflow stage name.")
    graph_node_name: str | None = Field(default=None, description="Graph node name.")
    load: int = Field(default=0, description="Load score.")
    position_in_epoch: int | None = Field(
        default=None, description="Position within the scheduled epoch."
    )
    selected_worker: list[str] | None = Field(
        default=None, description="Selected worker identifiers."
    )
    merged_children: list[str] | None = Field(
        default=None, description="Merged child task identifiers."
    )
    merged_parent_id: str | None = Field(
        default=None, description="Merged parent task identifier."
    )
    merged_dispatch_worker: str | None = Field(
        default=None,
        description="Worker the task's last merged dispatch went to.",
        exclude=True,
    )
    merge_slice: dict[str, int] | None = Field(
        default=None, description="Merge slice information."
    )
    merge_key: str | None = Field(default=None, description="Merge grouping key.")
    latest_update: dict[str, Any] | None = Field(
        default=None, description="Latest mid-task update payload."
    )
    latest_update_dispatch_id: str | None = Field(
        default=None,
        description="Dispatch whose report set the latest update.",
        exclude=True,
    )
    result_reference: ContentReference | None = Field(
        default=None,
        description="The stored result envelope this task's success is bound to; "
        "set once, at success.",
        exclude=True,
    )
    result_skip: dict[str, Any] | None = Field(
        default=None,
        description="Why a conditional task settled without running, when it did.",
        exclude=True,
    )
    credential_refs: dict[str, str] | None = Field(
        default=None,
        description="The vault ref of each inline credential masked in the spec, by "
        "its pointer; None on a record stored before its credentials were vaulted.",
        exclude=True,
    )
    pending_facade_group: FacadeTurnGroup | None = Field(
        default=None,
        description="A turn-scoped facade group the gateway captured for this agent "
        "episode, durable so a restart-replayed completion still routes its whole "
        "ordered membership.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def last_failed_worker(self) -> str | None:
        """The most recent worker to have failed this task."""
        return self.failed_workers[-1] if self.failed_workers else None

    def __setattr__(self, name: str, value: Any) -> None:
        # A terminal status is final; settlement and completion rely on it.
        if (
            name == "status"
            and self.status in TERMINAL_TASK_STATUSES
            and value not in TERMINAL_TASK_STATUSES
        ):
            raise TerminalStatusReverted(
                f"task {self.task_id} is {self.status} and cannot become {value}"
            )
        super().__setattr__(name, value)


class TaskInputElement(BaseModel):
    """The producer collection element a fan-out child runs on."""

    producer_task_id: str = Field(description="Task whose result holds the element.")
    index: int | None = Field(
        default=None, description="Position of the element in that collection."
    )
    path: list[str | int] = Field(
        default_factory=list,
        description="Path to the element inside the result or the collection member.",
    )


# A task's position in a listing: its submission time, then its id.
type TaskOrder = tuple[float, str]


def task_order(record: TaskRecord) -> TaskOrder:
    return record.submitted_ts, record.task_id


class TaskLoopTime(BaseModel):
    """One loop time a task runs at."""

    loop: str = Field(description="Graph node name of the loop.")
    iteration: int = Field(description="The loop's time, from 0.")


class TaskOccurrence(BaseModel):
    """Where inside a template a task runs."""

    member: str = Field(description="The template member it runs, as template/node.")
    context: str | None = Field(
        default=None, description="The spawned child whose template it runs in."
    )
    time: list[TaskLoopTime] = Field(
        default_factory=list, description="The loop times it runs at, outermost first."
    )


class TaskInfo(TaskRecord):
    depends_on: list[str] = Field(description="Dependency task IDs.")
    pending_dependencies: list[str] = Field(
        description="Unresolved dependency task IDs."
    )
    dependents: list[str] = Field(description="Dependent task IDs.")
    completed: bool = Field(description="Whether the task completed successfully.")
    failed: bool = Field(description="Whether the task failed.")
    input_element: TaskInputElement | None = Field(
        default=None, description="The producer element a fan-out child runs on."
    )
    occurrence: TaskOccurrence | None = Field(
        default=None, description="Where inside a template the task runs."
    )


class TaskParsingResult(BaseModel):
    task_id: str = Field(description="Task identifier.")
    graph_node_name: str | None = Field(description="Original graph node name.")
    depends_on: list[str] = Field(description="Dependency task IDs.")
