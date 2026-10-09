import json
from collections.abc import AsyncGenerator, Collection, Generator, Iterator, Sequence
from contextlib import aclosing, closing
from enum import StrEnum
from itertools import batched
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
)
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.client import Pipeline

from shared.tasks import PERSISTED_LOAD_CONTEXT
from shared.utils.time import iso_to_ns

from ..clients.redis import (
    WORKFLOWS_BY_SUBMISSION_KEY,
    WORKFLOWS_SET_KEY,
    RedisClient,
    task_state_key,
    workflow_blueprints_key,
    workflow_cancelled_tasks_key,
    workflow_credential_key,
    workflow_dispatched_tasks_key,
    workflow_ds_key,
    workflow_dynamic_tasks_key,
    workflow_failed_tasks_key,
    workflow_key,
    workflow_sched_key,
    workflow_tasks_key,
    workflow_v2_key,
)
from ..orchestration.state import LedgerSnapshot
from ..task.models import TaskRecord, TaskStatus
from ..task.v2 import PersistedV2Workflow
from ..utils.cursors import page_slice
from ..utils.query import QueryFilter
from ..utils.time import now_iso


class PersistedTask(BaseModel):
    """A durable per-task snapshot sufficient to rebuild scheduler state."""

    model_config = ConfigDict(frozen=True)

    record: TaskRecord
    depends_on: set[str] = Field(default_factory=set)
    epoch_index: int | None = None

    @field_serializer("depends_on")
    def _serialize_depends_on(self, value: set[str]) -> list[str]:
        return sorted(value)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        # ``failed_workers`` is excluded from TaskRecord's dump but routes retries, so
        # it must survive a restart.
        data["record"]["failed_workers"] = self.record.failed_workers.copy()
        # Likewise the result binding and skip detail: internal to the task's result
        # read, and what that read resolves from after a restart.
        reference = self.record.result_reference
        data["record"]["result_reference"] = (
            reference.model_dump(mode="json") if reference is not None else None
        )
        data["record"]["result_skip"] = self.record.result_skip
        # And which worker a merged dispatch went to, which tells a later report of
        # that worker's failure apart from the task's own.
        data["record"]["merged_dispatch_worker"] = self.record.merged_dispatch_worker
        # And the dispatch holding the task, which fences its worker's events.
        data["record"]["dispatch_id"] = self.record.dispatch_id
        # And the dispatch that reported its latest update, which fences an endpoint
        # the update carries to the dispatch that served it.
        data["record"][
            "latest_update_dispatch_id"
        ] = self.record.latest_update_dispatch_id
        # And where its vaulted credentials go back into its spec at dispatch.
        data["record"]["credential_refs"] = self.record.credential_refs
        # And when its first run started, which a serve task's TTL counts from.
        data["record"]["first_started_ts"] = self.record.first_started_ts
        return data


class TaskBlueprints(BaseModel):
    """The task each region-definition member and spawn child is materialized from.

    Written once at submission; a blueprint is never dispatched or counted as one of
    the workflow's tasks.
    """

    model_config = ConfigDict(frozen=True)

    tasks: list[PersistedTask] = Field(default_factory=list)


class WorkflowSched(BaseModel):
    """Durable per-workflow scheduling state (epoch ordering)."""

    model_config = ConfigDict(frozen=True)

    in_epoch_order: bool = False
    epoch_frontier: int = 0


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


TERMINAL_WORKFLOW_STATUSES = frozenset(
    {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED, WorkflowStatus.DONE}
)


class WorkflowControl(BaseModel):
    """What a workflow's ledger holds beyond its tasks: whether it still waits on a
    value read to route a branch or fan out a spawn, and why it failed outside any
    task, if it did."""

    model_config = ConfigDict(frozen=True)

    open: bool = False
    failure: str | None = None

    def fields(self) -> dict[str, str]:
        """The workflow-record fields that store it."""
        return {
            "control_open": "1" if self.open else "",
            "control_failure": self.failure or "",
        }


class WorkflowRecord(BaseModel):
    workflow_id: str = Field(description="Workflow identifier.")
    task_ids: list[str] = Field(description="Task identifiers in the workflow.")
    submitted_at: str = Field(
        default_factory=now_iso, description="Submission timestamp."
    )
    updated_at: str = Field(
        default_factory=now_iso, description="Last update timestamp."
    )
    control_open: bool = Field(
        default=False,
        description="Whether the workflow waits on a value read no task holds.",
    )
    control_failure: str = Field(
        default="", description="Why the workflow failed outside any task, if it did."
    )

    @field_serializer("control_open")
    def serialize_control_open(self, control_open: bool) -> str:
        return "1" if control_open else ""

    @field_validator("control_open", mode="before")
    def deserialize_control_open(cls, v: Any) -> bool:
        return v not in ("", "0", None, False)

    @field_serializer("task_ids")
    def serialize_task_ids(self, task_ids: list[str]) -> str:
        return json.dumps(task_ids)

    @field_validator("task_ids", mode="before")
    def deserialize_task_ids(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v


# A workflow's position in a listing: its submission time in epoch microseconds,
# then its id.
type WorkflowOrder = tuple[int, str]

# The submission index holds every workflow under one score as its zero-padded
# submission microsecond and id, so its lexical order is the listing order and a
# position is an exclusive lexical bound.
_SUBMISSION_DIGITS = 16
SUBMISSION_US_MAX = 10**_SUBMISSION_DIGITS - 1
# The most index entries one filtered scan step reads.
_SCAN_CHUNK_MAX = 1000
# The most index entries one backfill command adds.
_BACKFILL_BATCH = 1000


class Workflow(BaseModel):
    workflow_id: str = Field(description="Workflow identifier.")
    task_ids: list[str] = Field(description="Task identifiers in the workflow.")
    submitted_at: str = Field(description="Submission timestamp.")
    updated_at: str = Field(description="Last update timestamp.")
    status: WorkflowStatus = Field(description="Workflow status.")
    dispatched_tasks: list[str] = Field(description="Dispatched task identifiers.")
    completed_tasks: list[str] = Field(description="Completed task identifiers.")
    failed_tasks: list[str] = Field(description="Failed task identifiers.")
    cancelled_tasks: list[str] = Field(description="Cancelled task identifiers.")


def workflow_order(submitted_at: str, workflow_id: str) -> WorkflowOrder:
    try:
        micros = iso_to_ns(submitted_at) // 1_000
    except ValueError:
        micros = 0
    return min(max(micros, 0), SUBMISSION_US_MAX), workflow_id


def _index_member(order: WorkflowOrder) -> str:
    micros, workflow_id = order
    return f"{micros:0{_SUBMISSION_DIGITS}d}:{workflow_id}"


def _record_member(record: WorkflowRecord) -> str:
    return _index_member(workflow_order(record.submitted_at, record.workflow_id))


def _indexed_id(member: str) -> str:
    return member[_SUBMISSION_DIGITS + 1 :]


def _scan_sizes(limit: int, filtered: bool) -> Iterator[int]:
    """Yield the entries each scan step reads: ``limit``, then doubling up to the
    larger of ``limit`` and the chunk cap while a filter rejects entries."""
    size, cap = limit, max(limit, _SCAN_CHUNK_MAX) if filtered else limit
    while True:
        yield size
        size = min(size * 2, cap)


def _create_workflow_record(
    workflow_id: str, tasks: list[TaskRecord], submitted_at: str | None = None
) -> tuple[WorkflowRecord, list[str], list[str]]:
    """Return (WorkflowRecord, remaining_task_ids, failed_task_ids)

    ``submitted_at`` is the caller's own start of submission. Stamping it here instead
    would time the record's construction, which a v2 submission reaches only after
    compilation and the first ledger drive.
    """
    task_ids: list[str] = []
    remaining_tasks: list[str] = []
    failed_tasks: list[str] = []
    for task in tasks:
        task_ids.append(task.task_id)
        match task.status:
            case TaskStatus.DONE:
                continue
            case TaskStatus.FAILED:
                failed_tasks.append(task.task_id)
            case _:
                remaining_tasks.append(task.task_id)
    record = WorkflowRecord(
        workflow_id=workflow_id,
        task_ids=task_ids,
        submitted_at=submitted_at or now_iso(),
    )
    return record, remaining_tasks, failed_tasks


def _workflow_update(mapping: dict[str, Any] | None = None) -> dict[str, Any]:
    if mapping is None:
        mapping = {}
    if "updated_at" not in mapping:
        mapping["updated_at"] = now_iso()
    return mapping


type _AnyPipeline = Pipeline | AsyncPipeline


def _queue_workflow_reads(pipe: _AnyPipeline, workflow_ids: Sequence[str]) -> None:
    for workflow_id in workflow_ids:
        pipe.hgetall(workflow_key(workflow_id))
        pipe.smembers(workflow_dispatched_tasks_key(workflow_id))
        pipe.smembers(workflow_failed_tasks_key(workflow_id))
        pipe.smembers(workflow_cancelled_tasks_key(workflow_id))
        pipe.smembers(workflow_tasks_key(workflow_id))


def _queue_submission_stamps(pipe: _AnyPipeline, ids: Sequence[str]) -> None:
    for workflow_id in ids:
        pipe.hget(workflow_key(workflow_id), "submitted_at")


def _backfill_members(ids: Sequence[str], stamps: Sequence[str | None]) -> list[str]:
    return [
        _index_member(workflow_order(stamp, workflow_id))
        for workflow_id, stamp in zip(ids, stamps, strict=True)
        if stamp
    ]


def _queue_backfill(pipe: _AnyPipeline, members: Sequence[str]) -> None:
    for chunk in batched(members, _BACKFILL_BATCH):
        pipe.zadd(WORKFLOWS_BY_SUBMISSION_KEY, dict.fromkeys(chunk, 0))


def _index_scan_start(
    after: WorkflowOrder | None, before: WorkflowOrder | None
) -> tuple[bool, str | None]:
    """Whether an index scan runs forward, and its exclusive start member."""
    forward = after is not None
    bound = after if forward else before
    return forward, f"({_index_member(bound)}" if bound is not None else None


def _candidate_chunks(
    ids: Sequence[str],
    stamps: Sequence[str | None],
    sizes: Iterator[int],
    after: WorkflowOrder | None,
    before: WorkflowOrder | None,
) -> Iterator[list[str]]:
    """Yield the candidates past a position in scan order, as an index scan does."""
    keys = sorted(
        workflow_order(stamp, workflow_id)
        for workflow_id, stamp in zip(ids, stamps, strict=True)
        if stamp
    )
    window = keys[page_slice(keys, len(keys), after=after, before=before)]
    scan = window if after is not None else window[::-1]
    start = 0
    for size in sizes:
        if start >= len(scan):
            return
        yield [workflow_id for _, workflow_id in scan[start : start + size]]
        start += size


def _page_order(
    page: list[Workflow], limit: int, after: WorkflowOrder | None
) -> list[Workflow]:
    page = page[:limit]
    return page if after is not None else page[::-1]


def _queue_task_states(pipe: _AnyPipeline, items: Sequence[PersistedTask]) -> None:
    for item in items:
        pipe.set(task_state_key(item.record.task_id), item.model_dump_json())


def _queue_registration(
    pipe: _AnyPipeline,
    workflow_id: str,
    tasks: Sequence[PersistedTask],
    sched: WorkflowSched,
    v2: PersistedV2Workflow | None,
    ledger: LedgerSnapshot | None,
    submitted_at: str | None,
    blueprints: Sequence[PersistedTask],
) -> None:
    record, remaining_tasks, failed_tasks = _create_workflow_record(
        workflow_id, [item.record for item in tasks], submitted_at
    )
    pipe.sadd(WORKFLOWS_SET_KEY, workflow_id)
    pipe.zadd(WORKFLOWS_BY_SUBMISSION_KEY, {_record_member(record): 0})
    pipe.hset(workflow_key(workflow_id), mapping=record.model_dump())
    if remaining_tasks:
        pipe.sadd(workflow_tasks_key(workflow_id), *remaining_tasks)
    if failed_tasks:
        pipe.sadd(workflow_failed_tasks_key(workflow_id), *failed_tasks)
    _queue_task_states(pipe, tasks)
    pipe.set(workflow_sched_key(workflow_id), sched.model_dump_json())
    if v2 is not None:
        pipe.set(workflow_v2_key(workflow_id), v2.model_dump_json())
    if ledger is not None:
        pipe.set(workflow_ds_key(workflow_id), ledger.model_dump_json())
    if blueprints:
        pipe.set(
            workflow_blueprints_key(workflow_id),
            TaskBlueprints(tasks=list(blueprints)).model_dump_json(),
        )


def _queue_transition(
    pipe: _AnyPipeline,
    workflow_id: str,
    records: Sequence[PersistedTask],
    dispatched: Sequence[str],
    pending: Sequence[str],
    done: Sequence[str],
    failed: Sequence[str],
    cancelled: Sequence[str],
    sched: WorkflowSched | None,
) -> None:
    terminal = (*done, *failed, *cancelled)
    touched_membership = bool(dispatched or pending or terminal)
    _queue_task_states(pipe, records)
    if dispatched:
        pipe.sadd(workflow_dispatched_tasks_key(workflow_id), *dispatched)
    if pending:
        pipe.srem(workflow_dispatched_tasks_key(workflow_id), *pending)
    if terminal:
        pipe.srem(workflow_tasks_key(workflow_id), *terminal)
        pipe.srem(workflow_dispatched_tasks_key(workflow_id), *terminal)
    if failed:
        pipe.sadd(workflow_failed_tasks_key(workflow_id), *failed)
    if cancelled:
        pipe.sadd(workflow_cancelled_tasks_key(workflow_id), *cancelled)
    if touched_membership or sched is not None:
        pipe.hset(workflow_key(workflow_id), mapping=_workflow_update())
    if sched is not None:
        pipe.set(workflow_sched_key(workflow_id), sched.model_dump_json())


def _queue_dynamic_tasks(
    pipe: _AnyPipeline,
    workflow_id: str,
    records: Sequence[PersistedTask],
    snapshot: LedgerSnapshot,
    retire: Sequence[str],
    dispatched: Sequence[str],
    done: Sequence[str],
    failed: Sequence[str],
    cancelled: Sequence[str],
    sched: WorkflowSched | None,
    control: WorkflowControl | None,
) -> None:
    ids = [item.record.task_id for item in records]
    settled = {*done, *failed, *cancelled}
    _queue_task_states(pipe, records)
    if ids:
        pipe.sadd(workflow_dynamic_tasks_key(workflow_id), *ids)
    # Exact membership, so a child an ambiguous earlier write left behind converges.
    if remaining := [task_id for task_id in ids if task_id not in settled]:
        pipe.sadd(workflow_tasks_key(workflow_id), *remaining)
    if settled:
        pipe.srem(workflow_tasks_key(workflow_id), *settled)
    if idle := [task_id for task_id in ids if task_id not in dispatched]:
        pipe.srem(workflow_dispatched_tasks_key(workflow_id), *idle)
    if dispatched:
        pipe.sadd(workflow_dispatched_tasks_key(workflow_id), *dispatched)
    if failed:
        pipe.sadd(workflow_failed_tasks_key(workflow_id), *failed)
    if cancelled:
        pipe.sadd(workflow_cancelled_tasks_key(workflow_id), *cancelled)
    if retire:
        pipe.srem(workflow_tasks_key(workflow_id), *retire)
    _queue_ledger(pipe, workflow_id, snapshot, control)
    if sched is not None:
        pipe.set(workflow_sched_key(workflow_id), sched.model_dump_json())


def _queue_ledger(
    pipe: _AnyPipeline,
    workflow_id: str,
    snapshot: LedgerSnapshot,
    control: WorkflowControl | None,
) -> None:
    pipe.set(workflow_ds_key(workflow_id), snapshot.model_dump_json())
    update = control.fields() if control is not None else {}
    pipe.hset(workflow_key(workflow_id), mapping=_workflow_update(update))


def _blueprints(blob: str | bytes | None) -> list[PersistedTask]:
    if not blob:
        return []
    return TaskBlueprints.model_validate_json(
        blob, context=PERSISTED_LOAD_CONTEXT
    ).tasks


def _sched_payload(in_epoch_order: bool, epoch_frontier: int) -> str:
    return WorkflowSched(
        in_epoch_order=in_epoch_order, epoch_frontier=epoch_frontier
    ).model_dump_json()


class WorkflowRegistry:
    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def register_workflow(
        self,
        workflow_id: str,
        tasks: Sequence[PersistedTask],
        sched: WorkflowSched,
        v2: PersistedV2Workflow | None = None,
        ledger: LedgerSnapshot | None = None,
        submitted_at: str | None = None,
        blueprints: Sequence[PersistedTask] = (),
    ) -> None:
        """Register a workflow with its task states, schedule, plan, ledger and task
        blueprints as one atomic transaction."""
        with self._rds.sync.control_pipeline() as pipe:
            _queue_registration(
                pipe, workflow_id, tasks, sched, v2, ledger, submitted_at, blueprints
            )
            pipe.execute()

    async def register_workflow_async(
        self,
        workflow_id: str,
        tasks: Sequence[PersistedTask],
        sched: WorkflowSched,
        v2: PersistedV2Workflow | None = None,
        ledger: LedgerSnapshot | None = None,
        submitted_at: str | None = None,
        blueprints: Sequence[PersistedTask] = (),
    ) -> None:
        """Register a workflow as ``register_workflow`` does."""
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_registration(
                pipe, workflow_id, tasks, sched, v2, ledger, submitted_at, blueprints
            )
            await pipe.execute()

    def unregister_workflows(self, *workflow_ids: str) -> None:
        task_ids, members = self._collect(workflow_ids)
        with self._rds.sync.control_pipeline() as pipe:
            pipe.srem(WORKFLOWS_SET_KEY, *workflow_ids)
            if members:
                pipe.zrem(WORKFLOWS_BY_SUBMISSION_KEY, *members)
            pipe.delete(*(workflow_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dispatched_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dynamic_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_failed_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_cancelled_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_sched_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_v2_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_ds_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_blueprints_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_credential_key(wid) for wid in workflow_ids))
            for task_id in task_ids:
                pipe.delete(task_state_key(task_id))
            pipe.execute()

    async def unregister_workflows_async(self, *workflow_ids: str) -> None:
        task_ids, members = await self._collect_async(workflow_ids)
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.srem(WORKFLOWS_SET_KEY, *workflow_ids)
            if members:
                pipe.zrem(WORKFLOWS_BY_SUBMISSION_KEY, *members)
            pipe.delete(*(workflow_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dispatched_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dynamic_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_failed_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_cancelled_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_sched_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_v2_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_ds_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_blueprints_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_credential_key(wid) for wid in workflow_ids))
            for task_id in task_ids:
                pipe.delete(task_state_key(task_id))
            await pipe.execute()

    def get_workflow_ids(self) -> set[str]:
        return self._rds.sync.set_members(WORKFLOWS_SET_KEY)

    async def get_workflow_ids_async(self) -> set[str]:
        return await self._rds.asyncio.set_members(WORKFLOWS_SET_KEY)

    def get_workflow_record(self, workflow_id: str) -> WorkflowRecord | None:
        data = self._rds.sync.hash_getall(workflow_key(workflow_id))
        return WorkflowRecord.model_validate(data) if data else None

    async def get_workflow_record_async(
        self, workflow_id: str
    ) -> WorkflowRecord | None:
        data = await self._rds.asyncio.hash_getall(workflow_key(workflow_id))
        return WorkflowRecord.model_validate(data) if data else None

    def workflow_exists(self, workflow_id: str) -> bool:
        return self._rds.sync.exists(workflow_key(workflow_id))

    async def workflow_exists_async(self, workflow_id: str) -> bool:
        return await self._rds.asyncio.exists(workflow_key(workflow_id))

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        workflows = self.get_workflows([workflow_id])
        return workflows[0] if workflows else None

    async def get_workflow_async(self, workflow_id: str) -> Workflow | None:
        workflows = await self.get_workflows_async([workflow_id])
        return workflows[0] if workflows else None

    def get_workflows(self, workflow_ids: Sequence[str]) -> list[Workflow]:
        """Read the named workflows, in order, in one round trip; leave out a missing
        one."""
        with self._rds.sync.control_pipeline() as pipe:
            _queue_workflow_reads(pipe, workflow_ids)
            replies = pipe.execute()
        return self._built_workflows(replies)

    async def get_workflows_async(self, workflow_ids: Sequence[str]) -> list[Workflow]:
        """Read the named workflows, in order, in one round trip; leave out a missing
        one."""
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_workflow_reads(pipe, workflow_ids)
            replies = await pipe.execute()
        return self._built_workflows(replies)

    def index_submissions(self) -> int:
        """Add every registered workflow the submission index lacks; return how many
        it added."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.scard(WORKFLOWS_SET_KEY)
            pipe.zcard(WORKFLOWS_BY_SUBMISSION_KEY)
            registered, indexed = pipe.execute()
        # Registration and removal update the set and the index in one transaction,
        # so an index as large as the set covers it.
        if indexed >= registered:
            return 0
        ids = list(self.get_workflow_ids())
        if not (members := _backfill_members(ids, self._submission_stamps(ids))):
            return 0
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_backfill(pipe, members)
            added = pipe.execute()
        return sum(added)

    async def index_submissions_async(self) -> int:
        """Add every registered workflow the submission index lacks; return how many
        it added."""
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            pipe.scard(WORKFLOWS_SET_KEY)
            pipe.zcard(WORKFLOWS_BY_SUBMISSION_KEY)
            registered, indexed = await pipe.execute()
        # Registration and removal update the set and the index in one transaction,
        # so an index as large as the set covers it.
        if indexed >= registered:
            return 0
        ids = list(await self.get_workflow_ids_async())
        stamps = await self._submission_stamps_async(ids)
        if not (members := _backfill_members(ids, stamps)):
            return 0
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_backfill(pipe, members)
            added = await pipe.execute()
        return sum(added)

    def workflow_page(
        self,
        query: QueryFilter,
        limit: int,
        after: WorkflowOrder | None = None,
        before: WorkflowOrder | None = None,
        candidates: Collection[str] | None = None,
    ) -> list[Workflow]:
        """Return the workflows matching ``query``, among ``candidates`` when given,
        ordered by submission: the ``limit`` just after or before a position, or the
        newest ``limit``.

        Without candidates the page reads the submission index from the position;
        with them it orders just the candidates.
        """
        sizes = _scan_sizes(limit, bool(query))
        scan = (
            self._indexed_ids(sizes, after, before)
            if candidates is None
            else self._candidate_ids(candidates, sizes, after, before)
        )
        page: list[Workflow] = []
        with closing(scan) as chunks:
            for ids in chunks:
                page.extend(query.filter(self.get_workflows(ids)))
                if len(page) >= limit:
                    break
        return _page_order(page, limit, after)

    async def workflow_page_async(
        self,
        query: QueryFilter,
        limit: int,
        after: WorkflowOrder | None = None,
        before: WorkflowOrder | None = None,
        candidates: Collection[str] | None = None,
    ) -> list[Workflow]:
        """Return the page ``workflow_page`` does."""
        sizes = _scan_sizes(limit, bool(query))
        scan = (
            self._indexed_ids_async(sizes, after, before)
            if candidates is None
            else self._candidate_ids_async(candidates, sizes, after, before)
        )
        page: list[Workflow] = []
        async with aclosing(scan) as chunks:
            async for ids in chunks:
                page.extend(query.filter(await self.get_workflows_async(ids)))
                if len(page) >= limit:
                    break
        return _page_order(page, limit, after)

    def _indexed_ids(
        self,
        sizes: Iterator[int],
        after: WorkflowOrder | None,
        before: WorkflowOrder | None,
    ) -> Generator[list[str]]:
        """Yield the indexed workflow ids past a position in scan order: ascending
        after ``after``, else descending before ``before`` or from the newest."""
        forward, start = _index_scan_start(after, before)
        for size in sizes:
            members = self._rds.sync.lex_range(
                WORKFLOWS_BY_SUBMISSION_KEY,
                start or ("-" if forward else "+"),
                "+" if forward else "-",
                size,
                reverse=not forward,
            )
            if members:
                yield [_indexed_id(member) for member in members]
            if len(members) < size:
                return
            start = f"({members[-1]}"

    async def _indexed_ids_async(
        self,
        sizes: Iterator[int],
        after: WorkflowOrder | None,
        before: WorkflowOrder | None,
    ) -> AsyncGenerator[list[str]]:
        forward, start = _index_scan_start(after, before)
        for size in sizes:
            members = await self._rds.asyncio.lex_range(
                WORKFLOWS_BY_SUBMISSION_KEY,
                start or ("-" if forward else "+"),
                "+" if forward else "-",
                size,
                reverse=not forward,
            )
            if members:
                yield [_indexed_id(member) for member in members]
            if len(members) < size:
                return
            start = f"({members[-1]}"

    def _candidate_ids(
        self,
        candidates: Collection[str],
        sizes: Iterator[int],
        after: WorkflowOrder | None,
        before: WorkflowOrder | None,
    ) -> Generator[list[str]]:
        ids = list(candidates)
        stamps = self._submission_stamps(ids)
        yield from _candidate_chunks(ids, stamps, sizes, after, before)

    async def _candidate_ids_async(
        self,
        candidates: Collection[str],
        sizes: Iterator[int],
        after: WorkflowOrder | None,
        before: WorkflowOrder | None,
    ) -> AsyncGenerator[list[str]]:
        ids = list(candidates)
        stamps = await self._submission_stamps_async(ids)
        for chunk in _candidate_chunks(ids, stamps, sizes, after, before):
            yield chunk

    def _submission_stamps(self, ids: Sequence[str]) -> list[str | None]:
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_submission_stamps(pipe, ids)
            return pipe.execute()

    async def _submission_stamps_async(self, ids: Sequence[str]) -> list[str | None]:
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_submission_stamps(pipe, ids)
            return await pipe.execute()

    def commit_transition(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        """Apply a workflow state delta as one atomic control-Redis transaction.

        ``records`` are upserted; ``dispatched`` / ``pending`` / ``done`` /
        ``failed`` / ``cancelled`` move their task ids into the matching status-set
        membership; ``sched`` snapshots the schedule when present. The records,
        membership moves, the workflow's ``updated_at``, and the schedule snapshot
        commit together or not at all, so a crash mid-persist can never leave
        durable state half-applied.
        """
        with self._rds.sync.control_pipeline() as pipe:
            _queue_transition(
                pipe,
                workflow_id,
                records,
                dispatched,
                pending,
                done,
                failed,
                cancelled,
                sched,
            )
            pipe.execute()

    async def commit_transition_async(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        """Apply a workflow state delta as ``commit_transition`` does."""
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_transition(
                pipe,
                workflow_id,
                records,
                dispatched,
                pending,
                done,
                failed,
                cancelled,
                sched,
            )
            await pipe.execute()

    def commit_dynamic_tasks(
        self,
        workflow_id: str,
        records: Sequence[PersistedTask],
        snapshot: LedgerSnapshot,
        retire: Sequence[str] = (),
        *,
        dispatched: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
        control: WorkflowControl | None = None,
    ) -> None:
        """Persist newly materialized dynamic-child records with the ledger snapshot.

        The child records, their dynamic-tasks and status-set membership, and the ledger
        snapshot that carries their work items commit in one atomic transaction, so a
        crash can never leave the ledger's dynamic children without their durable task
        records or vice versa. The ids join the dynamic-tasks set so restart rehydration
        reloads them alongside the statically registered tasks. Each child is in the
        status sets its record is in and leaves the others: the remaining set unless
        listed in ``done``, ``failed`` or ``cancelled``, and the dispatched set only
        when listed in ``dispatched``. ``retire`` drops tasks from the remaining set as
        the spawn seals — the child template that has finished instantiating children
        and no longer holds the workflow short of completion — so the children replace
        the template atomically and never leave it transiently empty. ``sched``
        snapshots the schedule when present.
        """
        if not records and not retire:
            return
        with self._rds.sync.control_pipeline() as pipe:
            _queue_dynamic_tasks(
                pipe,
                workflow_id,
                records,
                snapshot,
                retire,
                dispatched,
                done,
                failed,
                cancelled,
                sched,
                control,
            )
            pipe.execute()

    async def commit_dynamic_tasks_async(
        self,
        workflow_id: str,
        records: Sequence[PersistedTask],
        snapshot: LedgerSnapshot,
        retire: Sequence[str] = (),
        *,
        dispatched: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
        control: WorkflowControl | None = None,
    ) -> None:
        """Persist dynamic-child records as ``commit_dynamic_tasks`` does."""
        if not records and not retire:
            return
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_dynamic_tasks(
                pipe,
                workflow_id,
                records,
                snapshot,
                retire,
                dispatched,
                done,
                failed,
                cancelled,
                sched,
                control,
            )
            await pipe.execute()

    def get_dynamic_task_ids(self, workflow_id: str) -> set[str]:
        return self._rds.sync.set_members(workflow_dynamic_tasks_key(workflow_id))

    async def get_dynamic_task_ids_async(self, workflow_id: str) -> set[str]:
        return await self._rds.asyncio.set_members(
            workflow_dynamic_tasks_key(workflow_id)
        )

    # ---- Durable task state (for restart rehydration) ----------------- #

    def save_task_states(self, items: Sequence[PersistedTask]) -> None:
        if not items:
            return
        with self._rds.sync.control_pipeline() as pipe:
            _queue_task_states(pipe, items)
            pipe.execute()

    async def save_task_states_async(self, items: Sequence[PersistedTask]) -> None:
        if not items:
            return
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_task_states(pipe, items)
            await pipe.execute()

    def load_task_states(self, *task_ids: str) -> list[PersistedTask | None]:
        if not task_ids:
            return []
        blobs = self._rds.sync.mget([task_state_key(task_id) for task_id in task_ids])
        return [
            (
                PersistedTask.model_validate_json(blob, context=PERSISTED_LOAD_CONTEXT)
                if blob
                else None
            )
            for blob in blobs
        ]

    async def load_task_states_async(
        self, *task_ids: str
    ) -> list[PersistedTask | None]:
        if not task_ids:
            return []
        blobs = await self._rds.asyncio.mget(
            [task_state_key(task_id) for task_id in task_ids]
        )
        return [
            (
                PersistedTask.model_validate_json(blob, context=PERSISTED_LOAD_CONTEXT)
                if blob
                else None
            )
            for blob in blobs
        ]

    def save_workflow_sched(
        self, workflow_id: str, in_epoch_order: bool, epoch_frontier: int
    ) -> None:
        self._rds.sync.set_value(
            workflow_sched_key(workflow_id),
            _sched_payload(in_epoch_order, epoch_frontier),
        )

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, epoch_frontier: int
    ) -> None:
        await self._rds.asyncio.set_value(
            workflow_sched_key(workflow_id),
            _sched_payload(in_epoch_order, epoch_frontier),
        )

    def load_workflow_sched(self, workflow_id: str) -> WorkflowSched | None:
        blob = self._rds.sync.get(workflow_sched_key(workflow_id))
        return WorkflowSched.model_validate_json(blob) if blob else None

    async def load_workflow_sched_async(self, workflow_id: str) -> WorkflowSched | None:
        blob = await self._rds.asyncio.get(workflow_sched_key(workflow_id))
        return WorkflowSched.model_validate_json(blob) if blob else None

    def get_v2_workflow(self, workflow_id: str) -> PersistedV2Workflow | None:
        blob = self._rds.sync.get(workflow_v2_key(workflow_id))
        return PersistedV2Workflow.model_validate_json(blob) if blob else None

    async def get_v2_workflow_async(
        self, workflow_id: str
    ) -> PersistedV2Workflow | None:
        blob = await self._rds.asyncio.get(workflow_v2_key(workflow_id))
        return PersistedV2Workflow.model_validate_json(blob) if blob else None

    # ---- Durable orchestration ledger (`DS`) -------------------------- #

    def save_ledger_snapshot(
        self,
        workflow_id: str,
        snapshot: LedgerSnapshot,
        control: WorkflowControl | None = None,
    ) -> None:
        """Save a workflow's ledger with what it holds beyond its tasks."""
        with self._rds.sync.control_pipeline() as pipe:
            _queue_ledger(pipe, workflow_id, snapshot, control)
            pipe.execute()

    async def save_ledger_snapshot_async(
        self,
        workflow_id: str,
        snapshot: LedgerSnapshot,
        control: WorkflowControl | None = None,
    ) -> None:
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_ledger(pipe, workflow_id, snapshot, control)
            await pipe.execute()

    def load_ledger_snapshot(self, workflow_id: str) -> LedgerSnapshot | None:
        blob = self._rds.sync.get(workflow_ds_key(workflow_id))
        return LedgerSnapshot.model_validate_json(blob) if blob else None

    async def load_ledger_snapshot_async(
        self, workflow_id: str
    ) -> LedgerSnapshot | None:
        blob = await self._rds.asyncio.get(workflow_ds_key(workflow_id))
        return LedgerSnapshot.model_validate_json(blob) if blob else None

    def load_blueprints(self, workflow_id: str) -> list[PersistedTask]:
        blob = self._rds.sync.get(workflow_blueprints_key(workflow_id))
        return _blueprints(blob)

    async def load_blueprints_async(self, workflow_id: str) -> list[PersistedTask]:
        blob = await self._rds.asyncio.get(workflow_blueprints_key(workflow_id))
        return _blueprints(blob)

    def get_remaining_tasks(self, workflow_id: str) -> set[str]:
        return self._rds.sync.set_members(workflow_tasks_key(workflow_id))

    async def get_remaining_tasks_async(self, workflow_id: str) -> set[str]:
        return await self._rds.asyncio.set_members(workflow_tasks_key(workflow_id))

    def _build_workflow(
        self,
        record: WorkflowRecord,
        dispatched_tasks: set[str],
        failed_tasks: set[str],
        cancelled_tasks: set[str],
        remaining_tasks: set[str],
    ) -> Workflow:
        active_dispatched = dispatched_tasks.intersection(remaining_tasks)
        if failed_tasks or record.control_failure:
            status = WorkflowStatus.FAILED
        elif remaining_tasks or record.control_open:
            if active_dispatched:
                status = WorkflowStatus.DISPATCHED
            else:
                status = WorkflowStatus.PENDING
        elif cancelled_tasks:
            status = WorkflowStatus.CANCELLED
        else:
            status = WorkflowStatus.DONE
        completed_tasks = (
            set(record.task_ids) - remaining_tasks - failed_tasks - cancelled_tasks
        )
        return Workflow(
            workflow_id=record.workflow_id,
            task_ids=record.task_ids,
            submitted_at=record.submitted_at,
            updated_at=record.updated_at,
            status=status,
            dispatched_tasks=list(active_dispatched),
            completed_tasks=list(completed_tasks),
            failed_tasks=list(failed_tasks),
            cancelled_tasks=list(cancelled_tasks),
        )

    def _built_workflows(self, replies: Sequence[Any]) -> list[Workflow]:
        return [
            self._build_workflow(
                WorkflowRecord.model_validate(data),
                dispatched,
                failed,
                cancelled,
                remaining,
            )
            for data, dispatched, failed, cancelled, remaining in batched(replies, 5)
            if data
        ]

    def _collect(self, workflow_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        """Return the workflows' task ids and their submission-index members."""
        task_ids: list[str] = []
        members: list[str] = []
        for wid in workflow_ids:
            record = self.get_workflow_record(wid)
            if record:
                task_ids.extend(record.task_ids)
                members.append(_record_member(record))
            task_ids.extend(self._rds.sync.set_members(workflow_dynamic_tasks_key(wid)))
        return task_ids, members

    async def _collect_async(
        self, workflow_ids: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        task_ids: list[str] = []
        members: list[str] = []
        for wid in workflow_ids:
            record = await self.get_workflow_record_async(wid)
            if record:
                task_ids.extend(record.task_ids)
                members.append(_record_member(record))
            task_ids.extend(
                await self._rds.asyncio.set_members(workflow_dynamic_tasks_key(wid))
            )
        return task_ids, members
