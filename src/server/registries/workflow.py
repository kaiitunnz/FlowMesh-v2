import json
from collections.abc import Sequence
from enum import StrEnum
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

from ..clients.redis import (
    WORKFLOWS_SET_KEY,
    RedisClient,
    task_state_key,
    workflow_cancelled_tasks_key,
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
        return data


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


class WorkflowRecord(BaseModel):
    workflow_id: str = Field(description="Workflow identifier.")
    task_ids: list[str] = Field(description="Task identifiers in the workflow.")
    submitted_at: str = Field(
        default_factory=now_iso, description="Submission timestamp."
    )
    updated_at: str = Field(
        default_factory=now_iso, description="Last update timestamp."
    )

    @field_serializer("task_ids")
    def serialize_task_ids(self, task_ids: list[str]) -> str:
        return json.dumps(task_ids)

    @field_validator("task_ids", mode="before")
    def deserialize_task_ids(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v


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


def _create_workflow_record(
    workflow_id: str, tasks: list[TaskRecord]
) -> tuple[WorkflowRecord, list[str], list[str]]:
    """Return (WorkflowRecord, remaining_task_ids, failed_task_ids)"""
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
    record = WorkflowRecord(workflow_id=workflow_id, task_ids=task_ids)
    return record, remaining_tasks, failed_tasks


def _workflow_update(mapping: dict[str, Any] | None = None) -> dict[str, Any]:
    if mapping is None:
        mapping = {}
    if "updated_at" not in mapping:
        mapping["updated_at"] = now_iso()
    return mapping


class WorkflowRegistry:
    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def register_workflow(
        self,
        workflow_id: str,
        tasks: list[TaskRecord],
        v2: PersistedV2Workflow | None = None,
    ) -> None:
        record, remaining_tasks, failed_tasks = _create_workflow_record(
            workflow_id, tasks
        )
        with self._rds.sync.control_pipeline() as pipe:
            pipe.sadd(WORKFLOWS_SET_KEY, workflow_id)
            pipe.hset(workflow_key(workflow_id), mapping=record.model_dump())
            if remaining_tasks:
                pipe.sadd(workflow_tasks_key(workflow_id), *remaining_tasks)
            if failed_tasks:
                pipe.sadd(workflow_failed_tasks_key(workflow_id), *failed_tasks)
            if v2 is not None:
                pipe.set(workflow_v2_key(workflow_id), v2.model_dump_json())
            pipe.execute()

    async def register_workflow_async(
        self,
        workflow_id: str,
        tasks: list[TaskRecord],
        v2: PersistedV2Workflow | None = None,
    ) -> None:
        record, remaining_tasks, failed_tasks = _create_workflow_record(
            workflow_id, tasks
        )
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.sadd(WORKFLOWS_SET_KEY, workflow_id)
            pipe.hset(workflow_key(workflow_id), mapping=record.model_dump())
            if remaining_tasks:
                pipe.sadd(workflow_tasks_key(workflow_id), *remaining_tasks)
            if failed_tasks:
                pipe.sadd(workflow_failed_tasks_key(workflow_id), *failed_tasks)
            if v2 is not None:
                pipe.set(workflow_v2_key(workflow_id), v2.model_dump_json())
            await pipe.execute()

    def unregister_workflows(self, *workflow_ids: str) -> None:
        task_ids = self._collect_task_ids(workflow_ids)
        with self._rds.sync.control_pipeline() as pipe:
            pipe.srem(WORKFLOWS_SET_KEY, *workflow_ids)
            pipe.delete(*(workflow_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dispatched_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dynamic_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_failed_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_cancelled_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_sched_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_v2_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_ds_key(wid) for wid in workflow_ids))
            for task_id in task_ids:
                pipe.delete(task_state_key(task_id))
            pipe.execute()

    async def unregister_workflows_async(self, *workflow_ids: str) -> None:
        task_ids = self._collect_task_ids(workflow_ids)
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.srem(WORKFLOWS_SET_KEY, *workflow_ids)
            pipe.delete(*(workflow_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dispatched_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_dynamic_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_failed_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_cancelled_tasks_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_sched_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_v2_key(wid) for wid in workflow_ids))
            pipe.delete(*(workflow_ds_key(wid) for wid in workflow_ids))
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
        record = self.get_workflow_record(workflow_id)
        if record is None:
            return None
        dispatched_tasks = self._rds.sync.set_members(
            workflow_dispatched_tasks_key(workflow_id)
        )
        failed_tasks = self._rds.sync.set_members(
            workflow_failed_tasks_key(workflow_id)
        )
        cancelled_tasks = self._rds.sync.set_members(
            workflow_cancelled_tasks_key(workflow_id)
        )
        remaining_tasks = self._rds.sync.set_members(workflow_tasks_key(workflow_id))
        return self._build_workflow(
            record,
            dispatched_tasks,
            failed_tasks,
            cancelled_tasks,
            remaining_tasks,
        )

    async def get_workflow_async(self, workflow_id: str) -> Workflow | None:
        record = await self.get_workflow_record_async(workflow_id)
        if record is None:
            return None
        dispatched_tasks = await self._rds.asyncio.set_members(
            workflow_dispatched_tasks_key(workflow_id)
        )
        failed_tasks = await self._rds.asyncio.set_members(
            workflow_failed_tasks_key(workflow_id)
        )
        cancelled_tasks = await self._rds.asyncio.set_members(
            workflow_cancelled_tasks_key(workflow_id)
        )
        remaining_tasks = await self._rds.asyncio.set_members(
            workflow_tasks_key(workflow_id)
        )
        return self._build_workflow(
            record,
            dispatched_tasks,
            failed_tasks,
            cancelled_tasks,
            remaining_tasks,
        )

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
        terminal = (*done, *failed, *cancelled)
        touched_membership = bool(dispatched or pending or terminal)
        with self._rds.sync.control_pipeline() as pipe:
            for item in records:
                pipe.set(task_state_key(item.record.task_id), item.model_dump_json())
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
            pipe.execute()

    def commit_dynamic_tasks(
        self,
        workflow_id: str,
        records: Sequence[PersistedTask],
        snapshot: LedgerSnapshot,
        retire: Sequence[str] = (),
    ) -> None:
        """Persist newly materialized dynamic-child records with the ledger snapshot.

        The child records, their dynamic-tasks set membership, and the ledger snapshot
        that carries their work items commit in one atomic transaction, so a crash can
        never leave the ledger's dynamic children without their durable task records or
        vice versa. The ids join the dynamic-tasks set so restart rehydration reloads
        them alongside the statically registered tasks. ``retire`` drops tasks from the
        remaining set as the spawn seals — the child template that has finished
        instantiating children and no longer holds the workflow short of completion —
        so the children replace the template atomically and never leave it transiently
        empty.
        """
        if not records and not retire:
            return
        ids = [item.record.task_id for item in records]
        with self._rds.sync.control_pipeline() as pipe:
            for item in records:
                pipe.set(task_state_key(item.record.task_id), item.model_dump_json())
            if ids:
                pipe.sadd(workflow_dynamic_tasks_key(workflow_id), *ids)
                pipe.sadd(workflow_tasks_key(workflow_id), *ids)
            if retire:
                pipe.srem(workflow_tasks_key(workflow_id), *retire)
            pipe.set(workflow_ds_key(workflow_id), snapshot.model_dump_json())
            pipe.hset(workflow_key(workflow_id), mapping=_workflow_update())
            pipe.execute()

    async def get_dynamic_task_ids_async(self, workflow_id: str) -> set[str]:
        return await self._rds.asyncio.set_members(
            workflow_dynamic_tasks_key(workflow_id)
        )

    # ---- Durable task state (for restart rehydration) ----------------- #

    async def save_task_states_async(self, items: Sequence[PersistedTask]) -> None:
        if not items:
            return
        async with self._rds.asyncio.control_pipeline() as pipe:
            for item in items:
                pipe.set(task_state_key(item.record.task_id), item.model_dump_json())
            await pipe.execute()

    def load_task_states(self, *task_ids: str) -> list[PersistedTask | None]:
        if not task_ids:
            return []
        blobs = self._rds.sync.mget([task_state_key(task_id) for task_id in task_ids])
        return [
            PersistedTask.model_validate_json(blob) if blob else None for blob in blobs
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
            PersistedTask.model_validate_json(blob) if blob else None for blob in blobs
        ]

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, epoch_frontier: int
    ) -> None:
        payload = WorkflowSched(
            in_epoch_order=in_epoch_order, epoch_frontier=epoch_frontier
        ).model_dump_json()
        await self._rds.asyncio.set_value(workflow_sched_key(workflow_id), payload)

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

    def save_ledger_snapshot(self, workflow_id: str, snapshot: LedgerSnapshot) -> None:
        self._rds.sync.set_value(
            workflow_ds_key(workflow_id), snapshot.model_dump_json()
        )

    async def save_ledger_snapshot_async(
        self, workflow_id: str, snapshot: LedgerSnapshot
    ) -> None:
        await self._rds.asyncio.set_value(
            workflow_ds_key(workflow_id), snapshot.model_dump_json()
        )

    def load_ledger_snapshot(self, workflow_id: str) -> LedgerSnapshot | None:
        blob = self._rds.sync.get(workflow_ds_key(workflow_id))
        return LedgerSnapshot.model_validate_json(blob) if blob else None

    async def load_ledger_snapshot_async(
        self, workflow_id: str
    ) -> LedgerSnapshot | None:
        blob = await self._rds.asyncio.get(workflow_ds_key(workflow_id))
        return LedgerSnapshot.model_validate_json(blob) if blob else None

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
        if failed_tasks:
            status = WorkflowStatus.FAILED
        elif remaining_tasks:
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

    def _collect_task_ids(self, workflow_ids: Sequence[str]) -> list[str]:
        task_ids: list[str] = []
        for wid in workflow_ids:
            record = self.get_workflow_record(wid)
            if record:
                task_ids.extend(record.task_ids)
            task_ids.extend(self._rds.sync.set_members(workflow_dynamic_tasks_key(wid)))
        return task_ids
