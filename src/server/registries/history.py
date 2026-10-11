"""Freed workflows' stored history: their certificates, the task and output
projections their reads derive from, and the indexes the task listing pages them by.

A workflow's history is published in two steps. Preparation writes its projection and
index entries, checking every reply. Certification is one script that validates the
preparation and the workflow's revision before it writes anything, then writes the
certificate, so a certificate is present only over a complete preparation of a workflow
no write has touched since its runtime state was read.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.client import Pipeline

from ..clients.redis import (
    HISTORY_COUNTS_KEY,
    HISTORY_INDEX_KEY,
    HISTORY_SEQ_KEY,
    TASK_HISTORY_MEMBERS_KEY,
    WORKFLOWS_OPEN_KEY,
    WORKFLOWS_SET_KEY,
    RedisClient,
    task_history_key,
    task_state_key,
    workflow_history_key,
    workflow_key,
    workflow_revision_key,
    workflow_sources_key,
)
from ..task.history import (
    CERTIFICATE_FIELD,
    HISTORY_SCHEMA,
    OUTPUTS_FIELD,
    HistoryCertificate,
    OutputsHistory,
    TaskHistory,
)
from ..task.models import TaskStatus
from .task_states import PersistedTask, load_context

type _AnyPipeline = Pipeline | AsyncPipeline

# Validates a prepared publication, then certifies it. Every key's type and every value
# a write reads is checked before the first write, and the certificate is written last,
# so a certificate is present only once every other write landed. KEYS: the workflow's
# revision, history hash and record, the members hash, the all-history index, the
# sequence, the counters, the open set, then each status index. ARGV: the revision read
# before sealing, the certificate, its sequence, the history hash's field count, the
# workflow id, how many tasks succeeded, how many tasks there are, how many of them each
# status index holds, then the task ids and their members, grouped by status in KEYS
# order. Lua unpacks a bounded number of values, so ids and members are checked a chunk
# at a time.
_CERTIFY = """
local function is(key, kind)
  return redis.call('TYPE', key)['ok'] == kind
end
local function absent_or(key, kind)
  local found = redis.call('TYPE', key)['ok']
  return found == kind or found == 'none'
end
local function integer(value)
  return not value or string.match(value, '^%-?%d+$') ~= nil
end
local function all_in(command, key, from, count, expected)
  for start = 0, count - 1, 1000 do
    local size = math.min(1000, count - start)
    local names = {}
    for i = 1, size do names[i] = ARGV[from + start + i - 1] end
    local found = redis.call(command, key, unpack(names))
    for i = 1, size do
      if not found[i] or (expected and found[i] ~= ARGV[expected + start + i - 1]) then
        return false
      end
    end
  end
  return true
end
if (redis.call('GET', KEYS[1]) or '0') ~= ARGV[1] then return 'revision changed' end
if not is(KEYS[2], 'hash') then return 'projection missing' end
if redis.call('HEXISTS', KEYS[2], '<certificate>') == 1 then return 'certified' end
if redis.call('HLEN', KEYS[2]) ~= tonumber(ARGV[4]) then
  return 'projection incomplete'
end
if not is(KEYS[3], 'hash') then return 'workflow missing' end
if not absent_or(KEYS[6], 'string') or not absent_or(KEYS[7], 'hash')
  or not absent_or(KEYS[8], 'set') then return 'index type' end
local seq = redis.call('GET', KEYS[6])
if seq and not string.match(seq, '^%d+$') then return 'sequence unreadable' end
for _, count in ipairs(redis.call('HMGET', KEYS[7], 'total', 'done')) do
  if not integer(count) then return 'counters unreadable' end
end
local tasks = tonumber(ARGV[7])
local ids = 8 + (#KEYS - 8)
local members = ids + tasks
if tasks > 0 then
  if not (is(KEYS[4], 'hash') and is(KEYS[5], 'zset')) then return 'index missing' end
  if not all_in('HMGET', KEYS[2], ids, tasks) then return 'projection incomplete' end
  if not all_in('HMGET', KEYS[4], ids, tasks, members)
    or not all_in('ZMSCORE', KEYS[5], members, tasks) then
    return 'index incomplete'
  end
end
local offset = members
for i = 9, #KEYS do
  if not is(KEYS[i], 'zset') then return 'index missing' end
  local count = tonumber(ARGV[i - 1])
  if not all_in('ZMSCORE', KEYS[i], offset, count) then return 'index incomplete' end
  offset = offset + count
end
if tonumber(ARGV[3]) > tonumber(seq or '0') then
  redis.call('SET', KEYS[6], ARGV[3])
end
redis.call('HINCRBY', KEYS[7], 'total', tasks)
redis.call('HINCRBY', KEYS[7], 'done', tonumber(ARGV[6]))
redis.call('SREM', KEYS[8], ARGV[5])
redis.call('HSET', KEYS[2], '<certificate>', ARGV[2])
return 'ok'
""".replace("<certificate>", CERTIFICATE_FIELD)

# How many fields or members one preparation write carries, so no single command holds
# the store for long.
_PREPARE_CHUNK = 500


class HistoryNotPrepared(Exception):
    """A write preparing a workflow's history failed."""


class HistoryUnreadable(Exception):
    """A certified workflow's stored history is missing a part or cannot be read."""


@dataclass(frozen=True)
class HistoryPublication:
    """What freeing one workflow publishes."""

    workflow_id: str
    certificate: HistoryCertificate
    tasks: Mapping[str, TaskHistory]
    # Each task's member in the history indexes, and its terminal status.
    members: Mapping[str, tuple[str, str]]
    outputs: OutputsHistory | None


@dataclass(frozen=True)
class StoredTask:
    """A freed task: its stored record, the history derived beside it, and its
    workflow's certificate."""

    persisted: PersistedTask
    history: TaskHistory
    certificate: HistoryCertificate


def _queue_preparation(pipe: _AnyPipeline, publication: HistoryPublication) -> None:
    fields = {
        task_id: history.model_dump_json()
        for task_id, history in publication.tasks.items()
    }
    if publication.outputs is not None:
        fields[OUTPUTS_FIELD] = publication.outputs.model_dump_json()
    history_key = workflow_history_key(publication.workflow_id)
    for chunk in _chunks(list(fields.items())):
        pipe.hset(history_key, mapping=dict(chunk))
    members = [
        (task_id, member) for task_id, (member, _) in publication.members.items()
    ]
    for chunk in _chunks(members):
        pipe.hset(TASK_HISTORY_MEMBERS_KEY, mapping=dict(chunk))
        pipe.zadd(task_history_key(None), {member: 0 for _, member in chunk})
    for status, status_members in _by_status(publication).items():
        for part in _chunks(status_members):
            pipe.zadd(task_history_key(status), dict.fromkeys(part, 0))


def _chunks[T](items: Sequence[T]) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), _PREPARE_CHUNK):
        yield items[start : start + _PREPARE_CHUNK]


def _by_status(publication: HistoryPublication) -> dict[str, list[str]]:
    by_status: dict[str, list[str]] = {}
    for member, status in publication.members.values():
        by_status.setdefault(status, []).append(member)
    return by_status


def _prepared(replies: Sequence[Any], workflow_id: str) -> None:
    if errors := [reply for reply in replies if isinstance(reply, Exception)]:
        raise HistoryNotPrepared(
            f"preparing the history of {workflow_id} failed: {errors[0]}"
        )


def _certify_args(
    publication: HistoryPublication, revision: int
) -> tuple[list[str], list[str | int]]:
    workflow_id = publication.workflow_id
    by_status: dict[str, list[tuple[str, str]]] = {}
    for task_id, (member, status) in publication.members.items():
        by_status.setdefault(status, []).append((task_id, member))
    grouped = [entry for entries in by_status.values() for entry in entries]
    keys = [
        workflow_revision_key(workflow_id),
        workflow_history_key(workflow_id),
        workflow_key(workflow_id),
        TASK_HISTORY_MEMBERS_KEY,
        task_history_key(None),
        HISTORY_SEQ_KEY,
        HISTORY_COUNTS_KEY,
        WORKFLOWS_OPEN_KEY,
        *(task_history_key(status) for status in by_status),
    ]
    done = len(by_status.get(TaskStatus.DONE, ()))
    args: list[str | int] = [
        revision,
        publication.certificate.model_dump_json(),
        publication.certificate.seq,
        len(publication.tasks) + (publication.outputs is not None),
        workflow_id,
        done,
        len(grouped),
        *(len(entries) for entries in by_status.values()),
        *(task_id for task_id, _ in grouped),
        *(member for _, member in grouped),
    ]
    return keys, args


def _certificate(raw: str | None) -> HistoryCertificate | None:
    if raw is None:
        return None
    try:
        certificate = HistoryCertificate.model_validate_json(raw)
    except ValidationError as exc:
        raise HistoryUnreadable(f"unreadable history certificate: {exc}") from exc
    if certificate.schema_version != HISTORY_SCHEMA:
        raise HistoryUnreadable(
            f"history schema {certificate.schema_version} is not {HISTORY_SCHEMA}"
        )
    return certificate


def _queue_load_first(pipe: _AnyPipeline, task_ids: Sequence[str]) -> None:
    pipe.mget([task_state_key(task_id) for task_id in task_ids])
    pipe.hmget(TASK_HISTORY_MEMBERS_KEY, list(task_ids))


@dataclass(frozen=True)
class _Located:
    """A freed task's stored state, decoded far enough to find what it reads."""

    task_id: str
    data: dict[str, Any]
    workflow_id: str
    digest: str | None


def _located(
    task_ids: Sequence[str], blobs: Sequence[str | None], members: Sequence[str | None]
) -> list[_Located | None]:
    located: list[_Located | None] = []
    for task_id, blob, member in zip(task_ids, blobs, members, strict=True):
        if member is None:
            located.append(None)
            continue
        if blob is None:
            raise HistoryUnreadable(f"freed task {task_id} has no stored state")
        data = json.loads(blob)
        located.append(
            _Located(
                task_id, data, data["record"]["workflow_id"], data.get("source_digest")
            )
        )
    return located


@dataclass(frozen=True)
class _WorkflowEntries:
    """The located tasks of one workflow, and the source digests they name."""

    entries: list[_Located]
    digests: list[str]


def _by_workflow(located: Sequence[_Located | None]) -> dict[str, _WorkflowEntries]:
    grouped: dict[str, list[_Located]] = {}
    for entry in located:
        if entry is not None:
            grouped.setdefault(entry.workflow_id, []).append(entry)
    return {
        workflow_id: _WorkflowEntries(
            entries, sorted({entry.digest for entry in entries if entry.digest})
        )
        for workflow_id, entries in grouped.items()
    }


def _queue_load_second(
    pipe: _AnyPipeline, grouped: Mapping[str, _WorkflowEntries]
) -> None:
    for workflow_id, located in grouped.items():
        fields = [entry.task_id for entry in located.entries]
        pipe.hmget(workflow_history_key(workflow_id), [*fields, CERTIFICATE_FIELD])
        pipe.hmget(workflow_sources_key(workflow_id), located.digests or ["-"])


def _stored_tasks(
    located: Sequence[_Located | None],
    grouped: Mapping[str, _WorkflowEntries],
    replies: Sequence[Any],
) -> list[StoredTask | None]:
    """Build each certified task; a task whose workflow has no certificate is not
    freed, whatever an uncertified preparation left in the indexes."""
    found: dict[str, StoredTask] = {}
    for index, (workflow_id, workflow) in enumerate(grouped.items()):
        histories, texts = replies[2 * index], replies[2 * index + 1]
        if (certificate := _certificate(histories[-1])) is None:
            continue
        sources = {
            digest: text
            for digest, text in zip(workflow.digests, texts)
            if text is not None
        }
        for entry, raw in zip(workflow.entries, histories, strict=False):
            if raw is None:
                raise HistoryUnreadable(f"freed task {entry.task_id} has no history")
            try:
                found[entry.task_id] = StoredTask(
                    PersistedTask.model_validate(
                        entry.data, context=load_context(sources)
                    ),
                    TaskHistory.model_validate_json(raw),
                    certificate,
                )
            except ValidationError as exc:
                raise HistoryUnreadable(
                    f"freed task {entry.task_id} cannot be read: {exc}"
                ) from exc
    return [
        found.get(entry.task_id) if entry is not None else None for entry in located
    ]


def _queue_certified(pipe: _AnyPipeline, workflow_ids: Sequence[str]) -> None:
    for workflow_id in workflow_ids:
        pipe.hexists(workflow_history_key(workflow_id), CERTIFICATE_FIELD)


def _certified(replies: Sequence[Any]) -> list[bool]:
    return [reply is True or reply == 1 for reply in replies]


def _outputs(
    replies: Sequence[str | None],
) -> tuple[HistoryCertificate, OutputsHistory | None] | None:
    raw_certificate, raw_outputs = replies
    if (certificate := _certificate(raw_certificate)) is None:
        return None
    if raw_outputs is None:
        return certificate, None
    try:
        return certificate, OutputsHistory.model_validate_json(raw_outputs)
    except ValidationError as exc:
        raise HistoryUnreadable(f"unreadable output history: {exc}") from exc


def _range_bounds(bound: str | None, forward: bool) -> tuple[str, str]:
    if forward:
        return f"({bound}" if bound is not None else "-", "+"
    return f"({bound}" if bound is not None else "+", "-"


def _task_fields(fields: Sequence[str]) -> list[str]:
    return [
        field for field in fields if field not in (CERTIFICATE_FIELD, OUTPUTS_FIELD)
    ]


class WorkflowHistoryStore:
    """Reads and writes the stored history of freed workflows."""

    _rds: RedisClient

    def workflow_revision(self, workflow_id: str) -> int:
        """How many writes of the workflow's state have landed."""
        return int(self._rds.sync.get(workflow_revision_key(workflow_id)) or 0)

    async def workflow_revision_async(self, workflow_id: str) -> int:
        return int(await self._rds.asyncio.get(workflow_revision_key(workflow_id)) or 0)

    def prepare_history(self, publication: HistoryPublication) -> None:
        """Write a workflow's projection and index entries, uncertified.

        Raises ``HistoryNotPrepared`` when any of the writes failed.
        """
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_preparation(pipe, publication)
            _prepared(pipe.execute(raise_on_error=False), publication.workflow_id)

    async def prepare_history_async(self, publication: HistoryPublication) -> None:
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_preparation(pipe, publication)
            _prepared(await pipe.execute(raise_on_error=False), publication.workflow_id)

    def certify_history(self, publication: HistoryPublication, revision: int) -> str:
        """Certify a prepared publication, unless the workflow was written since its
        state was read at ``revision`` or the preparation is incomplete; returns
        ``"ok"`` or why it was refused, having then written nothing."""
        keys, args = _certify_args(publication, revision)
        return str(self._rds.sync.eval(_CERTIFY, len(keys), *keys, *map(str, args)))

    async def certify_history_async(
        self, publication: HistoryPublication, revision: int
    ) -> str:
        keys, args = _certify_args(publication, revision)
        return str(
            await self._rds.asyncio.eval(_CERTIFY, len(keys), *keys, *map(str, args))
        )

    def history_certificate(self, workflow_id: str) -> HistoryCertificate | None:
        """The workflow's certificate, None while it is not freed."""
        (raw,) = self._rds.sync.hash_mget(
            workflow_history_key(workflow_id), [CERTIFICATE_FIELD]
        )
        return _certificate(raw)

    async def history_certificate_async(
        self, workflow_id: str
    ) -> HistoryCertificate | None:
        (raw,) = await self._rds.asyncio.hash_mget(
            workflow_history_key(workflow_id), [CERTIFICATE_FIELD]
        )
        return _certificate(raw)

    def history_certified(self, workflow_ids: Sequence[str]) -> list[bool]:
        """Whether each workflow holds a certificate; one whose history key is not a
        hash holds none."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_certified(pipe, workflow_ids)
            return _certified(pipe.execute(raise_on_error=False))

    async def history_certified_async(self, workflow_ids: Sequence[str]) -> list[bool]:
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_certified(pipe, workflow_ids)
            return _certified(await pipe.execute(raise_on_error=False))

    def outputs_history(
        self, workflow_id: str
    ) -> tuple[HistoryCertificate, OutputsHistory | None] | None:
        """A freed workflow's certificate and published outputs, None while it is not
        freed."""
        return _outputs(
            self._rds.sync.hash_mget(
                workflow_history_key(workflow_id), [CERTIFICATE_FIELD, OUTPUTS_FIELD]
            )
        )

    async def outputs_history_async(
        self, workflow_id: str
    ) -> tuple[HistoryCertificate, OutputsHistory | None] | None:
        return _outputs(
            await self._rds.asyncio.hash_mget(
                workflow_history_key(workflow_id), [CERTIFICATE_FIELD, OUTPUTS_FIELD]
            )
        )

    def history_members(self, task_ids: Sequence[str]) -> list[str | None]:
        """Each task's member in the history indexes, None for one not freed."""
        if not task_ids:
            return []
        return self._rds.sync.hash_mget(TASK_HISTORY_MEMBERS_KEY, list(task_ids))

    async def history_members_async(self, task_ids: Sequence[str]) -> list[str | None]:
        if not task_ids:
            return []
        return await self._rds.asyncio.hash_mget(
            TASK_HISTORY_MEMBERS_KEY, list(task_ids)
        )

    def load_history(self, task_ids: Sequence[str]) -> list[StoredTask | None]:
        """Each freed task with its history, None for a task not freed.

        Raises ``HistoryUnreadable`` for a freed task whose history is incomplete.
        """
        if not task_ids:
            return []
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_load_first(pipe, task_ids)
            blobs, members = pipe.execute()
        located = _located(task_ids, blobs, members)
        if not (grouped := _by_workflow(located)):
            return [None] * len(task_ids)
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            _queue_load_second(pipe, grouped)
            replies = pipe.execute()
        return _stored_tasks(located, grouped, replies)

    async def load_history_async(
        self, task_ids: Sequence[str]
    ) -> list[StoredTask | None]:
        if not task_ids:
            return []
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_load_first(pipe, task_ids)
            blobs, members = await pipe.execute()
        located = _located(task_ids, blobs, members)
        if not (grouped := _by_workflow(located)):
            return [None] * len(task_ids)
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            _queue_load_second(pipe, grouped)
            replies = await pipe.execute()
        return _stored_tasks(located, grouped, replies)

    def history_range(
        self, status: str | None, bound: str | None, forward: bool, count: int
    ) -> list[str]:
        """Up to ``count`` freed tasks' index members strictly past ``bound`` in scan
        order, ascending when ``forward``, else descending; from the oldest or newest
        when ``bound`` is None. Only tasks that settled ``status`` when given."""
        return self._rds.sync.lex_range(
            task_history_key(status), *_range_bounds(bound, forward), count, not forward
        )

    async def history_range_async(
        self, status: str | None, bound: str | None, forward: bool, count: int
    ) -> list[str]:
        return await self._rds.asyncio.lex_range(
            task_history_key(status), *_range_bounds(bound, forward), count, not forward
        )

    def history_task_ids(self, workflow_id: str) -> list[str]:
        """The ids of a freed workflow's tasks, as far as its history holds them."""
        return _task_fields(self._rds.sync.hash_keys(workflow_history_key(workflow_id)))

    async def history_task_ids_async(self, workflow_id: str) -> list[str]:
        return _task_fields(
            await self._rds.asyncio.hash_keys(workflow_history_key(workflow_id))
        )

    def history_counts(self) -> tuple[int, int]:
        """How many certified freed tasks the store holds, and how many succeeded."""
        total, done = self._rds.sync.hash_mget(HISTORY_COUNTS_KEY, ["total", "done"])
        return int(total or 0), int(done or 0)

    async def history_counts_async(self) -> tuple[int, int]:
        total, done = await self._rds.asyncio.hash_mget(
            HISTORY_COUNTS_KEY, ["total", "done"]
        )
        return int(total or 0), int(done or 0)

    def history_seq(self) -> int:
        """The highest sequence any published certificate carries."""
        return int(self._rds.sync.get(HISTORY_SEQ_KEY) or 0)

    async def history_seq_async(self) -> int:
        return int(await self._rds.asyncio.get(HISTORY_SEQ_KEY) or 0)

    def open_workflow_ids(self) -> set[str]:
        """Every workflow registered and not freed."""
        return self._rds.sync.set_members(WORKFLOWS_OPEN_KEY)

    async def open_workflow_ids_async(self) -> set[str]:
        return await self._rds.asyncio.set_members(WORKFLOWS_OPEN_KEY)

    def reopen_workflows(self, *workflow_ids: str) -> None:
        """Count workflows among the open ones."""
        if workflow_ids:
            self._rds.sync.sadd(WORKFLOWS_OPEN_KEY, *workflow_ids)

    async def reopen_workflows_async(self, *workflow_ids: str) -> None:
        if workflow_ids:
            await self._rds.asyncio.sadd(WORKFLOWS_OPEN_KEY, *workflow_ids)

    def history_index_state(self) -> str | None:
        """How far the open-workflow index has been built over workflows registered
        before it existed."""
        return self._rds.sync.get(HISTORY_INDEX_KEY)

    async def history_index_state_async(self) -> str | None:
        return await self._rds.asyncio.get(HISTORY_INDEX_KEY)

    def set_history_index_state(self, state: str) -> None:
        self._rds.sync.set_value(HISTORY_INDEX_KEY, state)

    async def set_history_index_state_async(self, state: str) -> None:
        await self._rds.asyncio.set_value(HISTORY_INDEX_KEY, state)

    def scan_workflow_ids(self, cursor: int, count: int) -> tuple[int, list[str]]:
        """One step of a scan over every registered workflow id."""
        return self._rds.sync.scan_set(WORKFLOWS_SET_KEY, cursor, count)

    async def scan_workflow_ids_async(
        self, cursor: int, count: int
    ) -> tuple[int, list[str]]:
        return await self._rds.asyncio.scan_set(WORKFLOWS_SET_KEY, cursor, count)


__all__ = [
    "HistoryNotPrepared",
    "HistoryPublication",
    "HistoryUnreadable",
    "StoredTask",
    "WorkflowHistoryStore",
]
