"""Where the fabric's content is currently held.

Location evidence, and only that. A holder record says a worker reported keeping a copy
of an object; it is not a binding, does not keep the object alive, and confers no right
to read it — a reader needs a grant, which is minted against the consumer binding. A
worker reports a copy when it keeps it and when it drops it, and its records last while
it is registered: a reader checks the holder's registration, and the records of an
unregistered worker are swept. Each worker's records are indexed under it, so a sweep
walks only that worker's copies.

An object may have several holders: a reference names bytes, and the same bytes written
in the same scope on two workers are the same object, so resolving one holder is a
choice among equals rather than the discovery of the one true copy.
"""

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from shared.content import HeldCopy

from ..clients.redis import (
    CONTENT_HELD_WORKERS_KEY,
    WORKERS_SET_KEY,
    RedisClient,
    content_held_key,
    content_held_seq_key,
    content_holders_key,
)

# Copies per scripted round trip: the loop serving every content request runs other
# work between chunks, so a large holding never blocks it for long.
_CHUNK = 256

# Applies one chunk of a registered worker's holding changes, and its report number.
# An unregistered worker gets nothing written, so a late report cannot bring back
# records its sweep removed.
_APPLY = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
    return 0
end
redis.call('SADD', KEYS[2], ARGV[1])
for i = 5, #KEYS do
    local op = ARGV[2 * (i - 5) + 4]
    local member = ARGV[2 * (i - 5) + 5]
    if op == '+' then
        redis.call('HSET', KEYS[i], ARGV[1], ARGV[2])
        redis.call('SADD', KEYS[3], member)
    else
        redis.call('HDEL', KEYS[i], ARGV[1])
        redis.call('SREM', KEYS[3], member)
    end
end
if ARGV[3] ~= '' then
    redis.call('SET', KEYS[4], ARGV[3])
end
return 1
"""

# Deletes a holder's field, and its entry in the worker's index, only while the field
# records the given incarnation.
_FORGET_GENERATION = """
local value = redis.call('HGET', KEYS[1], ARGV[1])
if not value then
    return 0
end
local ok, report = pcall(cjson.decode, value)
if ok and tonumber(report['generation']) == tonumber(ARGV[2]) then
    redis.call('SREM', KEYS[2], ARGV[3])
    return redis.call('HDEL', KEYS[1], ARGV[1])
end
return 0
"""

_ADD = "+"
_DROP = "-"


@dataclass(frozen=True)
class ContentHolderRecord:
    """One worker's report that it holds an object."""

    worker_id: str
    node_id: str
    generation: int
    recorded_at_epoch: float


class ContentHolderDirectory:
    """Each object's reported holders, indexed per worker."""

    def __init__(self, redis: RedisClient) -> None:
        self._rds = redis

    def apply(
        self,
        worker_id: str,
        *,
        node_id: str,
        generation: int,
        added: Sequence[HeldCopy] = (),
        dropped: Sequence[HeldCopy] = (),
        seq: int | None = None,
    ) -> bool:
        """Record the copies a worker kept and forget the ones it dropped, then note
        ``seq`` as its last report; False when the worker is not registered."""
        field = _field(node_id, generation)
        applied = True
        for keys, args in _scripts(worker_id, field, _changes(added, dropped), seq):
            applied = bool(self._rds.sync.eval(_APPLY, len(keys), *keys, *args))
            if not applied:
                break
        return applied

    async def apply_async(
        self,
        worker_id: str,
        *,
        node_id: str,
        generation: int,
        added: Sequence[HeldCopy] = (),
        dropped: Sequence[HeldCopy] = (),
        seq: int | None = None,
    ) -> bool:
        field = _field(node_id, generation)
        applied = True
        for keys, args in _scripts(worker_id, field, _changes(added, dropped), seq):
            applied = bool(
                await self._rds.asyncio.eval(_APPLY, len(keys), *keys, *args)
            )
            if not applied:
                break
        return applied

    def replace(
        self,
        worker_id: str,
        held: Sequence[HeldCopy],
        *,
        node_id: str,
        generation: int,
        seq: int,
    ) -> bool:
        """Make a worker's records exactly ``held``; False when it is not registered.

        Only the difference from what is recorded is written.
        """
        recorded = self._indexed(worker_id)
        added, dropped = _difference(recorded, held)
        return self.apply(
            worker_id,
            node_id=node_id,
            generation=generation,
            added=added,
            dropped=dropped,
            seq=seq,
        )

    async def replace_async(
        self,
        worker_id: str,
        held: Sequence[HeldCopy],
        *,
        node_id: str,
        generation: int,
        seq: int,
    ) -> bool:
        recorded = await self._indexed_async(worker_id)
        added, dropped = _difference(recorded, held)
        return await self.apply_async(
            worker_id,
            node_id=node_id,
            generation=generation,
            added=added,
            dropped=dropped,
            seq=seq,
        )

    def applied_seq(self, worker_id: str) -> int:
        """The number of the last report recorded for a worker, 0 before any."""
        return int(self._rds.sync.get(content_held_seq_key(worker_id)) or 0)

    async def applied_seq_async(self, worker_id: str) -> int:
        return int(await self._rds.asyncio.get(content_held_seq_key(worker_id)) or 0)

    def holders(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        """Every holder reported for this object, most recently recorded first."""
        return _records(
            self._rds.sync.hash_getall(content_holders_key(scope, digest)) or {}
        )

    async def holders_async(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        return _records(
            await self._rds.asyncio.hash_getall(content_holders_key(scope, digest))
            or {}
        )

    def forget(self, scope: str, digest: str, worker_id: str, generation: int) -> None:
        """Drop one holder's report, for a worker that can no longer serve it.

        Only the report of that worker incarnation goes.
        """
        self._rds.sync.eval(
            _FORGET_GENERATION, 2, *_forget_args(scope, digest, worker_id, generation)
        )

    async def forget_async(
        self, scope: str, digest: str, worker_id: str, generation: int
    ) -> None:
        await self._rds.asyncio.eval(
            _FORGET_GENERATION, 2, *_forget_args(scope, digest, worker_id, generation)
        )

    def unregistered(self) -> list[str]:
        """The unregistered workers with records indexed."""
        return sorted(
            self._rds.sync.set_difference(CONTENT_HELD_WORKERS_KEY, WORKERS_SET_KEY)
        )

    async def unregistered_async(self) -> list[str]:
        return sorted(
            await self._rds.asyncio.set_difference(
                CONTENT_HELD_WORKERS_KEY, WORKERS_SET_KEY
            )
        )

    def sweep(self, worker_id: str) -> int:
        """Remove every record a worker left, and return how many went."""
        index = content_held_key(worker_id)
        swept = 0
        cursor = 0
        while True:
            cursor, members = self._rds.sync.scan_set(index, cursor, _CHUNK)
            if members:
                with self._rds.sync.control_pipeline(transaction=False) as pipe:
                    for member in members:
                        pipe.hdel(_object_key(member), worker_id)
                    pipe.execute()
                swept += len(members)
            if cursor == 0:
                break
        with self._rds.sync.control_pipeline() as pipe:
            _queue_sweep_tail(pipe, worker_id)
            pipe.execute()
        return swept

    async def sweep_async(self, worker_id: str) -> int:
        index = content_held_key(worker_id)
        swept = 0
        cursor = 0
        while True:
            cursor, members = await self._rds.asyncio.scan_set(index, cursor, _CHUNK)
            if members:
                async with self._rds.asyncio.control_pipeline(
                    transaction=False
                ) as pipe:
                    for member in members:
                        pipe.hdel(_object_key(member), worker_id)
                    await pipe.execute()
                swept += len(members)
            if cursor == 0:
                break
        async with self._rds.asyncio.control_pipeline() as pipe:
            _queue_sweep_tail(pipe, worker_id)
            await pipe.execute()
        return swept

    def _indexed(self, worker_id: str) -> set[HeldCopy]:
        index = content_held_key(worker_id)
        found: set[HeldCopy] = set()
        cursor = 0
        while True:
            cursor, members = self._rds.sync.scan_set(index, cursor, _CHUNK)
            found.update(_copy(member) for member in members)
            if cursor == 0:
                return found

    async def _indexed_async(self, worker_id: str) -> set[HeldCopy]:
        index = content_held_key(worker_id)
        found: set[HeldCopy] = set()
        cursor = 0
        while True:
            cursor, members = await self._rds.asyncio.scan_set(index, cursor, _CHUNK)
            found.update(_copy(member) for member in members)
            if cursor == 0:
                return found


def _field(node_id: str, generation: int) -> str:
    return json.dumps(
        {"node_id": node_id, "generation": generation, "recorded_at": time.time()}
    )


def _member(scope: str, digest: str) -> str:
    return f"{scope}:{digest}"


def _copy(member: str) -> HeldCopy:
    # A digest is hex, so the last separator is the one between scope and digest.
    scope, digest = member.rsplit(":", 1)
    return scope, digest


def _object_key(member: str) -> str:
    return content_holders_key(*_copy(member))


def _changes(
    added: Iterable[HeldCopy], dropped: Iterable[HeldCopy]
) -> list[tuple[str, HeldCopy]]:
    return [(_DROP, copy) for copy in dropped] + [(_ADD, copy) for copy in added]


def _difference(
    recorded: set[HeldCopy], held: Sequence[HeldCopy]
) -> tuple[list[HeldCopy], list[HeldCopy]]:
    wanted = set(held)
    return sorted(wanted - recorded), sorted(recorded - wanted)


def _scripts(
    worker_id: str,
    field: str,
    changes: list[tuple[str, HeldCopy]],
    seq: int | None,
) -> Iterator[tuple[list[str], list[str]]]:
    """The keys and arguments of each chunk's script; the last one notes ``seq``, and
    an empty change set still has one script to note it."""
    starts = range(0, len(changes), _CHUNK) or range(1)
    for start in starts:
        chunk = changes[start : start + _CHUNK]
        last = start == starts[-1]
        keys = [
            WORKERS_SET_KEY,
            CONTENT_HELD_WORKERS_KEY,
            content_held_key(worker_id),
            content_held_seq_key(worker_id),
            *(content_holders_key(scope, digest) for _, (scope, digest) in chunk),
        ]
        args = [worker_id, field, str(seq) if last and seq is not None else ""]
        for op, (scope, digest) in chunk:
            args.extend((op, _member(scope, digest)))
        yield keys, args


def _forget_args(
    scope: str, digest: str, worker_id: str, generation: int
) -> tuple[str, ...]:
    return (
        content_holders_key(scope, digest),
        content_held_key(worker_id),
        worker_id,
        str(generation),
        _member(scope, digest),
    )


def _queue_sweep_tail(pipe: Any, worker_id: str) -> None:
    """Queue the removal of a swept worker's index, report number and roster entry."""
    pipe.delete(content_held_key(worker_id), content_held_seq_key(worker_id))
    pipe.srem(CONTENT_HELD_WORKERS_KEY, worker_id)


def _records(raw: dict[str, Any]) -> list[ContentHolderRecord]:
    records: list[ContentHolderRecord] = []
    for worker_id, value in raw.items():
        try:
            fields = json.loads(value)
        except ValueError:
            continue
        records.append(
            ContentHolderRecord(
                worker_id=str(worker_id),
                node_id=str(fields.get("node_id") or ""),
                generation=int(fields.get("generation") or 0),
                recorded_at_epoch=float(fields.get("recorded_at") or 0.0),
            )
        )
    records.sort(key=lambda r: r.recorded_at_epoch, reverse=True)
    return records
