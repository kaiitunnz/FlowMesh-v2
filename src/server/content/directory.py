"""Where the fabric's content is currently held.

Location evidence, and only that. A holder record says a worker reported holding an
object recently — a holder re-reports what it has on a cadence inside the record's
lifetime, so a record lapses when a holder stops reporting rather than when an object
stops being read; it is not a binding, does not keep the object alive, and confers no
right to read it — a reader still needs a grant, which is minted against the consumer
binding rather than against this. Records expire on their own, so a holder that stops
reporting simply falls out rather than leaving control believing an object is reachable.

An object may have several holders: a reference names bytes, and the same bytes written
in the same scope on two workers are the same object, so resolving one holder is a
choice among equals rather than the discovery of the one true copy.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..clients.redis import RedisClient, content_holders_key

# Objects per pipelined round trip of a holding report: the loop serving every content
# request runs other work between chunks, so a large cache never holds it for long.
_RECORD_CHUNK = 256

# Deletes a holder's field only while it still records the given incarnation.
_FORGET_GENERATION = """
local value = redis.call('HGET', KEYS[1], ARGV[1])
if not value then
    return 0
end
local ok, report = pcall(cjson.decode, value)
if ok and tonumber(report['generation']) == tonumber(ARGV[2]) then
    return redis.call('HDEL', KEYS[1], ARGV[1])
end
return 0
"""


@dataclass(frozen=True)
class ContentHolderRecord:
    """One worker's report that it holds an object, and when the report goes stale."""

    worker_id: str
    node_id: str
    generation: int
    expires_at_epoch: float

    def live(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) <= self.expires_at_epoch


class ContentHolderDirectory:
    """The holders each object was last reported on, kept per object."""

    def __init__(self, redis: RedisClient, *, record_ttl_sec: float) -> None:
        self._rds = redis
        self._ttl = record_ttl_sec

    def _field(self, expires_at_epoch: float, node_id: str, generation: int) -> str:
        return json.dumps(
            {
                "node_id": node_id,
                "generation": generation,
                "expires_at_epoch": expires_at_epoch,
            }
        )

    def _report(self, node_id: str, generation: int) -> tuple[float, str, int]:
        """A fresh report's expiry, its stored field, and the key's TTL."""
        expires_at = time.time() + self._ttl
        return (
            expires_at,
            self._field(expires_at, node_id, generation),
            int(self._ttl * 2) + 1,
        )

    def record_many(
        self,
        held: Sequence[tuple[str, str]],
        *,
        worker_id: str,
        node_id: str,
        generation: int,
    ) -> None:
        """Record a holder's whole report in a few pipelined round trips.

        A holder re-reports everything it has on each cadence, so this is proportional
        to that worker's cache; each round trip carries a bounded share of it.
        """
        _, field, ttl = self._report(node_id, generation)
        for chunk in _chunks(held):
            pipeline = self._rds.sync.control_pipeline()
            for scope, digest in chunk:
                key = content_holders_key(scope, digest)
                pipeline.hset(key, mapping={worker_id: field})
                pipeline.expire(key, ttl)
            pipeline.execute()

    async def record_many_async(
        self,
        held: Sequence[tuple[str, str]],
        *,
        worker_id: str,
        node_id: str,
        generation: int,
    ) -> None:
        _, field, ttl = self._report(node_id, generation)
        for chunk in _chunks(held):
            pipeline = self._rds.asyncio.control_pipeline()
            for scope, digest in chunk:
                key = content_holders_key(scope, digest)
                pipeline.hset(key, mapping={worker_id: field})
                pipeline.expire(key, ttl)
            await pipeline.execute()

    def record(
        self,
        scope: str,
        digest: str,
        *,
        worker_id: str,
        node_id: str,
        generation: int,
    ) -> ContentHolderRecord:
        """Note that a worker holds this object, refreshing an earlier report."""
        expires_at, field, ttl = self._report(node_id, generation)
        key = content_holders_key(scope, digest)
        self._rds.sync.hash_set(key, {worker_id: field})
        self._rds.sync.expire(key, ttl)
        return ContentHolderRecord(worker_id, node_id, generation, expires_at)

    async def record_async(
        self,
        scope: str,
        digest: str,
        *,
        worker_id: str,
        node_id: str,
        generation: int,
    ) -> ContentHolderRecord:
        expires_at, field, ttl = self._report(node_id, generation)
        key = content_holders_key(scope, digest)
        await self._rds.asyncio.hash_set(key, {worker_id: field})
        await self._rds.asyncio.expire(key, ttl)
        return ContentHolderRecord(worker_id, node_id, generation, expires_at)

    def holders(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        """Every live holder reported for this object, freshest report first."""
        return _live_records(
            self._rds.sync.hash_getall(content_holders_key(scope, digest)) or {}
        )

    async def holders_async(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        return _live_records(
            await self._rds.asyncio.hash_getall(content_holders_key(scope, digest))
            or {}
        )

    def forget(self, scope: str, digest: str, worker_id: str, generation: int) -> None:
        """Drop one holder's report, for a worker that can no longer serve it.

        Only the report of that worker incarnation goes, so a restarted holder's fresh
        report, landing meanwhile under the same worker, stays.
        """
        self._rds.sync.eval(
            _FORGET_GENERATION,
            1,
            content_holders_key(scope, digest),
            worker_id,
            str(generation),
        )

    async def forget_async(
        self, scope: str, digest: str, worker_id: str, generation: int
    ) -> None:
        await self._rds.asyncio.eval(
            _FORGET_GENERATION,
            1,
            content_holders_key(scope, digest),
            worker_id,
            str(generation),
        )


def _chunks(
    held: Sequence[tuple[str, str]],
) -> list[Sequence[tuple[str, str]]]:
    return [held[i : i + _RECORD_CHUNK] for i in range(0, len(held), _RECORD_CHUNK)]


def _live_records(raw: dict[str, Any]) -> list[ContentHolderRecord]:
    records: list[ContentHolderRecord] = []
    for worker_id, value in raw.items():
        try:
            fields = json.loads(value)
        except ValueError:
            continue
        record = ContentHolderRecord(
            worker_id=str(worker_id),
            node_id=str(fields.get("node_id") or ""),
            generation=int(fields.get("generation") or 0),
            expires_at_epoch=float(fields.get("expires_at_epoch") or 0.0),
        )
        if record.live():
            records.append(record)
    records.sort(key=lambda r: r.expires_at_epoch, reverse=True)
    return records
