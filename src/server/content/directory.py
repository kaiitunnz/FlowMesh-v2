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
from dataclasses import dataclass

from ..clients.redis import RedisClient


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

    @staticmethod
    def _key(scope: str, digest: str) -> str:
        return f"ct:holders:{scope}:{digest}"

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
        record = ContentHolderRecord(
            worker_id=worker_id,
            node_id=node_id,
            generation=generation,
            expires_at_epoch=time.time() + self._ttl,
        )
        key = self._key(scope, digest)
        self._rds.sync.hash_set(
            key,
            {
                worker_id: json.dumps(
                    {
                        "node_id": node_id,
                        "generation": generation,
                        "expires_at_epoch": record.expires_at_epoch,
                    }
                )
            },
        )
        self._rds.sync.expire(key, int(self._ttl * 2) + 1)
        return record

    def holders(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        """Every live holder reported for this object, freshest report first."""
        raw = self._rds.sync.hash_getall(self._key(scope, digest)) or {}
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

    def forget(self, scope: str, digest: str, worker_id: str) -> None:
        """Drop one holder's report, for a worker that can no longer serve it."""
        self._rds.sync.hash_delete(self._key(scope, digest), worker_id)
