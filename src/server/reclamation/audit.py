"""The capped stream of reclamation facts an operator reads: backfill coverage, scope
certification, eligibility and what each sweep found."""

import json
import time
from typing import Any

from ..clients.redis import RECLAIM_AUDIT_KEY, RedisClient

# Roughly how many facts the stream keeps; older ones are trimmed.
AUDIT_MAXLEN = 10_000


class AuditTrail:
    """Appends reclamation facts to the control store's audit stream."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def record(self, kind: str, **facts: Any) -> None:
        """Append one fact of ``kind``; a value that is not a string is stored as
        JSON."""
        fields: dict[Any, Any] = {"kind": kind, "at": f"{time.time():.3f}"}
        for name, value in facts.items():
            fields[name] = value if isinstance(value, str) else json.dumps(value)
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.xadd(RECLAIM_AUDIT_KEY, fields, maxlen=AUDIT_MAXLEN, approximate=True)
            pipe.execute()


__all__ = ["AUDIT_MAXLEN", "AuditTrail"]
