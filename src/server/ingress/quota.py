"""A bounded per-principal in-flight request limit for the ingress edge.

The edge admits at most a fixed number of concurrent ingress requests per principal, so
one principal cannot exhaust admission on behalf of the fabric. The limit is a
pre-admission gate: a rejected request raises no claim and consumes no resident credit.
"""

import threading


class QuotaExceeded(Exception):
    """A principal already holds its maximum concurrent ingress requests."""


class PrincipalQuota:
    """Counts each principal's in-flight ingress requests against a fixed ceiling."""

    def __init__(self, max_concurrent: int) -> None:
        self._max = max(1, max_concurrent)
        self._lock = threading.Lock()
        self._in_flight: dict[str, int] = {}

    def acquire(self, principal_id: str) -> None:
        """Reserve one slot for ``principal_id`` or raise ``QuotaExceeded``."""
        with self._lock:
            current = self._in_flight.get(principal_id, 0)
            if current >= self._max:
                raise QuotaExceeded(principal_id)
            self._in_flight[principal_id] = current + 1

    def release(self, principal_id: str) -> None:
        """Return one slot for ``principal_id``."""
        with self._lock:
            current = self._in_flight.get(principal_id, 0)
            if current <= 1:
                self._in_flight.pop(principal_id, None)
            else:
                self._in_flight[principal_id] = current - 1
