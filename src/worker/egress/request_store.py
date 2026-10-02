"""Worker-private custody for captured, not-yet-executed mediated-egress requests."""

import threading

from shared.tools.model.schema import ModelRequest
from shared.tools.search.schema import ToolRequest

# A captured worker-originated egress request: a fabric-tool request or a managed-model
# request, both held in worker-private custody behind their control-plane digest.
CapturedRequest = ToolRequest | ModelRequest


class PendingEgressRequestStore:
    """Worker-private store for captured, not-yet-executed mediated-egress requests.

    When a worker originates a mediated egress boundary it keeps the raw request here,
    keyed by the stable ``(agent_task_id, call_correlation)`` occurrence, and sends the
    control plane only a digest. The mediated-egress sidecar reads the request back on
    the same worker, so the raw request never crosses to the control plane. The
    store lives for one worker incarnation; a restart is a new incarnation whose rotated
    id and generation invalidate any outstanding permit and force the boundary to be
    re-proposed on the freshly assigned worker, so a durable backing is not needed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each request, with the dispatch it was captured under, which its off-lane
        # work runs for after the dispatch's own run ends.
        self._store: dict[tuple[str, str], tuple[CapturedRequest, str | None]] = {}

    def put(
        self,
        agent_task_id: str,
        call_correlation: str,
        request: CapturedRequest,
        dispatch_id: str | None,
    ) -> None:
        """Store a captured request, overwriting a stale recapture."""
        with self._lock:
            self._store[(agent_task_id, call_correlation)] = (request, dispatch_id)

    def dispatch_of(self, agent_task_id: str) -> str | None:
        """The dispatch a request this task holds was captured under, if any."""
        with self._lock:
            return next(
                (
                    dispatch_id
                    for (task_id, _), (_, dispatch_id) in self._store.items()
                    if task_id == agent_task_id and dispatch_id is not None
                ),
                None,
            )

    def peek(self, agent_task_id: str, call_correlation: str) -> CapturedRequest | None:
        """Return the request for an occurrence without removing it."""
        with self._lock:
            held = self._store.get((agent_task_id, call_correlation))
        return held[0] if held is not None else None

    def delete(self, agent_task_id: str, call_correlation: str) -> None:
        """Drop the request for an occurrence once its outcome has committed."""
        with self._lock:
            self._store.pop((agent_task_id, call_correlation), None)

    def discard(
        self, agent_task_id: str, call_correlation: str, request: CapturedRequest
    ) -> None:
        """Drop the request for an occurrence only while it is still ``request``, so a
        later capture of the same occurrence stays."""
        key = (agent_task_id, call_correlation)
        with self._lock:
            if (held := self._store.get(key)) is not None and held[0] is request:
                del self._store[key]

    def occurrences(self) -> list[tuple[str, str]]:
        """The occurrences whose requests the store holds."""
        with self._lock:
            return list(self._store)

    def clear(self) -> None:
        """Drop every request, as the incarnation that captured them has ended."""
        with self._lock:
            self._store.clear()
