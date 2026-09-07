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
        self._store: dict[tuple[str, str], CapturedRequest] = {}

    def put(
        self, agent_task_id: str, call_correlation: str, request: CapturedRequest
    ) -> None:
        """Store a captured request, overwriting a stale recapture."""
        with self._lock:
            self._store[(agent_task_id, call_correlation)] = request

    def peek(self, agent_task_id: str, call_correlation: str) -> CapturedRequest | None:
        """Return the request for an occurrence without removing it."""
        with self._lock:
            return self._store.get((agent_task_id, call_correlation))

    def delete(self, agent_task_id: str, call_correlation: str) -> None:
        """Drop the request for an occurrence once its outcome has committed."""
        with self._lock:
            self._store.pop((agent_task_id, call_correlation), None)
