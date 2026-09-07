"""Worker-private custody for a captured resident boundary's raw model request."""

import threading


class ResidentRequestStore:
    """Worker-private store for a captured resident boundary's raw model request.

    An agent worker holds its resident request here, keyed by the stable
    ``(agent_task_id, call_correlation)`` occurrence, and sends control only a digest.
    The origin driver reads it back on the same worker and carries it over the data
    path, so the raw request never crosses to the control plane. It lives for one worker
    incarnation; a restart re-captures on the freshly assigned worker.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[tuple[str, str], str] = {}

    def put(self, agent_task_id: str, call_correlation: str, request: str) -> None:
        with self._lock:
            self._store[(agent_task_id, call_correlation)] = request

    def peek(self, agent_task_id: str, call_correlation: str) -> str | None:
        with self._lock:
            return self._store.get((agent_task_id, call_correlation))

    def delete(self, agent_task_id: str, call_correlation: str) -> None:
        with self._lock:
            self._store.pop((agent_task_id, call_correlation), None)
