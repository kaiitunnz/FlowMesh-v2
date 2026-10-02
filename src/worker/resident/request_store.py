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
        # The dispatch each request was captured under, which its off-lane work runs
        # for after the dispatch's own run ends.
        self._dispatches: dict[tuple[str, str], str] = {}

    def put(
        self,
        agent_task_id: str,
        call_correlation: str,
        request: str,
        dispatch_id: str | None = None,
    ) -> None:
        key = (agent_task_id, call_correlation)
        with self._lock:
            self._store[key] = request
            self._dispatches.pop(key, None)
            if dispatch_id is not None:
                self._dispatches[key] = dispatch_id

    def dispatch_of(self, agent_task_id: str) -> str | None:
        """The dispatch a request this task holds was captured under, if any."""
        with self._lock:
            return next(
                (
                    d
                    for (task, _), d in self._dispatches.items()
                    if task == agent_task_id
                ),
                None,
            )

    def peek(self, agent_task_id: str, call_correlation: str) -> str | None:
        with self._lock:
            return self._store.get((agent_task_id, call_correlation))

    def delete(self, agent_task_id: str, call_correlation: str) -> None:
        with self._lock:
            self._store.pop((agent_task_id, call_correlation), None)
            self._dispatches.pop((agent_task_id, call_correlation), None)

    def occurrences(self) -> list[tuple[str, str]]:
        """The occurrences whose requests the store holds."""
        with self._lock:
            return list(self._store)

    def clear(self) -> None:
        """Drop every request, as the incarnation that captured them has ended."""
        with self._lock:
            self._store.clear()
            self._dispatches.clear()
