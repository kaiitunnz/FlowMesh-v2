"""The keys this worker's serve engines accept, held only inside the worker."""

import threading


class EngineKeyRegistry:
    """The engine key of each serve task this worker runs, keyed by its task id.

    A serve executor publishes the key its engine was launched with, and the replica
    sidecar resolves it when it binds to that engine, so the key stays in the worker.
    """

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()

    def publish(self, serve_task_id: str, key: str) -> None:
        with self._lock:
            self._keys[serve_task_id] = key

    def withdraw(self, serve_task_id: str) -> None:
        with self._lock:
            self._keys.pop(serve_task_id, None)

    def resolve(self, serve_task_id: str) -> str | None:
        with self._lock:
            return self._keys.get(serve_task_id)
