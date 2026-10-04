"""The keys this worker's serve engines accept, held only inside the worker."""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalEngineKey:
    """The key of an engine this worker launched; ``None`` once that engine stopped."""

    key: str | None


class EngineKeyRegistry:
    """The engine key of each serve task this worker runs, keyed by its task id.

    A serve executor publishes the key its engine was launched with, and the replica
    sidecar resolves it when it binds to that engine, so the key stays in the worker.
    A withdrawn key leaves its task marked as launched here, so a late bind to that
    task's port is refused rather than reaching whatever listens there next.
    """

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._withdrawn: set[str] = set()
        self._lock = threading.Lock()

    def publish(self, serve_task_id: str, key: str) -> None:
        with self._lock:
            self._keys[serve_task_id] = key
            self._withdrawn.discard(serve_task_id)

    def withdraw(self, serve_task_id: str) -> None:
        with self._lock:
            if self._keys.pop(serve_task_id, None) is not None:
                self._withdrawn.add(serve_task_id)

    def resolve(self, serve_task_id: str) -> str | None:
        with self._lock:
            return self._keys.get(serve_task_id)

    def lookup(self, serve_task_id: str) -> LocalEngineKey | None:
        """The key of the engine this worker launched for a task, if it launched one."""
        with self._lock:
            if (key := self._keys.get(serve_task_id)) is not None:
                return LocalEngineKey(key)
            if serve_task_id in self._withdrawn:
                return LocalEngineKey(None)
            return None
