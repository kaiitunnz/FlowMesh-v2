"""The worker-local sandbox host allocation.

A sandbox host is a resident allocation holding one worker's reusable sandbox runtime
state and its concurrent-session capacity. It reports itself live so admission can place
sessions on this worker, then holds the allocation until its lifecycle drains or stops
it. Each admitted session runs in its own private tree under the worker's private-state
root; the host owns none of that state.
"""

import logging
import threading
import time
from pathlib import Path
from typing import ClassVar

from shared.schemas.result import SandboxHostResult
from shared.tasks.specs import SandboxHostSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import parse_float_env

from .base_executor import Executor, ExecutorTask, TaskCancelledError

logger = logging.getLogger("sandbox-host-executor")

_DEFAULT_TTL_SEC = 24 * 60 * 60.0
_POLL_INTERVAL_SEC = 0.5


class SandboxHostExecutor(Executor):
    """Holds a sandbox host allocation open for the sessions admitted to it."""

    name = "sandbox_host"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.SANDBOX_HOST}
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()

    def run(self, task: ExecutorTask, out_dir: Path) -> SandboxHostResult:
        spec = self.require_spec(task, SandboxHostSpecStrict)
        ttl_sec = spec.ttlSeconds or parse_float_env(
            "SERVE_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._stop_event.is_set():
            raise TaskCancelledError(
                f"sandbox host {task.task_id} stopped before it opened"
            )
        self.emit_update(task.task_id, {"sandbox_host": {"profile": spec.profile}})
        try:
            self._hold(ttl_sec)
        finally:
            self._cancel_event.clear()
            self._stop_event.clear()
        return SandboxHostResult(profile=spec.profile)

    def _hold(self, ttl_sec: float) -> None:
        deadline = time.time() + ttl_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("sandbox host cancelled")
            if self._stop_event.is_set():
                logger.info("sandbox host stop requested; releasing the allocation")
                return
            time.sleep(_POLL_INTERVAL_SEC)
        logger.info("sandbox host TTL reached; releasing the allocation")

    def cancel(self, task_id: str) -> None:
        self._cancel_event.set()

    def stop(self, task_id: str) -> None:
        self._stop_event.set()
