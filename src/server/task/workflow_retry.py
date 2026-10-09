"""Per-workflow retries with backoff, run on one daemon thread."""

import heapq
import logging
import threading
import time
from collections.abc import Callable

DEFAULT_BASE_DELAY_SEC = 1.0
DEFAULT_MAX_DELAY_SEC = 30.0


class WorkflowRetryScheduler:
    """At most one pending retry per workflow, on one daemon thread.

    Each workflow backs off from ``base_delay_sec`` doubling to ``max_delay_sec`` across
    consecutive retries, and returns to the base delay once one gets through. A retry
    is taken off the schedule before it fires and is settled after it unless the fired
    callback schedules the next one.
    """

    def __init__(
        self,
        fire: Callable[[str], None],
        logger: logging.Logger,
        *,
        base_delay_sec: float = DEFAULT_BASE_DELAY_SEC,
        max_delay_sec: float = DEFAULT_MAX_DELAY_SEC,
        clock: Callable[[], float] = time.monotonic,
        run_thread: bool = True,
        thread_name: str = "workflow-retry",
    ) -> None:
        self._fire = fire
        self._logger = logger
        self._base = base_delay_sec
        self._max = max_delay_sec
        self._clock = clock
        self._run_thread = run_thread
        self._thread_name = thread_name
        self._cv = threading.Condition()
        self._due: dict[str, float] = {}
        self._heap: list[tuple[float, str]] = []
        self._streak: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stopped = False

    def schedule(self, workflow_id: str) -> None:
        """Retry a workflow after its backoff; a workflow already waiting keeps its one
        slot."""
        with self._cv:
            if self._stopped or workflow_id in self._due:
                return
            streak = self._streak.get(workflow_id, 0) + 1
            self._streak[workflow_id] = streak
            delay = self._delay(streak)
            self._enqueue_locked(workflow_id, self._clock() + delay)
            self._on_scheduled(workflow_id, streak, delay)

    def drive_now(self, workflow_id: str) -> None:
        """Retry a workflow as soon as the thread is free.

        A workflow already waiting keeps its slot and its backoff.
        """
        with self._cv:
            if self._stopped or workflow_id in self._due:
                return
            self._enqueue_locked(workflow_id, self._clock())

    def settle(self, workflow_id: str) -> None:
        """Drop a workflow's pending retry and backoff."""
        with self._cv:
            self._due.pop(workflow_id, None)
            self._streak.pop(workflow_id, None)

    def pending(self, workflow_id: str) -> bool:
        with self._cv:
            return workflow_id in self._due

    def run_due(self) -> list[str]:
        """Fire every retry that has come due, returning the workflows it fired."""
        with self._cv:
            due = self._take_due_locked(self._clock())
        for workflow_id in due:
            try:
                self._fire(workflow_id)
            except Exception:
                self._logger.exception("Retrying workflow %s failed", workflow_id)
            if not self.pending(workflow_id):
                self.settle(workflow_id)
        return due

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._due.clear()
            self._heap.clear()
            self._clear_locked()
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _delay(self, streak: int) -> float:
        return min(self._max, self._base * 2 ** (streak - 1))

    def _on_scheduled(self, workflow_id: str, streak: int, delay: float) -> None:
        """Called under the scheduler's lock after a backed-off retry is scheduled."""

    def _clear_locked(self) -> None:
        self._streak.clear()

    def _enqueue_locked(self, workflow_id: str, due: float) -> None:
        self._due[workflow_id] = due
        heapq.heappush(self._heap, (due, workflow_id))
        self._ensure_thread()
        self._cv.notify_all()

    def _take_due_locked(self, now: float) -> list[str]:
        due: list[str] = []
        while self._heap and self._heap[0][0] <= now:
            at, workflow_id = heapq.heappop(self._heap)
            if self._due.get(workflow_id) == at:
                del self._due[workflow_id]
                due.append(workflow_id)
        return due

    def _ensure_thread(self) -> None:
        if not self._run_thread or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=self._thread_name, daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._stopped and not self._heap:
                    self._cv.wait()
                if self._stopped:
                    return
                wait = self._heap[0][0] - self._clock()
                if wait > 0:
                    self._cv.wait(wait)
                    continue
            self.run_due()


__all__ = ["WorkflowRetryScheduler"]
