"""Driving a workflow's advances that wait on a read of stored results.

A spawn's fan-out and an agent's bound inputs are read from the shared store, and a read
never runs under the runtime lock, so those advances are driven here, off it. A settled
producer's result is in the store before its success is reported, so a read that
cannot reach the store is a pause, not a loss: the advance waits and is driven again
once the store answers. Nothing else re-fires those advances — the producer has already
settled — so this keeps one pending drive per workflow, backing off while the store
stays away.

A re-drive is not durable and does not need to be: a restart re-drives every workflow
with a settled producer's unsealed spawn or an agent waiting on its inputs.
"""

import heapq
import logging
import threading
import time
from collections.abc import Callable

_BASE_DELAY_SEC = 1.0
_MAX_DELAY_SEC = 30.0
# Consecutive re-drives after which a workflow still waiting on the store is reported.
_WARN_AFTER = 5


class StoreRedriveScheduler:
    """At most one pending re-drive per workflow, on one daemon thread.

    Each workflow backs off from ``base_delay_sec`` doubling to ``max_delay_sec`` for as
    long as its re-drives keep finding the store unreachable, and returns to the base
    delay once one gets through. A workflow still waiting after ``warn_after`` re-drives
    is logged at warning on every re-drive, so a store that is not coming back — or an
    error that never clears — is visible rather than a quiet stall.
    """

    def __init__(
        self,
        fire: Callable[[str], None],
        logger: logging.Logger,
        *,
        base_delay_sec: float = _BASE_DELAY_SEC,
        max_delay_sec: float = _MAX_DELAY_SEC,
        warn_after: int = _WARN_AFTER,
        clock: Callable[[], float] = time.monotonic,
        run_thread: bool = True,
    ) -> None:
        self._fire = fire
        self._logger = logger
        self._base = base_delay_sec
        self._max = max_delay_sec
        self._warn_after = warn_after
        self._clock = clock
        self._run_thread = run_thread
        self._cv = threading.Condition()
        self._due: dict[str, float] = {}
        self._heap: list[tuple[float, str]] = []
        self._streak: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stopped = False

    def schedule(self, workflow_id: str) -> None:
        """Re-drive a workflow later; a workflow already waiting keeps its one slot."""
        with self._cv:
            if self._stopped or workflow_id in self._due:
                return
            streak = self._streak.get(workflow_id, 0) + 1
            self._streak[workflow_id] = streak
            delay = min(self._max, self._base * 2 ** (streak - 1))
            due = self._clock() + delay
            self._due[workflow_id] = due
            heapq.heappush(self._heap, (due, workflow_id))
            if streak >= self._warn_after:
                self._logger.warning(
                    "Workflow %s is still waiting on the content store after %d "
                    "re-drives; next in %.0fs",
                    workflow_id,
                    streak,
                    delay,
                )
            self._ensure_thread()
            self._cv.notify_all()

    def drive_now(self, workflow_id: str) -> None:
        """Drive a workflow as soon as the re-drive thread is free.

        A workflow already waiting keeps its slot and its backoff: the drive it is
        waiting for reads everything the workflow waits on, and a backoff means the
        store is away for this read as much as for the last.
        """
        with self._cv:
            if self._stopped or workflow_id in self._due:
                return
            due = self._clock()
            self._due[workflow_id] = due
            heapq.heappush(self._heap, (due, workflow_id))
            self._ensure_thread()
            self._cv.notify_all()

    def settle(self, workflow_id: str) -> None:
        """Drop a workflow's pending re-drive and backoff; it no longer waits."""
        with self._cv:
            self._due.pop(workflow_id, None)
            self._streak.pop(workflow_id, None)

    def pending(self, workflow_id: str) -> bool:
        with self._cv:
            return workflow_id in self._due

    def run_due(self) -> list[str]:
        """Fire every re-drive that has come due, returning the workflows it drove.

        A workflow is taken off the schedule before it fires, so a re-drive that finds
        the store away again schedules its next one; one that gets through is settled.
        """
        with self._cv:
            due = self._take_due_locked(self._clock())
        for workflow_id in due:
            try:
                self._fire(workflow_id)
            except Exception:
                self._logger.exception("Re-driving workflow %s failed", workflow_id)
            if not self.pending(workflow_id):
                self.settle(workflow_id)
        return due

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._due.clear()
            self._heap.clear()
            self._streak.clear()
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

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
            target=self._loop, name="store-redrive", daemon=True
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


__all__ = ["StoreRedriveScheduler"]
