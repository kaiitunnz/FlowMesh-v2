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

import logging
import time
from collections.abc import Callable

from .workflow_retry import (
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_MAX_DELAY_SEC,
    WorkflowRetryScheduler,
)

# Consecutive re-drives after which a workflow still waiting on the store is reported.
_WARN_AFTER = 5


class StoreRedriveScheduler(WorkflowRetryScheduler):
    """Re-drives each workflow waiting on the content store.

    A workflow still waiting after ``warn_after`` re-drives is logged at warning on
    every re-drive, so a store that is not coming back, or an error that never clears,
    is visible.
    """

    def __init__(
        self,
        fire: Callable[[str], None],
        logger: logging.Logger,
        *,
        base_delay_sec: float = DEFAULT_BASE_DELAY_SEC,
        max_delay_sec: float = DEFAULT_MAX_DELAY_SEC,
        warn_after: int = _WARN_AFTER,
        clock: Callable[[], float] = time.monotonic,
        run_thread: bool = True,
    ) -> None:
        super().__init__(
            fire,
            logger,
            base_delay_sec=base_delay_sec,
            max_delay_sec=max_delay_sec,
            clock=clock,
            run_thread=run_thread,
            thread_name="store-redrive",
        )
        self._warn_after = warn_after
        self._recheck_streak: dict[str, int] = {}

    def recheck(self, workflow_id: str) -> None:
        """Re-drive a workflow to confirm what it reported, on a backoff of its own.

        The delay doubles from the base to the cap across consecutive rechecks, apart
        from the store-wait backoff and its warning. A workflow already waiting keeps
        its slot.
        """
        with self._cv:
            if self._stopped or workflow_id in self._due:
                return
            streak = self._recheck_streak.get(workflow_id, 0) + 1
            self._recheck_streak[workflow_id] = streak
            self._enqueue_locked(workflow_id, self._clock() + self._delay(streak))

    def settle(self, workflow_id: str) -> None:
        """Drop a workflow's pending re-drive and backoff."""
        with self._cv:
            super().settle(workflow_id)
            self._recheck_streak.pop(workflow_id, None)

    def reset_recheck(self, workflow_id: str) -> None:
        """Start a workflow's recheck backoff over; it has nothing left to confirm."""
        with self._cv:
            self._recheck_streak.pop(workflow_id, None)

    def _on_scheduled(self, workflow_id: str, streak: int, delay: float) -> None:
        if streak >= self._warn_after:
            self._logger.warning(
                "Workflow %s is still waiting on the content store after %d "
                "re-drives; next in %.0fs",
                workflow_id,
                streak,
                delay,
            )

    def _clear_locked(self) -> None:
        super()._clear_locked()
        self._recheck_streak.clear()


__all__ = ["StoreRedriveScheduler"]
