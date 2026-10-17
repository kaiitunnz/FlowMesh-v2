"""Pacing for the content audit's background work.

The audit shares the root's interpreter with the runtime, so it works in short bursts
and rests between them for long enough that it holds at most a ``duty`` share of the
time, never under a minimum pause. A burst counts only the CPU time its thread spent,
so time waiting on the network or the store costs no rest.
"""

import threading
import time
from collections.abc import Callable

# The most of the interpreter's time a pass takes, and the least it rests at a pause.
DUTY = 0.1
MIN_PAUSE_SEC = 0.005


class Pacer:
    """Rests after every ``every`` units of work, in proportion to the CPU time spent
    since the last rest."""

    def __init__(
        self,
        stop: threading.Event,
        every: int,
        *,
        duty: float = DUTY,
        min_pause_sec: float = MIN_PAUSE_SEC,
        clock: Callable[[], float] = time.thread_time,
    ) -> None:
        self._stop = stop
        self._every = every
        self._ratio = (1 - duty) / duty
        self._min_pause = min_pause_sec
        self._clock = clock
        self._count = 0
        self._since = clock()

    def tick(self) -> bool:
        """Count one unit of work; True once the work is to stop."""
        self._count += 1
        if self._count % self._every:
            return self._stop.is_set()
        return self.pause()

    def pause(self) -> bool:
        """Rest for the work done since the last rest; True once the work is to stop."""
        busy = self._clock() - self._since
        stopped = self._stop.wait(max(self._min_pause, busy * self._ratio))
        self._since = self._clock()
        return stopped


__all__ = ["DUTY", "MIN_PAUSE_SEC", "Pacer"]
