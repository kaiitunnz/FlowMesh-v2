"""A bounded lane that runs publications off the task-event consumer.

The lane holds at most ``capacity`` publications, admitted or reserved, and runs them on
a fixed set of threads. A publication offered while it is already held coalesces with
it. One offered while the lane is full waits: it is told to come back, and the slot
each finished publication frees is reserved, in turn across workflows, for the oldest
publication waiting in that workflow, which its next offer takes. A waiting publication
keeps its place across offers, so a continuously offered one is never overtaken
indefinitely; a reservation its source does not take up within ``reservation_sec``
lapses, and that publication stops waiting.
"""

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable
from enum import StrEnum

# How many publications run at once.
THREADS = 4
# How many publications the lane holds, running, queued or reserved.
CAPACITY = 256
# How long a slot stays reserved for a waiting publication's next offer.
RESERVATION_SEC = 15.0


class Admission(StrEnum):
    ADMITTED = "admitted"
    COALESCED = "coalesced"
    """The same publication is already held; the held one answers."""
    BUSY = "busy"
    """The lane is full; the publication waits for its turn and is offered again."""


class PublicationLane:
    """Runs offered publications on bounded threads with fair admission."""

    def __init__(
        self,
        threads: int = THREADS,
        capacity: int = CAPACITY,
        reservation_sec: float = RESERVATION_SEC,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threads < 1 or capacity < threads:
            raise ValueError("a lane needs a thread and room for each")
        self._threads_wanted = threads
        self._capacity = capacity
        self._reservation_sec = reservation_sec
        self._logger = logger or logging.getLogger("publication-lane")
        self._clock = clock
        self._cv = threading.Condition()
        self._held: dict[Hashable, Callable[[], None] | None] = {}
        self._queue: deque[Hashable] = deque()
        # The publications waiting for a slot, oldest first within each workflow, and
        # the workflows in the order their turns come.
        self._waiting: dict[str, OrderedDict[Hashable, None]] = {}
        self._turns: deque[str] = deque()
        self._reserved: dict[Hashable, tuple[str, float]] = {}
        self._stopping = False
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for index in range(self._threads_wanted):
            thread = threading.Thread(
                target=self._run, name=f"publication-lane-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop taking publications and let the running ones finish within
        ``timeout``; queued ones are dropped, as their sources offer them again."""
        with self._cv:
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()
        deadline = self._clock() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - self._clock()))

    def offer(
        self, key: Hashable, workflow_id: str, run: Callable[[], None]
    ) -> Admission:
        """Offer a publication identified by ``key``; ``run`` runs it if admitted."""
        with self._cv:
            if key in self._held:
                return Admission.COALESCED
            if self._stopping:
                return Admission.BUSY
            self._lapse()
            if self._reserved.pop(key, None) is not None or (
                not self._turns and self._free() > 0
            ):
                self._unwait(workflow_id, key)
                self._held[key] = run
                self._queue.append(key)
                self._cv.notify()
                return Admission.ADMITTED
            self._wait(workflow_id, key)
            self._grant()
            return Admission.BUSY

    def depth(self) -> tuple[int, int, int]:
        """Return how many publications are held, waiting and reserved."""
        with self._cv:
            waiting = sum(len(keys) for keys in self._waiting.values())
            return len(self._held), waiting, len(self._reserved)

    def _free(self) -> int:
        return self._capacity - len(self._held) - len(self._reserved)

    def _wait(self, workflow_id: str, key: Hashable) -> None:
        if (keys := self._waiting.get(workflow_id)) is None:
            keys = self._waiting[workflow_id] = OrderedDict()
            self._turns.append(workflow_id)
        keys.setdefault(key, None)

    def _unwait(self, workflow_id: str, key: Hashable) -> None:
        if (keys := self._waiting.get(workflow_id)) is None:
            return
        keys.pop(key, None)
        if not keys:
            del self._waiting[workflow_id]
            self._turns.remove(workflow_id)

    def _grant(self) -> None:
        """Reserve each free slot for the next workflow's oldest unreserved
        publication."""
        passed = 0
        while self._free() > 0 and passed < len(self._turns):
            workflow_id = self._turns[0]
            self._turns.rotate(-1)
            head = next(
                (k for k in self._waiting[workflow_id] if k not in self._reserved), None
            )
            if head is None:
                passed += 1
                continue
            passed = 0
            self._reserved[head] = (workflow_id, self._clock() + self._reservation_sec)

    def _lapse(self) -> None:
        now = self._clock()
        lapsed = [
            (key, workflow_id)
            for key, (workflow_id, deadline) in self._reserved.items()
            if deadline <= now
        ]
        for key, workflow_id in lapsed:
            del self._reserved[key]
            self._unwait(workflow_id, key)
        if lapsed:
            self._grant()

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._stopping)
                if not self._queue:
                    return
                key = self._queue.popleft()
                run = self._held[key]
            try:
                if run is not None:
                    run()
            except Exception:
                self._logger.exception("A publication failed unexpectedly")
            finally:
                with self._cv:
                    del self._held[key]
                    self._lapse()
                    self._grant()


__all__ = ["Admission", "PublicationLane"]
