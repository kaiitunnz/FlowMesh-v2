"""Freeing a settled workflow's runtime state once nothing needs it.

A workflow leaves memory only after a certified copy of what its reads derive from is
in the store. Its runtime state is sealed while that copy is published, so no runtime
entry changes it meanwhile, and a publication the store did not certify leaves it
resident and unsealed for the next attempt.
"""

import logging
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

from ...registries.history import Certification, HistoryPublication
from ...registries.workflow import WorkflowRegistry
from .commits import store_unavailable

# A workflow not freed by a request is evaluated again this often.
SWEEP_INTERVAL_SEC = 30.0
# The longest the freeing thread waits before it looks for a stop.
_POLL_SEC = 1.0


@dataclass(frozen=True)
class ReclaimCheck:
    """What a reclaimable workflow's eligibility rests on outside the runtime: the
    invocations whose resident claims must all be terminal."""

    invocation_ids: frozenset[str]


@dataclass(frozen=True)
class Freed:
    """A workflow's removed tasks, and how many of them succeeded."""

    task_ids: list[str]
    done: int


@dataclass
class _Seal:
    """A workflow whose publication is under way, and whether it holds a
    certificate is to be read back."""

    ambiguous: bool = False


class SettledHistory:
    """Frees each settled workflow's runtime state behind its certified history.

    A workflow is evaluated when a request names it and again on every sweep until it
    is freed. Every field but the request queue is read and changed under the runtime
    lock, which ``evaluate`` takes itself.
    """

    def __init__(
        self,
        holds: Callable[[str], bool],
        check: Callable[[str], ReclaimCheck | None],
        publication: Callable[[str, int], HistoryPublication],
        forget: Callable[[str], Freed],
        registry: WorkflowRegistry,
        lock: threading.RLock,
        logger: logging.Logger,
    ) -> None:
        # Each reads or changes the runtime's state under its lock: whether it holds a
        # workflow, whether nothing in it still needs the workflow's state, the history
        # a sealed workflow's reads derive from, certified under a sequence,
        # which takes the lock itself, and its removal.
        self._holds = holds
        self._check = check
        self._publication = publication
        self._forget = forget
        self._registry = registry
        self._lock = lock
        self._logger = logger
        self._cv = threading.Condition()
        self._requested: set[str] = set()
        self._candidates: set[str] = set()
        self._started = False
        self._next_sweep = time.monotonic() + SWEEP_INTERVAL_SEC
        # What the runtime lock guards.
        self._sealed: dict[str, _Seal] = {}
        self._tails: dict[str, int] = {}
        self._closed: set[str] = set()
        self._archived: set[str] = set()
        self._archived_count: dict[str, int] = {}
        self._next_seq = 1
        self._published_through = 0
        # Every certificate sequenced at or below this is in the counts read at start.
        self._counted_through = 0
        self._total = 0
        self._done = 0
        self._requires_closure = True
        self._requires_archive = True
        self._claims_settled: Callable[[Collection[str]], bool] | None = None
        self._on_freed: Callable[[list[str]], None] | None = None

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

    def start(self, seq: int, total: int, done: int) -> None:
        """Begin freeing, numbering certificates after ``seq`` and counting the
        ``total`` freed tasks, ``done`` of them succeeded, already stored."""
        with self._lock:
            self._next_seq = max(self._next_seq, seq + 1)
            self._published_through = max(self._published_through, seq)
            self._counted_through = seq
            self._total, self._done = total, done
        with self._cv:
            self._started = True
            self._cv.notify_all()

    def require(self, closure: bool, archive: bool) -> None:
        """Set whether each workflow is kept until its closure is reported, and until
        every task's archive disposition is; both are by default."""
        with self._lock:
            self._requires_closure = closure
            self._requires_archive = archive

    def set_claims_settled(self, settled: Callable[[Collection[str]], bool]) -> None:
        """Install the check that every resident claim the given invocations raised
        is terminal."""
        self._claims_settled = settled

    def set_on_freed(self, on_freed: Callable[[list[str]], None]) -> None:
        """Install the callback told the tasks of each workflow freed, off the lock."""
        self._on_freed = on_freed

    # ------------------------------------------------------------------ #
    # Requests and reports
    # ------------------------------------------------------------------ #

    def request(self, workflow_id: str) -> None:
        """Ask for a workflow to be freed once it can be. Cheap and safe from any
        thread, the runtime lock held or not."""
        with self._cv:
            self._candidates.add(workflow_id)
            self._requested.add(workflow_id)
            self._cv.notify_all()

    def closed_locked(self, workflow_id: str) -> bool:
        """Record a workflow's closure; returns whether it was new."""
        if workflow_id in self._closed:
            return False
        self._closed.add(workflow_id)
        return True

    def archived_locked(self, task_id: str, workflow_id: str, tasks: int) -> bool:
        """Record that a task of a workflow holding ``tasks`` tasks is archived;
        returns whether every one of them is."""
        if task_id not in self._archived:
            self._archived.add(task_id)
            self._archived_count[workflow_id] = (
                self._archived_count.get(workflow_id, 0) + 1
            )
        return self._archived_count[workflow_id] >= tasks

    def hold_locked(self, workflow_id: str) -> None:
        """Keep a workflow resident until a matching ``release_locked``."""
        self._tails[workflow_id] = self._tails.get(workflow_id, 0) + 1

    def release_locked(self, workflow_id: str) -> bool:
        """Release one hold; returns whether the workflow is held no more."""
        if (count := self._tails.get(workflow_id, 0) - 1) > 0:
            self._tails[workflow_id] = count
            return False
        self._tails.pop(workflow_id, None)
        return True

    # ------------------------------------------------------------------ #
    # Reads the runtime makes under its lock
    # ------------------------------------------------------------------ #

    def sealed_locked(self, workflow_id: str) -> bool:
        return workflow_id in self._sealed

    def sealed_ids_locked(self) -> frozenset[str]:
        return frozenset(self._sealed)

    def blocks_locked(self, workflow_id: str, tasks: int) -> bool:
        """Whether a hold, a missing closure or a missing archive disposition of one
        of its ``tasks`` tasks keeps the workflow resident."""
        return (
            workflow_id in self._tails
            or (self._requires_closure and workflow_id not in self._closed)
            or (
                self._requires_archive
                and self._archived_count.get(workflow_id, 0) < tasks
            )
        )

    @property
    def published_through(self) -> int:
        """The highest sequence of a freed workflow, read under the runtime lock."""
        return self._published_through

    def counts_locked(self) -> tuple[int, int]:
        """How many freed tasks there are, and how many of them succeeded."""
        return self._total, self._done

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def run(self, stop_event: threading.Event) -> None:
        """Free requested workflows until stopped, sweeping every candidate on each
        interval."""
        while not stop_event.is_set():
            self.drain(min(_POLL_SEC, max(0.0, self._next_sweep - time.monotonic())))

    def sweep(self) -> None:
        """Evaluate every workflow not yet freed on the next drain."""
        with self._cv:
            self._requested |= self._candidates
            self._cv.notify_all()

    def drain(self, timeout: float = 0.0) -> None:
        """Evaluate every requested workflow, waiting up to ``timeout`` for one, and
        every candidate when a sweep is due."""
        with self._cv:
            self._cv.wait_for(lambda: self._started and bool(self._requested), timeout)
            if not self._started:
                return
            if (now := time.monotonic()) >= self._next_sweep:
                self._requested |= self._candidates
                self._next_sweep = now + SWEEP_INTERVAL_SEC
            pending, self._requested = self._requested, set()
        for workflow_id in sorted(pending):
            try:
                self.evaluate(workflow_id)
            except Exception as exc:
                if not store_unavailable(exc):
                    self._logger.exception("Freeing workflow %s failed", workflow_id)
                    continue
                # Every candidate stays for the next sweep.
                self._logger.warning(
                    "Freeing settled workflows waits for the store: %s", exc
                )
                return

    def evaluate(self, workflow_id: str) -> bool:
        """Free the workflow if nothing needs its runtime state; returns whether it is
        freed. A failure leaves it resident for a later evaluation."""
        with self._lock:
            held = self._holds(workflow_id)
            seal = self._sealed.get(workflow_id)
            check = None if seal or not held else self._check(workflow_id)
        if not held:
            with self._cv:
                self._candidates.discard(workflow_id)
            return False
        if seal is not None:
            return seal.ambiguous and self._resolve_ambiguous(workflow_id)
        if check is None:
            return False
        if check.invocation_ids and self._claims_settled is not None:
            if not self._claims_settled(check.invocation_ids):
                return False
        revision = self._registry.workflow_revision(workflow_id)
        with self._lock:
            if self._check(workflow_id) != check:
                return False
            seq = self._next_seq
            self._next_seq += 1
            seal = self._sealed[workflow_id] = _Seal()
        try:
            publication = self._publication(workflow_id, seq)
        except Exception:
            self._unseal(workflow_id, "its history could not be read")
            raise
        try:
            prepared = self._registry.prepare_history(publication)
        except Exception as exc:
            self._unseal(workflow_id, "its preparation failed", exc)
            return False
        if not prepared:
            return self._read_back(workflow_id, seal)
        try:
            outcome = self._registry.certify_history(publication, revision)
        except Exception as exc:
            self._logger.warning(
                "Certifying workflow %s's history got no answer; reading it back: %s",
                workflow_id,
                exc,
            )
            return self._read_back(workflow_id, seal)
        if outcome == Certification.CERTIFIED:
            return self._read_back(workflow_id, seal)
        if outcome != Certification.OK:
            self._unseal(workflow_id, f"its certification was refused ({outcome})")
            return False
        self._free(workflow_id, publication.certificate.seq)
        return True

    def _read_back(self, workflow_id: str, seal: _Seal) -> bool:
        with self._lock:
            seal.ambiguous = True
        return self._resolve_ambiguous(workflow_id)

    def _resolve_ambiguous(self, workflow_id: str) -> bool:
        """Settle whether a sealed workflow holds a certificate by reading it back.
        Any certificate frees the workflow: every publisher certifies a settled
        workflow at a revision nothing wrote since, whether it is this attempt, an
        earlier one whose answer was lost, or an earlier process. None unseals it,
        and an unreadable one keeps it sealed for the next evaluation."""
        try:
            certificate = self._registry.history_certificate(workflow_id)
        except Exception as exc:
            self._logger.warning(
                "Workflow %s stays sealed; its certificate could not be read: %s",
                workflow_id,
                exc,
            )
            return False
        if certificate is None:
            self._unseal(workflow_id, "its certification did not land")
            return False
        self._free(workflow_id, certificate.seq)
        return True

    def _unseal(
        self, workflow_id: str, why: str, error: BaseException | None = None
    ) -> None:
        with self._lock:
            self._sealed.pop(workflow_id, None)
        self._logger.info(
            "Workflow %s stays resident: %s%s",
            workflow_id,
            why,
            f": {error}" if error is not None else "",
        )

    def _free(self, workflow_id: str, seq: int) -> None:
        with self._lock:
            freed = self._forget(workflow_id)
            del self._sealed[workflow_id]
            self._tails.pop(workflow_id, None)
            self._closed.discard(workflow_id)
            self._archived.difference_update(freed.task_ids)
            self._archived_count.pop(workflow_id, None)
            self._published_through = max(self._published_through, seq)
            if seq > self._counted_through:
                self._total += len(freed.task_ids)
                self._done += freed.done
        with self._cv:
            self._candidates.discard(workflow_id)
            self._requested.discard(workflow_id)
        self._logger.debug(
            "Freed workflow %s (%d tasks)", workflow_id, len(freed.task_ids)
        )
        if self._on_freed is not None:
            try:
                self._on_freed(freed.task_ids)
            except Exception:
                self._logger.exception(
                    "Reporting the freed tasks of workflow %s failed", workflow_id
                )


__all__ = [
    "Freed",
    "ReclaimCheck",
    "SWEEP_INTERVAL_SEC",
    "SettledHistory",
]
