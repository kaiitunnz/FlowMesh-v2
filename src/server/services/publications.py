"""Taking worker publications off the task-event consumer.

A report naming objects its worker stored is a publication. A report that can bind
nothing new, as a stale one or one about a settling task, is applied as it arrives and
answered from what its task already binds; any other goes to the publication lane, off
the event stream. Its worker reports it again until it is committed or rejected.

On the lane, every object the publication names is held against retirement first, then
the report is applied in a transition of its own that re-reads everything, and the
answer and the release of the holds go out once that transition is durable.
"""

import heapq
import itertools
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from shared.content import ContentReference, StoreRefusal
from shared.schemas.event import TaskEvent
from shared.tasks.publication import (
    Publication,
    PublicationAnswer,
    PublicationDisposition,
)

from ..reclamation.epoch import EpochFenced
from ..reclamation.lane import Admission, LaneDepth, PublicationLane
from ..reclamation.publications import Holds, PublicationHolds, assertion_of
from ..reclamation.trees import TreeStatus
from ..task.runtime import TaskRuntime
from ..task.runtime.commits import TransitionNotDurable

# The longest a finish the stores could not take waits before it is tried again.
_FINISH_RETRY_MAX_SEC = 60.0


def artifacts_unaccepted(event: TaskEvent, reason: str | None) -> TaskEvent:
    """Return the failure a success settles as when its publications cannot all be
    bound."""
    return event.model_copy(
        update={
            "type": "TASK_FAILED",
            "error": f"artifacts_unaccepted: {reason}",
            "retryable": False,
        }
    )


class PublicationIntake:
    """Receives publications from the task-event consumer and commits them on the
    publication lane."""

    def __init__(
        self,
        runtime: TaskRuntime,
        holds: PublicationHolds,
        apply: Callable[[TaskEvent], None],
        lane: PublicationLane | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._runtime = runtime
        self._holds = holds
        self._apply = apply
        self._logger = logger or logging.getLogger("publications")
        self._lane = lane or PublicationLane(logger=self._logger)
        self._finisher = TreeFinisher(runtime, holds, self._logger)

    def depth(self) -> LaneDepth:
        """Return how many publications the lane holds, keeps waiting and reserves
        a slot for."""
        return self._lane.depth()

    def start(self) -> None:
        self._lane.start()
        self._finisher.start()
        self._finisher.resume_staged()

    def stop(self, timeout: float = 5.0) -> None:
        self._lane.stop(timeout)
        self._finisher.stop(timeout)

    def receive(self, event: TaskEvent, publication: Publication) -> None:
        """Apply a publication that binds nothing new, or hand it to the lane."""
        workflow_id = self._runtime.binding_workflow(
            event.task_id, publication.registration, event.dispatch_id
        )
        if workflow_id is None:
            self._apply(event)
            self._runtime.answer_publication(event.task_id, publication)
            return
        admission = self._lane.offer(
            publication.correlation,
            workflow_id,
            lambda: self._publish(event, publication),
        )
        if admission is Admission.BUSY:
            self._refuse(publication, StoreRefusal.BUSY, "the publication lane is full")

    def _publish(self, event: TaskEvent, publication: Publication) -> None:
        locator = {
            "task": event.task_id,
            "dispatch": publication.dispatch_id,
            "kind": publication.kind.value,
            "key": publication.key,
        }
        assertion = assertion_of(event.type, event.payload or {})
        preparation = self._holds.acquire(assertion, locator)
        if (holds := preparation.holds) is None:
            if preparation.malformed and event.type == "TASK_SUCCEEDED":
                self._refuse_success(event, publication, preparation.reason)
                return
            if preparation.refusal is None:
                self._answer_now(
                    PublicationAnswer(
                        publication=publication,
                        disposition=PublicationDisposition.REJECTED,
                        reason=preparation.reason,
                        digests=list(preparation.digests),
                    )
                )
                return
            self._refuse(
                publication,
                preparation.refusal,
                preparation.reason,
                list(preparation.digests),
            )
            return
        staged = [m.content_digest for m in holds.staged]

        def finish(bound: Mapping[str, frozenset[str]]) -> None:
            self._finisher.submit(event.task_id, holds, bound, locator)

        answered = False
        asserted: int | None = None
        try:
            with (
                self._runtime.event_tail(event.task_id),
                self._runtime.acknowledging(),
            ):
                asserted = self._runtime.assert_publication(event.task_id)
                self._apply(event)
                self._runtime.answer_publication(
                    event.task_id, publication, finish, staged, asserted
                )
                answered = True
        except TransitionNotDurable as exc:
            # The answer and the finish go out once the held writes land.
            self._logger.warning(
                "Publication %s of %s is not durable yet: %s",
                publication.kind,
                event.task_id,
                exc,
            )
        except EpochFenced:
            # A later root owns the holds now; this one acts on nothing more.
            return
        except Exception:
            self._logger.exception(
                "Applying publication %s of %s failed", publication.kind, event.task_id
            )
        if not answered:
            self._runtime.answer_publication(
                event.task_id, publication, finish, staged, asserted
            )

    def _refuse_success(
        self, event: TaskEvent, publication: Publication, reason: str | None
    ) -> None:
        """Fail a success naming a tree that is not one, as a success whose
        publications cannot all be bound fails, and answer it once that is durable."""
        failed = artifacts_unaccepted(event, reason)
        try:
            with (
                self._runtime.event_tail(event.task_id),
                self._runtime.acknowledging(),
            ):
                self._apply(failed)
        except TransitionNotDurable:
            pass
        except EpochFenced:
            return
        except Exception:
            self._logger.exception(
                "Failing %s for a malformed tree failed; its worker reports it again",
                event.task_id,
            )
            return
        self._runtime.answer_publication(event.task_id, publication)

    def _refuse(
        self,
        publication: Publication,
        refusal: StoreRefusal,
        reason: str | None,
        digests: list[str] | None = None,
    ) -> None:
        self._logger.info(
            "Publication %s of %s refused (%s): %s",
            publication.key,
            publication.dispatch_id,
            refusal.value,
            reason,
        )
        self._answer_now(
            PublicationAnswer(
                publication=publication,
                disposition=PublicationDisposition.REFUSED,
                refusal=refusal,
                reason=reason,
                digests=digests or [],
            )
        )

    def _answer_now(self, answer: PublicationAnswer) -> None:
        try:
            self._runtime.send_publication_answer(answer)
        except Exception as exc:
            # The worker reports it again either way.
            self._logger.debug(
                "Could not answer publication %s: %s", answer.publication, exc
            )


class TreeFinisher:
    """Finishes committed publications' holds off the delivery path: protects the
    members of the trees each bound, then ends its holds, retrying what the stores
    cannot take now."""

    def __init__(
        self,
        runtime: TaskRuntime,
        holds: PublicationHolds,
        logger: logging.Logger,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runtime = runtime
        self._holds = holds
        self._logger = logger
        self._clock = clock
        self._cv = threading.Condition()
        # Unbounded: every job carries holds only its finish releases.
        self._queue: list[tuple[float, int, _Finish]] = []
        self._seq = itertools.count()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="publication-finisher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def submit(
        self,
        task_id: str,
        holds: Holds,
        bound: Mapping[str, frozenset[str]],
        locator: dict[str, Any],
    ) -> None:
        self._push(_Finish(task_id, holds, bound, locator))

    def resume_staged(self) -> None:
        """Protect the members of every staged tree an owed upload names, as after a
        restart, holding each upload short of durable until its trees are done."""
        by_task: dict[str, list[ContentReference]] = {}
        for task_id, manifest in self._runtime.staged_trees():
            by_task.setdefault(task_id, []).append(manifest)
        for task_id, manifests in by_task.items():
            self._runtime.expanding_trees(
                task_id, [m.content_digest for m in manifests]
            )
            holds = Holds("", {}, staged=manifests)
            self.submit(
                task_id,
                holds,
                {task_id: frozenset(m.content_digest for m in manifests)},
                {"task": task_id, "kind": "expansion"},
            )

    def _push(self, job: "_Finish") -> None:
        with self._cv:
            heapq.heappush(self._queue, (job.at, next(self._seq), job))
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._due() or self._stopping, self._wait())
                if self._stopping:
                    return
                if not self._due():
                    continue
                _, _, job = heapq.heappop(self._queue)
            self._finish(job)

    def _due(self) -> bool:
        return bool(self._queue) and self._queue[0][0] <= self._clock()

    def _wait(self) -> float:
        if not self._queue:
            return 5.0
        return max(0.05, self._queue[0][0] - self._clock())

    def _finish(self, job: "_Finish") -> None:
        try:
            unfinished = self._holds.finish(
                job.holds, frozenset().union(*job.bound.values()), job.locator
            )
        except Exception as exc:
            self._retry(job, str(exc))
            return
        staged = {m.content_digest for m in job.holds.staged}
        retry: str | None = None
        failed: dict[str, str] = {}
        retired: dict[str, tuple[ContentReference, ...]] = {}
        for manifest, outcome in unfinished:
            digest = manifest.content_digest
            match outcome.status:
                case TreeStatus.FENCED:
                    # A later root holds the epoch, and finishes the tree itself.
                    self._logger.info(
                        "Leaving tree %s of %s to a later root", digest, job.task_id
                    )
                    return
                case TreeStatus.UNAVAILABLE:
                    retry = "a store could not be read"
                case TreeStatus.RETIRING:
                    retry = f"a member of tree {digest} is being retired"
                case TreeStatus.INCOMPLETE if digest in staged:
                    retry = f"a member of tree {digest} was not held"
                case TreeStatus.INCOMPLETE:
                    # Its members' holds ended unpromoted; the tree stays unexpanded
                    # for reconciliation to protect under a fresh hold.
                    self._logger.info("Tree %s is left to reconciliation", digest)
                case TreeStatus.DONE:
                    retired[digest] = outcome.retired
                case _ if digest in staged:
                    failed[digest] = outcome.reason or "malformed"
                case _:
                    self._logger.error(
                        "Tree %s verified before its commit no longer reads: %s",
                        digest,
                        outcome.reason,
                    )
        if retry is not None:
            self._retry(job, retry)
            return
        for task_id, bound in job.bound.items():
            for digest in staged & bound:
                self._runtime.tree_expanded(
                    task_id, digest, failed.get(digest), retired.get(digest, ())
                )

    def _retry(self, job: "_Finish", reason: str) -> None:
        self._logger.warning(
            "Finishing a publication of %s failed; retrying in %.0f s: %s",
            job.task_id,
            job.backoff,
            reason,
        )
        job.at = self._clock() + job.backoff
        job.backoff = min(_FINISH_RETRY_MAX_SEC, job.backoff * 2)
        self._push(job)


@dataclass
class _Finish:
    task_id: str
    holds: Holds
    bound: Mapping[str, frozenset[str]]
    locator: dict[str, Any]
    at: float = 0.0
    backoff: float = 1.0


__all__ = ["PublicationIntake", "TreeFinisher", "artifacts_unaccepted"]
