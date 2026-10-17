"""Taking worker publications off the task-event consumer.

A report naming objects its worker stored is a publication. The consumer only receives
it: a report that can bind nothing new, as a stale one or one about a settling task, is
applied as it arrives and answered from what its task already binds, while any other is
handed to the publication lane and the consumer moves on, so a publication waiting on
the store or the control store never holds up the event stream. Its worker reports it
again until it hears it is committed or rejected.

On the lane, every object the publication names is held against retirement first, then
the report is applied as the consumer would apply it, in a transition of its own that
re-reads everything, and the answer and the release of the holds go out once that
transition is durable.
"""

import logging
import threading
import time
from collections import deque
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
from ..reclamation.lane import Admission, PublicationLane
from ..reclamation.publications import Holds, PublicationHolds, assertion_of
from ..reclamation.trees import TreeStatus
from ..task.runtime import TaskRuntime
from ..task.runtime.commits import TransitionNotDurable

# The longest a finish the stores could not take waits before it is tried again.
_FINISH_RETRY_MAX_SEC = 60.0


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

    @property
    def lane(self) -> PublicationLane:
        return self._lane

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
        failed = event.model_copy(
            update={
                "type": "TASK_FAILED",
                "error": f"artifacts_unaccepted: {reason}",
                "retryable": False,
            }
        )
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
        self._queue: deque[_Finish] = deque()
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
        with self._cv:
            self._queue.append(_Finish(task_id, holds, bound, locator))
            self._cv.notify_all()

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

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._due() or self._stopping, self._wait())
                if self._stopping:
                    return
                if (job := self._due()) is None:
                    continue
                self._queue.remove(job)
            self._finish(job)

    def _due(self) -> "_Finish | None":
        now = self._clock()
        return next((job for job in self._queue if job.at <= now), None)

    def _wait(self) -> float:
        now = self._clock()
        return max(0.05, min((job.at - now for job in self._queue), default=5.0))

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
            if outcome.status is TreeStatus.UNAVAILABLE:
                retry = "a store could not be read"
            elif outcome.status is TreeStatus.RETIRING:
                retry = f"a member of tree {manifest.content_digest} is being retired"
            elif outcome.status is TreeStatus.DONE:
                retired[manifest.content_digest] = outcome.retired
            elif manifest.content_digest in staged:
                failed[manifest.content_digest] = outcome.reason or "malformed"
            else:
                self._logger.error(
                    "Tree %s verified before its commit no longer reads: %s",
                    manifest.content_digest,
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
        with self._cv:
            job.at = self._clock() + job.backoff
            job.backoff = min(_FINISH_RETRY_MAX_SEC, job.backoff * 2)
            self._queue.append(job)
            self._cv.notify_all()


@dataclass
class _Finish:
    task_id: str
    holds: Holds
    bound: Mapping[str, frozenset[str]]
    locator: dict[str, Any]
    at: float = 0.0
    backoff: float = 1.0


__all__ = ["PublicationIntake", "TreeFinisher"]
