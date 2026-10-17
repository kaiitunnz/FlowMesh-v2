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
from collections.abc import Callable

from shared.schemas.event import TaskEvent
from shared.tasks.publication import (
    BUSY_REASON,
    Publication,
    PublicationAnswer,
    PublicationDisposition,
)

from ..reclamation.epoch import EpochFenced
from ..reclamation.lane import Admission, PublicationLane
from ..reclamation.publications import PublicationHolds, assertion_of
from ..task.runtime import TaskRuntime
from ..task.runtime.commits import TransitionNotDurable


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

    @property
    def lane(self) -> PublicationLane:
        return self._lane

    def start(self) -> None:
        self._lane.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._lane.stop(timeout)

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
            self._answer_now(
                publication,
                PublicationDisposition.RETRY,
                BUSY_REASON,
            )

    def _publish(self, event: TaskEvent, publication: Publication) -> None:
        locator = {
            "task": event.task_id,
            "dispatch": publication.dispatch_id,
            "kind": publication.kind.value,
            "key": publication.key,
        }
        preparation = self._holds.acquire(
            assertion_of(event.type, event.payload or {}), locator
        )
        if (holds := preparation.holds) is None:
            self._answer_now(
                publication,
                preparation.refusal or PublicationDisposition.RETRY,
                preparation.reason,
                list(preparation.digests),
            )
            return

        def release() -> None:
            self._holds.release(holds)

        try:
            with (
                self._runtime.event_tail(event.task_id),
                self._runtime.acknowledging(),
            ):
                self._runtime.assert_publication(event.task_id)
                self._apply(event)
                self._runtime.answer_publication(
                    event.task_id, publication, release if holds else None
                )
        except TransitionNotDurable as exc:
            # The answer and the release go out once the held writes land.
            self._logger.warning(
                "Publication %s of %s is not durable yet: %s",
                publication.kind,
                event.task_id,
                exc,
            )
            self._runtime.answer_publication(
                event.task_id, publication, release if holds else None
            )
        except EpochFenced:
            # A later root owns the holds now; this one acts on nothing more.
            return
        except Exception:
            self._logger.exception(
                "Applying publication %s of %s failed", publication.kind, event.task_id
            )
            if holds:
                self._runtime.file_hold_release(event.task_id, release)

    def _answer_now(
        self,
        publication: Publication,
        disposition: PublicationDisposition,
        reason: str | None,
        digests: list[str] | None = None,
    ) -> None:
        try:
            self._runtime.send_publication_answer(
                PublicationAnswer(
                    publication=publication,
                    disposition=disposition,
                    reason=reason,
                    digests=digests or [],
                )
            )
        except Exception as exc:
            # The worker reports it again either way.
            self._logger.debug("Could not answer publication %s: %s", publication, exc)


__all__ = ["PublicationIntake"]
