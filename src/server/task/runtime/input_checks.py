"""Control's checks of the inputs a worker could not read."""

import logging
import threading
from dataclasses import dataclass, replace

from shared.content import ContentReference
from shared.schemas.event import TaskEvent, TaskFailureKind

from ..models import TaskRecord, TaskStatus
from ..redrive import StoreRedriveScheduler
from ..results import ResultReader, ResultUnreadable
from .commits import TransitionCommitter
from .scheduling import ReadyQueue


@dataclass(frozen=True)
class _InputCheck:
    """A task held while control checks the inputs its worker could not read."""

    worker_id: str
    dispatch_id: str | None
    references: tuple[ContentReference, ...]
    # Why control found an input unreadable, once it has; the task then fails as a
    # report of the dispatch that could not read it.
    unreadable: str | None = None


INPUT_VERDICT_REPORT = "TASK_FAILED:input_unreadable"


def _input_verdict(task_id: str, check: _InputCheck) -> TaskEvent:
    """Control's report that a held task's input is unreadable, as the dispatch that
    could not read it."""
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=check.worker_id,
        dispatch_id=check.dispatch_id,
        error=f"input_unreadable: {check.unreadable}",
        retryable=False,
        failure_kind=TaskFailureKind.INPUT_UNREADABLE,
    )


class InputChecks:
    """Holds each task whose worker could not read its inputs while control reads
    them itself, and settles each check: the task runs again, waits, or fails as a
    report of the dispatch that could not read them."""

    def __init__(
        self,
        ready: ReadyQueue,
        committer: TransitionCommitter,
        tasks: dict[str, TaskRecord],
        results: ResultReader,
        redrive: StoreRedriveScheduler,
        logger: logging.Logger,
        cv: threading.Condition,
    ) -> None:
        self._ready = ready
        self._committer = committer
        self._tasks = tasks
        self._results = results
        self._redrive = redrive
        self._logger = logger
        self._cv = cv
        self.input_checks: dict[str, _InputCheck] = {}

    def settle_input_checks_locked(
        self,
        workflow_id: str,
        checks: dict[str, _InputCheck],
        verdicts: dict[str, ResultUnreadable | Exception | None],
    ) -> list[TaskEvent]:
        """Apply control's reads of the held tasks' inputs, and return the failures
        to report."""
        failures: list[TaskEvent] = []
        for task_id, check in checks.items():
            if self.input_checks.get(task_id) is not check:
                continue
            record = self._tasks.get(task_id)
            if check.unreadable is not None:
                if record is not None and (
                    record.status == TaskStatus.PENDING
                    or self._verdict_unacknowledged_locked(task_id)
                ):
                    failures.append(_input_verdict(task_id, check))
                else:
                    del self.input_checks[task_id]
                continue
            if record is None or record.status != TaskStatus.PENDING:
                del self.input_checks[task_id]
                continue
            verdict = verdicts[task_id]
            if isinstance(verdict, ResultUnreadable):
                self._logger.warning(
                    "Task %s input is unreadable at control: %s", task_id, verdict
                )
                check = self.input_checks[task_id] = replace(
                    check, unreadable=str(verdict)
                )
                failures.append(_input_verdict(task_id, check))
                continue
            if verdict is not None:
                self._logger.warning(
                    "Task %s waits: control cannot read its inputs either: %s",
                    task_id,
                    verdict,
                )
                self._redrive.schedule(workflow_id)
                continue
            self._logger.info(
                "Task %s runs again: control reads the inputs its worker could not",
                task_id,
            )
            if self._ready.enqueue_ready_locked(task_id, front=False):
                self._cv.notify_all()
            self._committer.commit_locked(task_id)
            del self.input_checks[task_id]
        if failures:
            # A later drive drops each check whose verdict has committed, or
            # reports it again.
            self._redrive.recheck(workflow_id)
        else:
            self._redrive.reset_recheck(workflow_id)
        return failures

    def _verdict_unacknowledged_locked(self, task_id: str) -> bool:
        """Whether control's verdict on a task failed a durable write and waits to be
        handled again."""
        pending = self._committer.unacknowledged.get(task_id)
        return pending is not None and pending.report == INPUT_VERDICT_REPORT

    def verify_inputs(
        self, references: tuple[ContentReference, ...]
    ) -> ResultUnreadable | Exception | None:
        """Why control cannot read these objects either, or None when it reads them."""
        pending: Exception | None = None
        for reference in references:
            try:
                self._results.verify(reference)
            except ResultUnreadable as exc:
                return exc
            except Exception as exc:
                # Unreachable, or an error that says nothing about the content itself.
                pending = exc
        return pending

    def hold_for_input_check_locked(
        self,
        record: TaskRecord,
        worker_id: str,
        dispatch_id: str | None,
        references: tuple[ContentReference, ...],
    ) -> None:
        """Keep a returned task out of the queue until control has read its inputs."""
        self._ready.remove_from_ready_locked(record.task_id)
        self.input_checks[record.task_id] = _InputCheck(
            worker_id, dispatch_id, references
        )
        self._redrive.drive_now(record.workflow_id)

    def is_input_verdict_locked(
        self,
        record: TaskRecord,
        worker_id: str,
        dispatch_id: str | None,
        failure_kind: TaskFailureKind | None,
    ) -> bool:
        """Whether a failure is control's verdict that a held task's input is
        unreadable.

        The held check stands in for the dispatch the task returned from, so the verdict
        reports as that dispatch.
        """
        check = self.input_checks.get(record.task_id)
        return (
            failure_kind is TaskFailureKind.INPUT_UNREADABLE
            and check is not None
            and check.unreadable is not None
            and record.status == TaskStatus.PENDING
            and (check.worker_id, check.dispatch_id) == (worker_id, dispatch_id)
        )

    def drop_check(self, task_id: str) -> None:
        self.input_checks.pop(task_id, None)
