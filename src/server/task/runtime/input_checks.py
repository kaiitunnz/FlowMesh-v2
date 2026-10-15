"""Control's checks of the inputs a worker could not read."""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from shared.content import ContentReference
from shared.schemas.event import TaskEvent, TaskFailureKind
from shared.tasks.artifact_binding import ArtifactBinding

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
    # The artifact members among them, which control checks for presence and size
    # only; their worker's verified read judges their bytes.
    members: frozenset[ContentReference] = frozenset()
    # Why control found an input unreadable, once it has; the task then fails as a
    # report of the dispatch that could not read it.
    unreadable: str | None = None
    # The upstream tasks whose accepted artifacts the task reads and whose upload is
    # still owed; control reads nothing until they settle.
    awaiting: frozenset[str] = frozenset()
    # The accepted artifacts the task reads, whose loss an unreadable verdict names.
    reads: tuple[ArtifactBinding, ...] = ()

    @property
    def checkable(self) -> bool:
        """Whether control reads the check's inputs now."""
        return not self.awaiting


INPUT_VERDICT_REPORT = "TASK_FAILED:input_unreadable"
# A task whose inputs read fine at control is placed again no sooner than this, so a
# worker that keeps failing to read them is not handed the task in a tight loop.
INPUT_RECHECK_PACE_SEC = 1.0


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
        lost_reads: Callable[[Iterable[ArtifactBinding]], str | None],
    ) -> None:
        self._ready = ready
        self._committer = committer
        self._tasks = tasks
        self._results = results
        self._redrive = redrive
        self._logger = logger
        self._cv = cv
        self._lost_reads = lost_reads
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
                lost = self._lost_reads(check.reads)
                check = self.input_checks[task_id] = replace(
                    check,
                    unreadable=(
                        f"{lost}: {verdict}" if lost is not None else str(verdict)
                    ),
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
            if self._ready.enqueue_ready_locked(
                task_id,
                front=False,
                not_before=time.monotonic() + INPUT_RECHECK_PACE_SEC,
            ):
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

    def verify_inputs(self, check: _InputCheck) -> ResultUnreadable | Exception | None:
        """Return why control cannot read a check's objects either, or None when it
        reads them."""
        pending: Exception | None = None
        for reference in check.references:
            try:
                if reference in check.members:
                    self._results.verify_stored(reference)
                else:
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
        members: frozenset[ContentReference] = frozenset(),
        awaiting: frozenset[str] = frozenset(),
        reads: tuple[ArtifactBinding, ...] = (),
    ) -> None:
        """Keep a returned task out of the queue until control has read its inputs,
        or, while uploads it reads are still owed in ``awaiting``, until they settle.
        ``reads`` are the accepted artifacts it reads."""
        self._ready.remove_from_ready_locked(record.task_id)
        self.input_checks[record.task_id] = _InputCheck(
            worker_id, dispatch_id, references, members, awaiting=awaiting, reads=reads
        )
        if not awaiting:
            self._redrive.drive_now(record.workflow_id)

    def wake_uploaded_locked(self, producer_id: str) -> None:
        """Let the checks waiting on a task's upload read their inputs, now that it
        settled."""
        for task_id, check in list(self.input_checks.items()):
            if producer_id not in check.awaiting:
                continue
            check = self.input_checks[task_id] = replace(
                check, awaiting=check.awaiting - {producer_id}
            )
            if check.checkable and (record := self._tasks.get(task_id)) is not None:
                self._redrive.drive_now(record.workflow_id)

    def retry_awaiting_locked(self, producer_ids: set[str]) -> None:
        """Place again each task waiting on one of these tasks' uploads, as a holder
        can serve it before the upload lands; it waits again if the read still
        misses."""
        retried: dict[str, list[str]] = defaultdict(list)
        for task_id, check in list(self.input_checks.items()):
            if not check.awaiting & producer_ids:
                continue
            record = self._tasks.get(task_id)
            if record is None or record.status != TaskStatus.PENDING:
                continue
            del self.input_checks[task_id]
            record.avoided_worker = None
            if self._ready.enqueue_ready_locked(
                task_id,
                front=False,
                not_before=time.monotonic() + INPUT_RECHECK_PACE_SEC,
            ):
                self._cv.notify_all()
            retried[record.workflow_id].append(task_id)
        for workflow_id in retried:
            # The record is unchanged; only the workflow's schedule moved.
            self._committer.commit_records_locked(
                workflow_id, (), with_membership=False, sched=True
            )

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
