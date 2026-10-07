"""The dispatch fence: which dispatch holds a task, and whose events apply."""

import time
from dataclasses import dataclass

from ...hooks import SUPPLIER_RESOLVERS
from ...orchestration import OrchestrationEngine
from ...registries.worker import Worker
from ..models import (
    SETTLING_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    TaskRecord,
    TaskStatus,
)
from . import episode_dispatch
from .commits import HeldWrites, TransitionCommitter
from .merges import TaskMerges
from .reservations import WorkerReservations
from .scheduling import ReadyQueue


@dataclass
class Publish:
    """A dispatch published to a worker: what recording it takes, and how far its
    worker's reports have taken it."""

    worker_id: str
    dispatch_id: str | None
    supplier_id: str
    input_preparation: bool
    recorded: bool = False
    reported: bool = False


def supplier_id(worker: Worker) -> str:
    for resolver in SUPPLIER_RESOLVERS:
        if (resolved := resolver.resolve(worker)) is not None:
            return resolved
    return ""


class DispatchFence:
    """Marks each dispatch while it is published, records it once published, decides
    whether a worker's event belongs to the dispatch holding its task, and remembers
    the last dispatch that returned each task."""

    def __init__(
        self,
        merges: TaskMerges,
        committer: TransitionCommitter,
        ready: ReadyQueue,
        reservations: WorkerReservations,
        tasks: dict[str, TaskRecord],
        engines: dict[str, OrchestrationEngine],
    ) -> None:
        self._merges = merges
        self._committer = committer
        self._ready = ready
        self._reservations = reservations
        self._tasks = tasks
        self._engines = engines
        # The dispatch being published for each task, until the dispatcher records it;
        # None when its worker was lost first.
        self.publishing: dict[str, Publish | None] = {}
        # The last dispatch of each task that ended by returning it to the queue: its
        # worker and dispatch id, and the tasks the return moved.
        self.returned_dispatches: dict[str, tuple[str, str | None, list[str]]] = {}

    def return_failed_merge_locked(self, record: TaskRecord, worker_id: str) -> bool:
        """Return a merged dispatch that failed or lost ``worker_id``, if it is one.

        Such a failure belongs to no single task in the batch, so the parent and every
        child still merged into it go back to the head of the queue to run alone,
        spending no attempt. Returns whether the report was for a merged dispatch to
        ``worker_id``.
        """
        if (
            record.merged_dispatch_worker != worker_id
            or record.status != TaskStatus.DISPATCHED
        ):
            return False
        task_id = record.task_id
        dispatch_id = record.dispatch_id
        moved = self._merges.return_merged_children_locked(
            [task_id, *self._merges.merge_children_map.pop(task_id, [])], unmerge=True
        )
        self.returned_dispatches[task_id] = (worker_id, dispatch_id, moved)
        record.merged_children = None
        self._committer.commit_locked(*moved)
        return True

    def holds_dispatch_locked(
        self, record: TaskRecord, worker_id: str | None, dispatch_id: str | None
    ) -> bool:
        """Whether an event from ``worker_id`` belongs to the dispatch holding a task.

        A task is held by its recorded dispatch, or by one published and not yet
        recorded. An event naming no dispatch matches on its worker. A root-internal
        transition names no worker and is not fenced.
        """
        if worker_id is None:
            return True
        held = [(record.assigned_worker, record.dispatch_id)]
        if (publish := self.publishing.get(record.task_id)) and not publish.recorded:
            held.append((publish.worker_id, publish.dispatch_id))
        return any(
            worker_id == held_worker
            and (dispatch_id is None or dispatch_id == held_dispatch)
            for held_worker, held_dispatch in held
        )

    def dispatch_live_locked(
        self, record: TaskRecord, worker_id: str, dispatch_id: str | None
    ) -> bool:
        """Whether the named dispatch, on ``worker_id``, holds a task still running."""
        return (
            dispatch_id is not None
            and record.status not in TERMINAL_TASK_STATUSES
            and self.holds_dispatch_locked(record, worker_id, dispatch_id)
        )

    def accepts_event_locked(
        self, record: TaskRecord, worker_id: str | None, dispatch_id: str | None
    ) -> bool:
        """Whether an event belongs to the dispatch holding a task.

        The first event of a dispatch published and not yet recorded records it, so
        the event applies to a recorded dispatch.
        """
        if not self.holds_dispatch_locked(record, worker_id, dispatch_id):
            return False
        if worker_id is not None and (publish := self.publishing.get(record.task_id)):
            publish.reported = True
            if not publish.recorded and record.status == TaskStatus.PENDING:
                self._record_dispatch_locked(record, publish)
        return True

    def begin_publish(self, task_id: str, publish: Publish) -> bool:
        """Mark a dispatch as being published, so its worker's earliest events apply."""
        record = self._tasks.get(task_id)
        if record is None or record.status != TaskStatus.PENDING:
            return False
        self.publishing[task_id] = publish
        return True

    def mark_dispatched(self, task_id: str) -> bool:
        """Record the publish `begin_publish` marked; returns whether it holds the
        task."""
        publish = self.publishing.pop(task_id, None)
        if publish is None:
            return False
        record = self._tasks.get(task_id)
        if publish.recorded:
            return (
                record is not None
                and record.dispatch_id == publish.dispatch_id
                and record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
            )
        if not record or record.status in SETTLING_TASK_STATUSES:
            # A replayed or late dispatch must not regress a settling task.
            return False
        self._record_dispatch_locked(record, publish)
        return True

    def _record_dispatch_locked(self, record: TaskRecord, publish: Publish) -> None:
        self.take_dispatch_locked(record, publish)
        self._committer.commit_transition_locked(
            record.workflow_id,
            records=self._committer.records_locked(record.task_id),
            dispatched=[record.task_id],
        )
        self._committer.save_ledger_locked(record.workflow_id)

    def take_dispatch_locked(self, record: TaskRecord, publish: Publish) -> None:
        """Record a published dispatch as the one holding its task, in memory."""
        task_id = record.task_id
        publish.recorded = True
        self.returned_dispatches.pop(task_id, None)
        if (
            stash := self._committer.unacknowledged.get(task_id)
        ) and stash.dispatch_id is None:
            # A report naming no dispatch matches on its worker alone, which the next
            # dispatch may share.
            del self._committer.unacknowledged[task_id]
        record.status = TaskStatus.DISPATCHED
        record.assigned_worker = publish.worker_id
        record.dispatch_id = publish.dispatch_id
        if publish.dispatch_id is not None:
            held = (publish.worker_id, publish.dispatch_id)
            self._reservations.hold_dispatch_locked(task_id, held)
        record.merged_dispatch_worker = (
            publish.worker_id if self._merges.merge_children_map.get(task_id) else None
        )
        record.topic = "tasks"
        record.dispatched_ts = time.time()
        record.next_retry_at = None
        record.supplier_id = publish.supplier_id
        self._ready.remove_from_ready_locked(task_id)
        self._ready.merge_bucket_remove(task_id)
        if engine := self._engines.get(record.workflow_id):
            if publish.input_preparation:
                engine.on_input_preparation_dispatched(task_id, publish.worker_id)
            else:
                engine.on_dispatched(task_id, publish.worker_id)

    def heal_returned_locked(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> None:
        """Commit again what returning a dispatch moved, on a report of that dispatch.

        Such a report may be handled again because the return's commit failed.
        """
        if (returned := self.returned_dispatches.get(task_id)) is None:
            return
        held_worker, held_dispatch, moved = returned
        if worker_id == held_worker and dispatch_id in (None, held_dispatch):
            self._committer.recommit_locked(HeldWrites(moved.copy()))

    def dispatch_in_flight(
        self, task_id: str, dispatch_id: str, worker_id: str
    ) -> bool:
        """Whether a dispatch to a worker is being published or holds its task, and
        has not ended at a suspension."""
        record = self._tasks.get(task_id)
        if record is None or episode_dispatch.dispatch_ended_at_suspension_locked(
            self._engines, record
        ):
            return False
        publish = self.publishing.get(task_id)
        in_flight = record.status in (
            TaskStatus.DISPATCHED,
            TaskStatus.CANCELLING,
        ) or (publish is not None and not publish.recorded)
        return in_flight and self.dispatch_live_locked(record, worker_id, dispatch_id)
