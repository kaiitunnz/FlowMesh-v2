"""Resident serve tasks and the dispatch each runs under."""

from collections.abc import Callable

from ..models import SETTLING_TASK_STATUSES, TaskRecord, TaskStatus


class ResidentServeTasks:
    """Records the dispatch each resident serve task runs under and tells resident
    capacity when one ends, reports an update, or should yield its worker."""

    def __init__(self, tasks: dict[str, TaskRecord]) -> None:
        self._tasks = tasks
        self.resident_task_ended: Callable[[str], None] | None = None
        self.resident_task_updated: Callable[[str], None] | None = None
        self.resident_yield_requested: Callable[[str], None] | None = None
        # The dispatch each resident task was last committed DISPATCHED under, so a
        # commit moving it elsewhere, or under another dispatch, reports that the
        # earlier dispatch ended.
        self.dispatched_resident: dict[str, str | None] = {}

    def seed_dispatched_resident_locked(self) -> None:
        """Record the dispatch each restored resident serve task runs under."""
        self.dispatched_resident.update(
            (task_id, record.dispatch_id)
            for task_id, record in self._tasks.items()
            if record.resident and record.status == TaskStatus.DISPATCHED
        )

    def request_yield(self, task_id: str) -> bool:
        if self.resident_yield_requested is None:
            return False
        self.resident_yield_requested(task_id)
        return True

    def dispatches_on_locked(self, worker_id: str) -> list[tuple[str, str]]:
        return [
            (task_id, dispatch_id)
            for task_id, dispatch_id in self.dispatched_resident.items()
            if dispatch_id is not None
            and (record := self._tasks.get(task_id)) is not None
            and record.assigned_worker == worker_id
        ]

    def observe_resident_locked(self, record: TaskRecord) -> None:
        task_id = record.task_id
        ended = task_id in self.dispatched_resident
        if record.status == TaskStatus.DISPATCHED:
            ended = ended and self.dispatched_resident[task_id] != record.dispatch_id
            self.dispatched_resident[task_id] = record.dispatch_id
        else:
            self.dispatched_resident.pop(task_id, None)
            ended = ended or record.status in SETTLING_TASK_STATUSES
        if ended and self.resident_task_ended is not None:
            self.resident_task_ended(task_id)
        elif (
            record.status == TaskStatus.DISPATCHED
            and record.latest_update_dispatch_id == record.dispatch_id
            and self.resident_task_updated is not None
        ):
            self.resident_task_updated(task_id)

    def set_resident_task_end_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer told when a resident serve task stops serving."""
        self.resident_task_ended = hook

    def set_resident_task_update_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer told when a dispatched resident serve task reports an
        update under its current dispatch, such as its engine endpoint."""
        self.resident_task_updated = hook

    def set_resident_yield_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer asked to free a worker a resident serve task
        occupies."""
        self.resident_yield_requested = hook
