"""Failing task records in memory."""

import time

from ...orchestration import dependency_failed
from ..models import TaskRecord, TaskStatus
from .scheduling import ReadyQueue
from .static_dag import StaticDag


class RecordFailures:
    """Fails a task record in memory, and every pending task downstream of a failed
    v1 task."""

    def __init__(
        self,
        dag: StaticDag,
        ready: ReadyQueue,
        tasks: dict[str, TaskRecord],
        failed: set[str],
    ) -> None:
        self._dag = dag
        self._ready = ready
        self._tasks = tasks
        self._failed = failed

    def fail_record_locked(self, record: TaskRecord, reason: str) -> None:
        task_id = record.task_id
        record.status = TaskStatus.FAILED
        record.error = reason
        record.assigned_worker = None
        record.finished_ts = time.time()
        self._failed.add(task_id)
        self._dag.pending_deps.pop(task_id, None)
        self._ready.remove_from_ready_locked(task_id)

    def fail_v1_dependents_locked(self, primary: str) -> list[tuple[str, str]]:
        """Fail every pending task downstream of a failed v1 task, however deep."""
        reason = dependency_failed(primary)
        impacted: list[tuple[str, str]] = []
        frontier = [primary]
        while frontier:
            failed = frontier.pop()
            for child in self._dag.dependents.pop(failed, set()):
                if (pending := self._dag.pending_deps.get(child)) is not None:
                    pending.discard(failed)
                record = self._tasks.get(child)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                self.fail_record_locked(record, reason)
                impacted.append((child, reason))
                frontier.append(child)
        return impacted
