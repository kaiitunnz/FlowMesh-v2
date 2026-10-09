"""The runtime's task records, indexed by the workflow each belongs to."""

from typing import Any

from ..models import TERMINAL_TASK_STATUSES, TaskRecord


class TaskTable(dict[str, TaskRecord]):
    """Task records by task id, with each workflow's task ids kept beside them in the
    order they were added, so a question about one workflow reads its own tasks
    rather than every task.

    A terminal status is final for a record, so each workflow also keeps the tasks
    not yet seen terminal, dropped as they are found terminal, with the last finish
    among those dropped.
    """

    def __init__(self) -> None:
        super().__init__()
        self._by_workflow: dict[str, dict[str, None]] = {}
        self._open: dict[str, dict[str, None]] = {}
        self._last_finish: dict[str, float] = {}

    def __setitem__(self, task_id: str, record: TaskRecord) -> None:
        if (previous := self.get(task_id)) is not None:
            self._forget(task_id, previous.workflow_id)
        super().__setitem__(task_id, record)
        self._by_workflow.setdefault(record.workflow_id, {})[task_id] = None
        self._open.setdefault(record.workflow_id, {})[task_id] = None

    def __delitem__(self, task_id: str) -> None:
        self._forget(task_id, self[task_id].workflow_id)
        super().__delitem__(task_id)

    def pop(self, task_id: str, *default: Any) -> Any:
        if (record := self.get(task_id)) is not None:
            self._forget(task_id, record.workflow_id)
        return super().pop(task_id, *default)

    def setdefault(self, task_id: str, record: TaskRecord) -> TaskRecord:
        if (current := self.get(task_id)) is not None:
            return current
        self[task_id] = record
        return record

    def update(self, *args: Any, **kwargs: Any) -> None:
        for task_id, record in dict(*args, **kwargs).items():
            self[task_id] = record

    def clear(self) -> None:
        super().clear()
        self._by_workflow.clear()
        self._open.clear()
        self._last_finish.clear()

    def of_workflow(self, workflow_id: str) -> list[TaskRecord]:
        """The records of one workflow's tasks."""
        return [self[task_id] for task_id in self._by_workflow.get(workflow_id, ())]

    def ids_of(self, workflow_id: str) -> list[str]:
        """The ids of one workflow's tasks."""
        return list(self._by_workflow.get(workflow_id, ()))

    def holds(self, workflow_id: str) -> bool:
        """Whether the table holds any task of the workflow."""
        return workflow_id in self._by_workflow

    def first_unsettled(self, workflow_id: str) -> str | None:
        """A task of the workflow not in a terminal status, or None when every one
        is; the tasks found terminal before it are dropped from those still open."""
        open_ids = self._open.get(workflow_id, {})
        for task_id in list(open_ids):
            record = self[task_id]
            if record.status not in TERMINAL_TASK_STATUSES:
                return task_id
            del open_ids[task_id]
            if record.finished_ts is not None:
                self._last_finish[workflow_id] = max(
                    self._last_finish.get(workflow_id, record.finished_ts),
                    record.finished_ts,
                )
        return None

    def last_finish(self, workflow_id: str) -> float | None:
        """The last finish among the workflow's tasks found terminal."""
        return self._last_finish.get(workflow_id)

    def _forget(self, task_id: str, workflow_id: str) -> None:
        if (ids := self._by_workflow.get(workflow_id)) is not None:
            ids.pop(task_id, None)
            if not ids:
                del self._by_workflow[workflow_id]
                self._open.pop(workflow_id, None)
                self._last_finish.pop(workflow_id, None)
        if (open_ids := self._open.get(workflow_id)) is not None:
            open_ids.pop(task_id, None)


__all__ = ["TaskTable"]
