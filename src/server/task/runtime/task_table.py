"""The runtime's task records, indexed by the workflow each belongs to."""

from typing import Any

from ..models import TaskRecord


class TaskTable(dict[str, TaskRecord]):
    """Task records by task id, with each workflow's task ids kept beside them in the
    order they were added, so a question about one workflow reads its own tasks
    rather than every task."""

    def __init__(self) -> None:
        super().__init__()
        self._by_workflow: dict[str, dict[str, None]] = {}

    def __setitem__(self, task_id: str, record: TaskRecord) -> None:
        if (previous := self.get(task_id)) is not None:
            self._forget(task_id, previous.workflow_id)
        super().__setitem__(task_id, record)
        self._by_workflow.setdefault(record.workflow_id, {})[task_id] = None

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

    def of_workflow(self, workflow_id: str) -> list[TaskRecord]:
        """The records of one workflow's tasks."""
        return [self[task_id] for task_id in self._by_workflow.get(workflow_id, ())]

    def ids_of(self, workflow_id: str) -> list[str]:
        """The ids of one workflow's tasks."""
        return list(self._by_workflow.get(workflow_id, ()))

    def _forget(self, task_id: str, workflow_id: str) -> None:
        if (ids := self._by_workflow.get(workflow_id)) is not None:
            ids.pop(task_id, None)
            if not ids:
                del self._by_workflow[workflow_id]


__all__ = ["TaskTable"]
