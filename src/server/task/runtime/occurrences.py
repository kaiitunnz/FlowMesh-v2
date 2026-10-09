"""Task records for the work that region definitions and spawns materialize."""

import time
from collections.abc import Iterable

from ...orchestration import OrchestrationEngine
from ...orchestration.engine.advance import Advance
from ..models import TaskRecord, TaskStatus
from .commits import TransitionCommitter


class OccurrenceMaterializer:
    """Gives each materialized work item a task record made from its operator's
    blueprint.

    A blueprint is the task an operator was submitted as; it is never dispatched and
    never one of the workflow's tasks. A record made from it belongs to one work item.
    """

    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        original_deps: dict[str, set[str]],
        committer: TransitionCommitter,
    ) -> None:
        self._tasks = tasks
        self._original_deps = original_deps
        self._committer = committer
        self._blueprints: dict[str, dict[str, TaskRecord]] = {}

    def install_locked(
        self, workflow_id: str, blueprints: Iterable[TaskRecord]
    ) -> None:
        self._blueprints.setdefault(workflow_id, {}).update(
            (record.task_id, record) for record in blueprints
        )

    def blueprint_locked(self, workflow_id: str, operator_id: str) -> TaskRecord | None:
        return self._blueprints.get(workflow_id, {}).get(operator_id)

    def register_locked(self, workflow_id: str, task_id: str, operator_id: str) -> bool:
        """Install the record one work item runs as; False when its operator has no
        blueprint."""
        blueprint = self.blueprint_locked(workflow_id, operator_id)
        if blueprint is None:
            return False
        self._tasks[task_id] = blueprint.model_copy(
            deep=True,
            update={
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "assigned_worker": None,
                "started_ts": None,
                "finished_ts": None,
                "error": None,
                "usages": [],
                "position_in_epoch": None,
                "graph_node_name": None,
                "local_name": None,
                "merge_key": None,
                "merged_children": None,
                "selected_worker": None,
                "submitted_ts": time.time(),
            },
        )
        self._original_deps[task_id] = set()
        self._committer.note_child_locked(workflow_id, task_id)
        return True

    def materialize_locked(
        self, workflow_id: str, engine: OrchestrationEngine, advance: Advance
    ) -> list[str]:
        """Give each work item an advance readies without a record its record, owed
        with the ledger snapshot that holds its work; returns the operators that have
        no blueprint to make one from."""
        missing: list[str] = []
        for task_id in advance.ready:
            if (
                task_id not in self._tasks
                and (wi := engine.work_item(task_id)) is not None
                and not self.register_locked(workflow_id, task_id, wi.operator_id)
            ):
                missing.append(wi.operator_id)
        return missing
