"""Task merging: coalescing ready tasks into one dispatch and returning them."""

import logging
import threading
from collections import defaultdict

from shared.content import ContentReference

from ..models import TERMINAL_TASK_STATUSES, TaskRecord, TaskStatus, TaskUsage
from . import content_bindings
from .commits import TransitionCommitter
from .reports import reset_to_pending
from .scheduling import ReadyQueue
from .static_dag import StaticDag


class TaskMerges:
    """Coalesces ready tasks that share a merge key into one dispatch, settles each
    merged child from its parent's result, and returns children to the queue when
    their dispatch ends without one."""

    def __init__(
        self,
        ready: ReadyQueue,
        dag: StaticDag,
        committer: TransitionCommitter,
        tasks: dict[str, TaskRecord],
        completed: set[str],
        failed: set[str],
        logger: logging.Logger,
        cv: threading.Condition,
    ) -> None:
        self._ready = ready
        self._dag = dag
        self._committer = committer
        self._tasks = tasks
        self._completed = completed
        self._failed = failed
        self._logger = logger
        self._cv = cv
        self.merge_children_map: dict[str, list[str]] = defaultdict(list)
        self.merge_parent_map: dict[str, str] = {}

    def restore_merges_locked(self) -> None:
        """Rebuild the in-flight merges from durable records.

        A merge may span workflows, so it is rebuilt once every workflow is restored.
        A merge whose parent never dispatched returns its children to the queue, and
        one whose parent settled returns them to run alone.
        """
        for child_id, record in self._tasks.items():
            if record.merged_parent_id and record.status == TaskStatus.DISPATCHED:
                self.merge_parent_map[child_id] = record.merged_parent_id
                self.merge_children_map[record.merged_parent_id].append(child_id)
        parents = {
            *self.merge_children_map,
            *(
                task_id
                for task_id, rec in self._tasks.items()
                if rec.merged_children and rec.status not in TERMINAL_TASK_STATUSES
            ),
        }
        for parent_id in parents:
            parent = self._tasks.get(parent_id)
            if parent is None or parent.status in TERMINAL_TASK_STATUSES:
                children = self.merge_children_map.pop(parent_id, [])
                self._committer.commit_locked(
                    *self.return_merged_children_locked(children, unmerge=True)
                )
            elif parent.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self.release_merge_locked(parent_id)

    def plan_merge_locked(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return []
        if record.merge_key is None:
            return []
        if self.merge_children_map.get(task_id):
            return []
        if record.selected_worker and assigned_worker not in record.selected_worker:
            raise ValueError(
                f"The worker assigned for task {task_id} ({assigned_worker}) "
                f"is not in selected workers {record.selected_worker}."
            )
        bucket = (
            self._ready.merge_buckets[(record.merge_key, assigned_worker)]
            + self._ready.merge_buckets[(record.merge_key, None)]
        )
        if not bucket or len(bucket) <= 1:
            return []
        siblings: list[str] = []
        for candidate in bucket:
            if candidate == task_id:
                continue
            if len(siblings) >= max_batch_size - 1:
                break
            candidate_record = self._tasks.get(candidate)
            if not candidate_record or candidate_record.status != TaskStatus.PENDING:
                continue
            if (
                candidate_record.selected_worker
                and assigned_worker not in candidate_record.selected_worker
            ):
                continue
            if candidate not in self._ready.ready_index:
                continue
            siblings.append(candidate)
        if not siblings:
            return []

        record.merged_children = siblings
        self.merge_children_map[task_id] = siblings.copy()
        for sibling in siblings:
            self.merge_parent_map[sibling] = task_id
            self._ready.remove_from_ready_locked(sibling)
            self._ready.merge_bucket_remove(sibling)
            sibling_record = self._tasks.get(sibling)
            if sibling_record:
                sibling_record.status = TaskStatus.DISPATCHED
                sibling_record.merged_parent_id = task_id
                sibling_record.assigned_worker = None
                sibling_record.merge_slice = None
        self._committer.commit_locked(task_id, *siblings)
        return siblings

    def release_merge_locked(self, task_id: str) -> None:
        if parent := self._tasks.get(task_id):
            parent.merged_children = None
        returned = self.return_merged_children_locked(
            self.merge_children_map.pop(task_id, [])
        )
        self._committer.commit_locked(task_id, *returned)

    def merged_child_record(self, task_id: str, child_id: str) -> TaskRecord | None:
        """A child's record while it is still merged into ``task_id``'s dispatch."""
        record = self._tasks.get(child_id)
        if (
            record is None
            or record.status != TaskStatus.DISPATCHED
            or self.merge_parent_map.get(child_id) != task_id
        ):
            return None
        return record

    def release_merged_child(
        self, task_id: str, child_id: str, merge_key: str | None
    ) -> None:
        """Take one child out of a task's merge and return it to the ready queue, to
        merge next under ``merge_key``, or to run alone when it is None."""
        if parent := self._tasks.get(task_id):
            parent.merged_children = [
                sibling
                for sibling in parent.merged_children or []
                if sibling != child_id
            ] or None
        if (siblings := self.merge_children_map.get(task_id)) and (
            child_id in siblings
        ):
            siblings.remove(child_id)
        returned: list[str] = []
        if self.merge_parent_map.get(child_id) == task_id and (
            child := self._tasks.get(child_id)
        ):
            child.merge_key = merge_key
            _, selected_worker_hint = self._ready.merge_key_by_task.get(
                child_id, (None, None)
            )
            self._ready.merge_key_by_task[child_id] = (
                merge_key,
                selected_worker_hint,
            )
            returned = self.return_merged_children_locked([child_id])
        self._committer.commit_locked(task_id, *returned)

    def return_merged_children_locked(
        self, child_ids: list[str], unmerge: bool = False
    ) -> list[str]:
        """Return merged children to the head of the ready queue, spending no attempt.

        ``unmerge`` also drops a child's merge key, so its next dispatch runs it alone
        rather than merging it into another batch that may again leave it without a
        result of its own. Returns the children it moved, for the caller to commit.
        """
        returned: list[str] = []
        for child_id in child_ids:
            self.merge_parent_map.pop(child_id, None)
            child_record = self._tasks.get(child_id)
            if not child_record or child_record.status in TERMINAL_TASK_STATUSES:
                continue
            reset_to_pending(child_record)
            child_record.merged_parent_id = None
            child_record.merge_slice = None
            if unmerge:
                child_record.merge_key = None
                self._ready.merge_key_by_task.pop(child_id, None)
            self._ready.remove_from_ready_locked(child_id)
            self._ready.enqueue_ready_locked(child_id, front=True)
            returned.append(child_id)
        if returned:
            self._cv.notify_all()
        return returned

    def partition_merged_children_locked(
        self, child_ids: list[str], child_references: dict[str, ContentReference]
    ) -> tuple[list[str], list[str]]:
        """Split merged children into those with a result of their own and the rest.

        A child's result counts as its own only in the scope control gave the child.
        """
        settled: list[str] = []
        unsettled: list[str] = []
        for child_id in child_ids:
            record = self._tasks.get(child_id)
            reference = child_references.get(child_id)
            if record is not None and content_bindings.accepted_reference(
                self._logger, record, reference
            ):
                settled.append(child_id)
            else:
                unsettled.append(child_id)
        return settled, unsettled

    def settle_workflows_locked(self, record: TaskRecord) -> list[str]:
        """The workflows a task's settlement touches: its own and its merged
        children's."""
        return list(
            dict.fromkeys(
                [
                    record.workflow_id,
                    *(
                        child.workflow_id
                        for child_id in record.merged_children or []
                        if (child := self._tasks.get(child_id)) is not None
                    ),
                ]
            )
        )

    def finalize_merged_child_success(
        self,
        child_id: str,
        worker_id: str | None,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
        reference: ContentReference,
    ) -> list[str]:
        ready_children: list[str] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return ready_children
        child_record.status = TaskStatus.DONE
        content_bindings.bind_result_locked(self._logger, child_record, reference, None)
        child_record.error = None
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        if worker_id:
            child_record.assigned_worker = worker_id
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        if usage is not None:
            child_record.usages.append(usage)
        self._completed.add(child_id)
        self._failed.discard(child_id)
        self._dag.pending_deps.pop(child_id, None)
        self.merge_parent_map.pop(child_id, None)
        self._ready.merge_key_by_task.pop(child_id, None)
        self._ready.remove_from_ready_locked(child_id)
        self._ready.merge_bucket_remove(child_id)
        dependents = list(self._dag.dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._dag.pending_deps.get(dep_id)
            if pending is None:
                continue
            pending.discard(child_id)
            if not pending:
                dep_record = self._tasks.get(dep_id)
                if dep_record and dep_record.status == TaskStatus.PENDING:
                    if self._ready.enqueue_ready_locked(dep_id):
                        ready_children.append(dep_id)
        return ready_children
