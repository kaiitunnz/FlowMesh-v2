"""The ready queue and the epoch frontier that gates it."""

import heapq
import time
from collections import defaultdict, deque

from ..models import TaskRecord, TaskStatus
from .static_dag import StaticDag


class EpochFrontier:
    """Holds each epoch-ordered workflow's remaining epochs and current frontier, and
    the epoch each of its tasks belongs to."""

    def __init__(self) -> None:
        self.workflow_epoch_tasks: dict[str, deque[set[str]]] = {}
        self.workflow_epoch_frontier: dict[str, int] = {}
        self.workflow_in_epoch_order: dict[str, bool] = {}
        self.task_epoch_index: dict[str, int] = {}

    def is_epoch_ready_locked(self, record: TaskRecord) -> bool:
        epoch_index = self.task_epoch_index.get(record.task_id)
        if epoch_index is None:
            return True
        frontier = self.workflow_epoch_frontier.get(record.workflow_id)
        if frontier is None:
            return True
        return epoch_index == frontier

    def set_frontier(self, workflow_id: str, frontier: int) -> None:
        self.workflow_epoch_frontier[workflow_id] = frontier

    def complete_epoch(self, workflow_id: str) -> None:
        self.workflow_epoch_tasks[workflow_id].popleft()

    def drop_frontier(self, workflow_id: str) -> None:
        self.workflow_epoch_frontier.pop(workflow_id, None)

    def drop_epochs(self, workflow_id: str) -> None:
        self.workflow_epoch_tasks.pop(workflow_id, None)

    def forget_workflow(self, workflow_id: str) -> None:
        self.workflow_epoch_tasks.pop(workflow_id, None)
        self.workflow_epoch_frontier.pop(workflow_id, None)
        self.workflow_in_epoch_order.pop(workflow_id, None)

    def forget_task(self, task_id: str) -> None:
        self.task_epoch_index.pop(task_id, None)


class ReadyQueue:
    """Queues the pending tasks ready to dispatch, in position order within an
    epoch-ordered workflow, buckets them by merge key, and advances a workflow's
    epoch frontier once its current epoch is done."""

    def __init__(
        self,
        epochs: EpochFrontier,
        dag: StaticDag,
        tasks: dict[str, TaskRecord],
    ) -> None:
        self._epochs = epochs
        self._dag = dag
        self._tasks = tasks
        self.ready_by_workflow: dict[str, list[tuple[int, str]]] = {}
        self.ready_queue: deque[tuple[str, bool]] = (
            deque()
        )  # task_id | workflow_id, is_workflow
        self.ready_index: set[str] = set()
        self.merge_key_by_task: dict[str, tuple[str | None, str | None]] = {}
        self.merge_buckets: dict[tuple[str, str | None], list[str]] = defaultdict(list)

    def enqueue_ready_locked(self, task_id: str, *, front: bool = False) -> bool:
        """Add a task to the ready queue if it is pending and not already queued."""
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return False
        if task_id in self.ready_index:
            return False
        if not self._epochs.is_epoch_ready_locked(record):
            return False
        workflow_id = record.workflow_id
        if (
            workflow_id in self._epochs.workflow_in_epoch_order
            and task_id in self._epochs.task_epoch_index
        ):
            queue = self.ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            heapq.heappush(queue, (position_in_epoch, task_id))
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        if front:
            self.ready_queue.appendleft(ready_entry)
        else:
            self.ready_queue.append(ready_entry)
        self.ready_index.add(task_id)
        record.last_queue_ts = time.time()
        self._merge_bucket_add(task_id)
        return True

    def pop_ready_locked(self) -> str | None:
        while self.ready_queue:
            task_or_workflow_id, is_workflow = self.ready_queue.popleft()
            if is_workflow:
                _, task_id = heapq.heappop(self.ready_by_workflow[task_or_workflow_id])
            else:
                task_id = task_or_workflow_id
            self.ready_index.discard(task_id)
            record = self._tasks.get(task_id)
            if not record or record.status != TaskStatus.PENDING:
                continue
            return task_id
        return None

    def remove_from_ready_locked(self, task_id: str) -> None:
        if task_id not in self.ready_index:
            return
        record = self._tasks.get(task_id)
        if not record:
            return
        workflow_id = record.workflow_id
        if (
            workflow_id in self._epochs.workflow_in_epoch_order
            and task_id in self._epochs.task_epoch_index
        ):
            queue = self.ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            queue.remove((position_in_epoch, task_id))
            heapq.heapify(queue)
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        self.ready_queue.remove(ready_entry)
        self.ready_index.discard(task_id)

    def _merge_bucket_add(self, task_id: str) -> None:
        key = self.merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self.merge_buckets.setdefault((merge_key, selected_worker), [])
        if task_id not in bucket:
            bucket.append(task_id)

    def merge_bucket_remove(self, task_id: str) -> None:
        key = self.merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self.merge_buckets.get((merge_key, selected_worker))
        if not bucket:
            return
        try:
            bucket.remove(task_id)
        except ValueError:
            pass
        if not bucket:
            self.merge_buckets.pop((merge_key, selected_worker), None)

    def try_advance_epoch_frontier_locked(self, workflow_id: str) -> list[str]:
        epoch_tasks = self._epochs.workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return []
        frontier = self._epochs.workflow_epoch_frontier[workflow_id]

        ready: list[str] = []
        while True:
            self._epochs.set_frontier(workflow_id, frontier)
            current_tasks = epoch_tasks[0] if epoch_tasks else set()
            if current_tasks and not all(
                (task := self._tasks.get(task_id)) is not None
                and task.status == TaskStatus.DONE
                for task_id in current_tasks
            ):
                break

            if epoch_tasks:
                self._epochs.complete_epoch(workflow_id)
            frontier += 1
            self._epochs.set_frontier(workflow_id, frontier)
            if not epoch_tasks:
                self._epochs.drop_frontier(workflow_id)
                self._epochs.drop_epochs(workflow_id)
                break

            for task_id in epoch_tasks[0]:
                record = self._tasks.get(task_id)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                if self._dag.pending_deps.get(task_id):
                    continue
                if self.enqueue_ready_locked(task_id):
                    ready.append(task_id)

        return ready

    def set_merge_key(self, task_id: str, key: tuple[str | None, str | None]) -> None:
        self.merge_key_by_task[task_id] = key

    def forget_merge_key(self, task_id: str) -> None:
        self.merge_key_by_task.pop(task_id, None)
