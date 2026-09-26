"""Record a task's dispatch on a runtime the way the dispatcher does."""

from types import SimpleNamespace
from typing import cast

from server.registries.worker import Worker
from server.task.runtime import TaskRuntime


def record_dispatch(
    runtime: TaskRuntime,
    task_id: str,
    worker: str | Worker = "wkr-1",
    dispatch_id: str | None = None,
    input_preparation: bool = False,
) -> bool:
    if isinstance(worker, str):
        worker = cast(Worker, SimpleNamespace(id=worker, node_id="nde-1"))
    runtime.begin_publish(
        task_id, worker, dispatch_id, input_preparation=input_preparation
    )
    return runtime.mark_dispatched(task_id)
