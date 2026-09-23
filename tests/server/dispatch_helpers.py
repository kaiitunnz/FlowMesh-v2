"""Record a task's dispatch on a runtime the way the dispatcher does."""

from types import SimpleNamespace
from typing import cast

from server.registries.worker import Worker
from server.task.runtime import TaskRuntime


def record_dispatch(
    runtime: TaskRuntime,
    task_id: str,
    worker_id: str = "wkr-1",
    dispatch_id: str | None = None,
    input_preparation: bool = False,
) -> None:
    runtime.mark_dispatched(
        task_id,
        cast(Worker, SimpleNamespace(id=worker_id, node_id="nde-1")),
        dispatch_id,
        input_preparation=input_preparation,
    )
