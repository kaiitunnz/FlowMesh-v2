"""The dispatcher's BUSY write for a dispatched worker, ordered against the IDLE its
task's end writes."""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.tasks.worker_message import WorkerStatus
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault

_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: busy-status
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""

_WORKER = Worker(
    id="wkr-1",
    namespace="ns",
    cluster="c",
    node_id="nde-1",
    node_alias="node",
    incarnation=1,
)


def _setup() -> tuple[TaskRuntime, str, mock.Mock, list[str]]:
    runtime = TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("busy-status-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_WORKER]
    registry.satisfying_workers.return_value = [_WORKER]
    registry.publish_task.return_value = 1
    writes: list[str] = []
    registry.update_worker_status.side_effect = lambda _wid, status: writes.append(
        status.value
    )
    return runtime, results[0].task_id, registry, writes


def _dispatch(runtime: TaskRuntime, task_id: str, registry: mock.Mock) -> None:
    CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("busy-status-test"),
    ).dispatch_once(task_id)


def test_a_task_that_ends_before_its_dispatch_is_recorded_leaves_the_worker_idle() -> (
    None
):
    runtime, task_id, registry, writes = _setup()
    mark_dispatched = runtime.mark_dispatched

    def finish_between(task: str) -> bool:
        held = mark_dispatched(task)
        # The task's end is handled before the dispatch thread writes on: its success
        # marks the worker IDLE.
        registry.update_worker_status(_WORKER.id, WorkerStatus.IDLE)
        return held

    cast(Any, runtime).mark_dispatched = finish_between

    _dispatch(runtime, task_id, registry)

    assert writes[-1] == WorkerStatus.IDLE


def test_busy_is_written_before_the_task_is_published() -> None:
    runtime, task_id, registry, writes = _setup()

    def publish(*_: Any) -> int:
        writes.append("published")
        return 1

    registry.publish_task.side_effect = publish

    _dispatch(runtime, task_id, registry)

    assert writes == [WorkerStatus.BUSY, "published"]


def test_an_abandoned_publish_leaves_the_worker_idle() -> None:
    for outcome in (0, RuntimeError("redis down")):
        runtime, task_id, registry, writes = _setup()
        registry.publish_task.side_effect = (
            outcome if isinstance(outcome, Exception) else None
        )
        registry.publish_task.return_value = outcome

        _dispatch(runtime, task_id, registry)

        assert writes == [WorkerStatus.BUSY, WorkerStatus.IDLE]
