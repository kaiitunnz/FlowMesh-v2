"""The dispatcher reserves a worker for a dispatch before publishing it, and releases
the reservation only for a dispatch no report of the worker's holds."""

import asyncio
import logging
import threading
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.dispatcher.base import Dispatcher
from server.registries.worker import Worker
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry

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
        credential_vault=InMemoryCredentialVault(),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )
    task_id = results[0].task_id
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_WORKER]
    registry.satisfying_workers.return_value = [_WORKER]
    writes: list[str] = []

    def recording(write: str) -> Any:
        def record(*_: Any) -> bool:
            writes.append(write)
            return True

        return record

    registry.reserve_worker.side_effect = recording("reserved")
    registry.release_worker.side_effect = recording("released")

    def publish(*_: Any) -> int:
        writes.append("published")
        return 1

    registry.publish_task.side_effect = publish
    return runtime, task_id, registry, writes


def _dispatch(runtime: TaskRuntime, task_id: str, registry: mock.Mock) -> bool:
    return CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("busy-status-test"),
    ).dispatch_once(task_id)


def test_the_worker_is_reserved_for_the_dispatch_before_it_is_published() -> None:
    runtime, task_id, registry, writes = _setup()

    _dispatch(runtime, task_id, registry)

    assert writes == ["reserved", "published"]
    record = runtime.get_record(task_id)
    assert record is not None
    registry.reserve_worker.assert_called_once_with(
        _WORKER.id, task_id, record.dispatch_id
    )


def test_an_abandoned_publish_releases_its_reservation() -> None:
    for outcome in (0, RuntimeError("redis down")):
        runtime, task_id, registry, writes = _setup()
        if isinstance(outcome, Exception):
            registry.publish_task.side_effect = outcome
        else:
            registry.publish_task.side_effect = None
            registry.publish_task.return_value = outcome

        _dispatch(runtime, task_id, registry)

        assert writes == ["reserved", "released"]
        dispatch_id = registry.reserve_worker.call_args.args[2]
        registry.release_worker.assert_called_once_with(_WORKER.id, dispatch_id)


def test_a_publish_failing_after_its_worker_reported_keeps_the_worker_reserved() -> (
    None
):
    runtime, task_id, registry, writes = _setup()

    def delivered_then_raised(_worker: Any, message: Any) -> int:
        # The worker took the task and reported on it before the publish call failed.
        runtime.mark_started(
            task_id, _WORKER.id, {}, "2026-06-01T00:00:00Z", message.dispatch_id
        )
        raise RuntimeError("reply lost")

    registry.publish_task.side_effect = delivered_then_raised

    _dispatch(runtime, task_id, registry)

    assert writes == ["reserved"]
    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.DISPATCHED


def test_a_worker_gone_before_its_reservation_is_handed_nothing() -> None:
    runtime, task_id, registry, writes = _setup()
    registry.reserve_worker.side_effect = None
    registry.reserve_worker.return_value = False

    dispatcher = Dispatcher(runtime, registry, logging.getLogger("busy-status-test"))
    assert dispatcher.dispatch_once(task_id) is False

    registry.publish_task.assert_not_called()
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.PENDING and record.attempts == 0
    assert runtime.ready_queue_length() == 1
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
