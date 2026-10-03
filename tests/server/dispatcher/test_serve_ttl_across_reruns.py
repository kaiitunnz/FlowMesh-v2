"""A serve task's TTL counts from its first start: a re-run's dispatch carries the
time already served."""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry


def _workflow(task_type: str) -> str:
    return f"""
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: serve-ttl
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: {task_type}
"""


def _runtime() -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("serve-ttl-test"),
        credential_vault=InMemoryCredentialVault(),
    )


def _register(runtime: TaskRuntime, task_type: str) -> str:
    _workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _workflow(task_type), format="native")
    )
    return results[0].task_id


def _dispatch(runtime: TaskRuntime, task_id: str) -> WorkerTaskMessage:
    worker = Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("serve-ttl-dispatch"),
    ).dispatch_once(task_id)
    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    return message


def _start_then_requeue(
    runtime: TaskRuntime, task_id: str, message: WorkerTaskMessage, served_sec: float
) -> None:
    runtime.mark_started(
        task_id, "wkr-1", {}, "2026-06-01T00:00:00Z", dispatch_id=message.dispatch_id
    )
    record = runtime.get_record(task_id)
    assert record is not None and record.first_started_ts is not None
    record.first_started_ts -= served_sec
    runtime.return_dispatch(task_id, "wkr-1", increment_retry=False, front=True)
    assert record.status == TaskStatus.PENDING
    assert record.started_ts is None


def test_a_re_run_serve_task_carries_the_time_since_its_first_start() -> None:
    runtime = _runtime()
    task_id = _register(runtime, "dev_model")

    first = _dispatch(runtime, task_id)
    assert first.serve_elapsed_sec is None

    _start_then_requeue(runtime, task_id, first, served_sec=100.0)
    rerun = _dispatch(runtime, task_id)

    assert rerun.serve_elapsed_sec is not None
    assert 100.0 <= rerun.serve_elapsed_sec < 110.0


def test_a_second_start_keeps_the_first() -> None:
    runtime = _runtime()
    task_id = _register(runtime, "dev_model")
    first = _dispatch(runtime, task_id)
    _start_then_requeue(runtime, task_id, first, served_sec=100.0)
    record = runtime.get_record(task_id)
    assert record is not None
    first_started = record.first_started_ts

    rerun = _dispatch(runtime, task_id)
    runtime.mark_started(
        task_id, "wkr-1", {}, "2026-06-01T00:01:00Z", dispatch_id=rerun.dispatch_id
    )

    assert record.first_started_ts == first_started


def test_a_non_serve_task_carries_no_served_time() -> None:
    runtime = _runtime()
    task_id = _register(runtime, "echo")
    first = _dispatch(runtime, task_id)
    _start_then_requeue(runtime, task_id, first, served_sec=100.0)

    assert _dispatch(runtime, task_id).serve_elapsed_sec is None
