"""A task whose dispatch ends or returns releases what that dispatch exposed."""

import logging
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.dispatcher.base import Dispatcher
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _register,
    _runtime,
    _worker,
)

SERVE_V1 = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: serve}
spec:
  graph:
    nodes:
      - name: serve
        spec:
          taskType: serve
          resources: {hardware: {gpu: {type: any, count: 1}}}
          model: {source: {type: huggingface, identifier: org/served}}
"""


class _Harness:
    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self.serve = MagicMock()
        self.forward = MagicMock()
        self.metrics = MagicMock()
        self.redis = MagicMock()
        watchdog = MagicMock()
        watchdog.enabled = False
        self.monitor = EventMonitor(
            redis_client=self.redis,
            logger=logging.getLogger("release-test"),
            runtime=runtime,
            dispatcher=Dispatcher(runtime, MagicMock(), logging.getLogger("release")),
            worker_registry=MagicMock(),
            node_registry=MagicMock(),
            metrics_recorder=self.metrics,
            watchdog=watchdog,
            gated_serve=self.serve,
            port_forward=self.forward,
        )

    def deliver(self, *events: TaskEvent | WorkerEvent) -> None:
        for event in events:
            if isinstance(event, TaskEvent):
                self.monitor.handle_task_event(event)
            else:
                self.monitor._handle_worker_event(event)

    def released(self, task_id: str) -> bool:
        self.forward.unregister_task.assert_any_call(task_id)
        return True


async def _dispatched(
    harness: _Harness, workflow: str, node: str
) -> tuple[str, dict[str, str]]:
    workflow_id, ids = await _register(harness.runtime, workflow)
    assert harness.runtime.next_ready(threading.Event(), timeout=0.01) == ids[node]
    record_dispatch(harness.runtime, ids[node], cast(Any, _worker()), "dsp-1")
    return workflow_id, ids


def _given_up(task_id: str) -> TaskEvent:
    return TaskEvent(
        type="TASK_CANCELLED",
        task_id=task_id,
        worker_id="wkr-1",
        dispatch_id="dsp-1",
        ts=_TS,
    )


_LEFT = WorkerEvent(type="UNREGISTER", worker_id="wkr-1", graceful=True)


def _v1_serve_return(harness: _Harness, task_id: str, path: str) -> None:
    match path:
        case "given_up":
            harness.deliver(_given_up(task_id), _LEFT)
        case "left_first":
            harness.deliver(_LEFT, _given_up(task_id))
        case "lost":
            harness.deliver(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))
        case "failed":
            harness.deliver(
                TaskEvent(
                    type="TASK_FAILED",
                    task_id=task_id,
                    worker_id="wkr-1",
                    dispatch_id="dsp-1",
                    error="engine crashed",
                    retryable=True,
                    ts=_TS,
                )
            )


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["given_up", "left_first", "lost", "failed"])
async def test_a_returned_serve_task_drains_its_binding(path: str) -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    _, ids = await _dispatched(harness, SERVE_V1, "serve")
    task_id = ids["serve"]

    _v1_serve_return(harness, task_id, path)

    record = harness.runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.PENDING
    harness.serve.drain.assert_called_with(task_id)
    assert harness.released(task_id)
