"""A task whose dispatch ends or returns releases what that dispatch exposed, and a
task that fails with its worker's loss closes like any failed task."""

import logging
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.clients.redis import task_log_closed_key
from server.dispatcher.base import Dispatcher
from server.services.monitoring import EventMonitor
from server.services.watchdog import WorkerWatchdog
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_unrequested_cancel import SSH_THEN_ECHO
from tests.server.task.test_v2_orchestration import (
    _TS,
    LINEAR,
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

    def closed_as_failed(self, task_id: str) -> bool:
        self.metrics.finalize_task_failure.assert_any_call(task_id)
        self.redis.set_value.assert_any_call(task_log_closed_key(task_id), "1")
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


@pytest.mark.anyio
@pytest.mark.parametrize("left_first", [False, True])
async def test_a_task_failing_as_given_up_closes_with_its_dependents(
    left_first: bool,
) -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    workflow_id, ids = await _dispatched(harness, SSH_THEN_ECHO, "session")
    session, after = ids["session"], ids["after"]

    harness.deliver(
        *((_LEFT, _given_up(session)) if left_first else (_given_up(session), _LEFT))
    )

    record = harness.runtime.get_record(session)
    assert record is not None and record.status == TaskStatus.FAILED
    assert harness.released(session)
    assert harness.closed_as_failed(session)
    assert harness.closed_as_failed(after)
    failed = [
        call.args[0]
        for call in harness.metrics.record_task_event.call_args_list
        if call.args[0].type == "TASK_FAILED"
    ]
    assert [event.task_id for event in failed] == [session, after]
    assert harness.runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_task_failing_with_its_expired_worker_closes_with_its_dependents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    _, ids = await _dispatched(harness, SSH_THEN_ECHO, "session")
    watchdog = WorkerWatchdog(
        MagicMock(),
        MagicMock(),
        harness.runtime,
        logging.getLogger("release-watchdog"),
        enabled=True,
        check_interval=1,
        grace_seconds=0,
    )
    watchdog.set_loss_handler(harness.monitor.record_worker_losses)

    with caplog.at_level(logging.WARNING, logger="release-watchdog"):
        watchdog._handle_worker_expired("wkr-1")

    assert "Worker wkr-1 heartbeat expired" in caplog.text
    assert harness.released(ids["session"])
    assert harness.closed_as_failed(ids["session"])
    assert harness.closed_as_failed(ids["after"])


@pytest.mark.anyio
async def test_a_task_returning_with_its_lost_worker_releases_its_dispatch() -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    _, ids = await _dispatched(harness, LINEAR, "a")

    harness.deliver(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))

    record = harness.runtime.get_record(ids["a"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert harness.released(ids["a"])


@pytest.mark.anyio
async def test_a_forward_an_update_registers_as_its_task_fails_is_released() -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    _, ids = await _dispatched(harness, SSH_THEN_ECHO, "session")
    task_id = ids["session"]
    calls: list[str] = []

    def register(*args: Any) -> dict[str, Any]:
        # The watchdog resolves the task, and releases it, before the forward lands.
        harness.monitor.record_worker_losses(
            "wkr-1",
            harness.runtime.recover_tasks_for_worker("wkr-1").resolved,
            "worker_heartbeat_expired",
        )
        calls.append("registered")
        return cast(dict[str, Any], args[-1])

    harness.forward.register_port_forward.side_effect = register
    harness.forward.unregister_task.side_effect = lambda *_: calls.append("released")

    harness.deliver(
        TaskEvent(
            type="TASK_UPDATE",
            task_id=task_id,
            worker_id="wkr-1",
            dispatch_id="dsp-1",
            payload={"ssh": {"mode": "forward", "host": "h", "port": 22}},
            ts=_TS,
        )
    )

    record = harness.runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.FAILED
    assert calls[-1] == "released"


@pytest.mark.anyio
async def test_a_serve_task_returned_before_its_adoption_is_not_adopted() -> None:
    harness = _Harness(_runtime(FakeRegistry()))
    _, ids = await _dispatched(harness, SERVE_V1, "serve")
    task_id = ids["serve"]

    harness.deliver(
        TaskEvent(
            type="TASK_UPDATE",
            task_id=task_id,
            worker_id="wkr-1",
            dispatch_id="dsp-1",
            payload={"serve": {"_host": "h", "_port": 8000, "model": "org/served"}},
            ts=_TS,
        )
    )
    # The watchdog returns the task before the adoption runs on the control loop.
    harness.deliver(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))

    current = harness.serve.adopt.call_args.args[3]
    assert current() is False
