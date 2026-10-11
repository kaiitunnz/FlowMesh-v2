"""Undeliverable-dispatch grace behavior.

When a worker is selected but the task cannot be delivered to it (its node has
no live dispatch subscriber, or publish raises), the dispatcher must treat this
as an infrastructure fault: retry without spending the task's ``max_attempts``
budget, then fail with a descriptive error after the no-worker grace.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.task.models import TaskRecord
from tests.server.task.test_runtime_rehydrate import (
    GRAPH,
    FakeWorkflowRegistry,
    _register,
    _runtime,
)
from tests.support.waiting import pop_ready

from .helpers import make_capturing_dispatcher


def _record() -> TaskRecord:
    return cast(
        TaskRecord,
        SimpleNamespace(
            no_dispatch_since=None,
            last_error=None,
            last_failed_worker=None,
        ),
    )


def test_undeliverable_within_grace_requeues_without_retry() -> None:
    runtime = mock.Mock()
    disp = make_capturing_dispatcher(runtime=runtime, grace_sec=60)
    record = _record()

    result = disp._grace_then_fail_undeliverable(
        "tsk-1", record, reason="no_dispatch_subscriber", message="orphaned"
    )

    assert result is False
    assert not disp.failed
    assert len(disp.requeued) == 1
    _, kwargs = disp.requeued[0]
    # Infra fault must NOT consume the execution retry budget.
    assert kwargs["count_retry"] is False
    assert kwargs["reason"] == "no_dispatch_subscriber"
    assert record.no_dispatch_since is not None
    runtime.note_dispatch_error.assert_called_once_with("tsk-1", "orphaned")


def test_undeliverable_after_grace_fails_with_message() -> None:
    disp = make_capturing_dispatcher(grace_sec=60)
    record = _record()
    # Pretend the condition has persisted longer than the grace window.
    record.no_dispatch_since = time.time() - 120

    result = disp._grace_then_fail_undeliverable(
        "tsk-1",
        record,
        reason="no_dispatch_subscriber",
        message="orphaned worker w1 on node n1",
        extra_payload={"worker_id": "w1", "node_id": "n1"},
    )

    assert result is False
    assert not disp.requeued
    assert len(disp.failed) == 1
    task_id, error_message, kwargs = disp.failed[0]
    assert task_id == "tsk-1"
    assert error_message == "orphaned worker w1 on node n1"
    payload = kwargs["payload"]
    assert payload["reason"] == "no_dispatch_subscriber"
    assert payload["worker_id"] == "w1"
    assert payload["node_id"] == "n1"


def _served(runtime: Any, task_id: str) -> dict[str, Any]:
    info = runtime.describe_task(task_id)
    assert info is not None
    return info.model_dump(mode="json")


def test_an_undeliverable_dispatch_leaves_a_task_cancelled_meanwhile_unchanged() -> (
    None
):
    runtime = _runtime(FakeWorkflowRegistry())
    workflow_id, ids = asyncio.run(_register(runtime, GRAPH))
    task_id = ids["a"]
    assert pop_ready(runtime) == task_id
    record = runtime.get_record(task_id)
    assert record is not None
    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    runtime.begin_publish(task_id, cast(Any, worker), "dsp-1")
    assert runtime.abandon_publish(task_id)
    runtime.cancel_workflow(workflow_id)
    cancelled = _served(runtime, task_id)

    disp = make_capturing_dispatcher(runtime=runtime)
    disp._grace_then_fail_undeliverable(
        task_id, record, reason="publish_failed", message="unreachable"
    )

    assert _served(runtime, task_id) == cancelled
    assert record.last_error is None
