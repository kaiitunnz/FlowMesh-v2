"""A worker's task events name the dispatch running the task."""

import logging
from collections.abc import Callable
from typing import Any, cast

import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from worker.supervisor_client import SupervisorClient


def _client() -> SupervisorClient:
    client = SupervisorClient(
        worker_token="t",
        owner_principal=None,
        grpc_target="x",
        worker_namespace="ns",
        worker_cluster="c",
        worker_alias="a",
        logger=logging.getLogger("dispatch-id"),
    )
    client._worker_id = "wkr-1"
    client._stub = cast(Any, object())
    client._event_ready.set()
    client._stop.clear()
    return client


def _message(task_id: str, dispatch_id: str | None) -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": task_id,
            "workflow_id": "wf-1",
            "owner_id": "owner",
            "assigned_worker": "wkr-1",
            "dispatched_at": "2026-09-23T00:00:00Z",
            "dispatch_id": dispatch_id,
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "Task",
                "metadata": {"name": "wf:a"},
                "spec": {"taskType": "echo"},
            },
        }
    )


_REPORTS: dict[str, Callable[[SupervisorClient, str], None]] = {
    "TASK_STARTED": lambda client, task_id: client.task_started(task_id),
    "TASK_UPDATE": lambda client, task_id: client.task_update(task_id, {}),
    "TASK_SUCCEEDED": lambda client, task_id: client.task_succeeded(task_id),
    "TASK_FAILED": lambda client, task_id: client.task_failed(task_id, "boom"),
    "TASK_CANCELLED": lambda client, task_id: client.task_cancelled(task_id),
}


def _reported_dispatch(client: SupervisorClient, event_type: str, task_id: str) -> Any:
    _REPORTS[event_type](client, task_id)
    frame = cast(dict[str, Any], client._event_queue.get_nowait())
    assert frame["type"] == event_type
    return frame.get("dispatch_id")


@pytest.mark.parametrize("event_type", list(_REPORTS))
def test_a_task_event_names_the_dispatch_running_the_task(event_type: str) -> None:
    client = _client()
    for dispatch_id in ("dsp-1", "dsp-2"):
        client._task_queue.put(_message("tsk-a", dispatch_id))
    tasks = iter(client.iter_tasks())

    next(tasks)
    assert _reported_dispatch(client, event_type, "tsk-a") == "dsp-1"
    next(tasks)
    assert _reported_dispatch(client, event_type, "tsk-a") == "dsp-2"


def test_a_task_the_worker_is_not_running_names_no_dispatch() -> None:
    client = _client()
    client._task_queue.put(_message("tsk-a", "dsp-1"))
    next(iter(client.iter_tasks()))

    assert _reported_dispatch(client, "TASK_FAILED", "tsk-b") is None


def test_a_dispatch_without_an_id_is_reported_without_one() -> None:
    client = _client()
    client._task_queue.put(_message("tsk-a", None))
    next(iter(client.iter_tasks()))

    assert _reported_dispatch(client, "TASK_SUCCEEDED", "tsk-a") is None
