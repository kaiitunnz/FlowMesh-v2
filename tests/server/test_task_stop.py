"""Stopping a running task sends its worker a stop command for its dispatch."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from lumid_hooks import PrincipalContext

from server.routers.v1.tasks import stop_task
from server.task.runtime import TaskRuntime
from shared.tasks import TaskType

_PRINCIPAL = PrincipalContext(
    principal_id="p", org_id="o", external_id="e", principal_type="user", scopes=["*"]
)


def _stop(task_type: TaskType) -> MagicMock:
    record = SimpleNamespace(
        task=SimpleNamespace(spec=SimpleNamespace(taskType=task_type)),
        status="DISPATCHED",
        assigned_worker="wkr-1",
        dispatch_id="dsp-1",
    )
    runtime = MagicMock()
    runtime.get_record.return_value = record
    workers = MagicMock()
    workers.get_worker_async = AsyncMock(return_value=SimpleNamespace(id="wkr-1"))
    workers.publish_stop_async = AsyncMock()
    asyncio.run(
        stop_task(
            "tsk-1",
            _PRINCIPAL,
            cast(TaskRuntime, runtime),
            cast(Any, workers),
            logging.getLogger("test.task_stop"),
        )
    )
    return workers.publish_stop_async


@pytest.mark.parametrize(
    "task_type", [TaskType.SERVE, TaskType.DEV_MODEL, TaskType.SSH]
)
def test_a_running_serve_or_ssh_task_is_sent_a_stop(task_type: TaskType) -> None:
    publish = _stop(task_type)

    stop = publish.await_args.args[1]
    assert (stop.task_id, stop.worker_id, stop.dispatch_id) == (
        "tsk-1",
        "wkr-1",
        "dsp-1",
    )


def test_a_task_type_with_no_stop_is_refused() -> None:
    with pytest.raises(HTTPException) as refused:
        _stop(TaskType.INFERENCE)

    assert refused.value.status_code == 501
