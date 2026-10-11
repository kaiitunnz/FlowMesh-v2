"""The task and output routes answer a freed workflow as they did while it was
resident."""

import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from lumid_hooks import PrincipalContext

from server.routers.v1 import outputs as outputs_router
from server.routers.v1 import tasks as tasks_router
from tests.server.task.test_region_outputs import _CONCAT
from tests.server.task.test_runtime_control_flow import _Run, _workflow
from tests.server.task.test_settled_history import _registry, _resident

_LOGGER = logging.getLogger("test.settled_history_routes")
_PRINCIPAL = PrincipalContext(
    principal_id="p-1",
    org_id="org",
    external_id="ext",
    principal_type="user",
    scopes=[],
)


async def _answer(call: Callable[[], Awaitable[Any]]) -> Any:
    try:
        response = await call()
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return response.model_dump(mode="json")


async def _answers(run: _Run) -> list[Any]:
    runtime = run.runtime
    calls: list[Callable[[], Awaitable[Any]]] = []
    for task_id in [*run.ids.values(), "tsk-unknown"]:
        calls.append(
            partial(
                tasks_router.get_task,
                task_id,
                principal=_PRINCIPAL,
                runtime=runtime,
                logger=_LOGGER,
            )
        )
        calls.append(
            partial(
                tasks_router.stop_task,
                task_id,
                principal=_PRINCIPAL,
                runtime=runtime,
                worker_registry=MagicMock(),
                logger=_LOGGER,
            )
        )
    calls.append(
        lambda: outputs_router.list_outputs(
            run.workflow_id,
            limit=10,
            before=None,
            after=None,
            output=None,
            scope=None,
            principal=_PRINCIPAL,
            runtime=runtime,
            logger=_LOGGER,
        )
    )
    for key in ("l", "r", "missing"):
        calls.append(
            partial(
                outputs_router.get_output,
                run.workflow_id,
                "both",
                scope=None,
                key=key,
                sequence=None,
                principal=_PRINCIPAL,
                runtime=runtime,
                logger=_LOGGER,
            )
        )
    return [await _answer(call) for call in calls]


@pytest.mark.anyio
async def test_the_routes_answer_a_freed_workflow_as_they_did() -> None:
    run = await _Run(registry=_registry()).start(_workflow(_CONCAT))  # type: ignore[arg-type]
    run.run("a", {"inner": {"v": 1}, "outer": 2})
    run.run("b", {"v": 3})
    assert run.settled()
    answers = await _answers(run)
    await run.runtime.start_settled_history()

    run.runtime.settled_history.drain()

    assert not _resident(run.runtime, run.workflow_id)
    assert await _answers(run) == answers
