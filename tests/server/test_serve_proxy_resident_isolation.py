"""The legacy serve proxy cannot reach a resident allocation.

A resident replica's serve task is marked resident, so the proxy refuses to resolve a
relay target for it by task id — structurally, by allocation identity — even when its
access mode would otherwise be proxiable.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException

from server.routers.v1 import serve as serve_router
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.tasks import TaskType


def _runtime(*, resident: bool) -> TaskRuntime:
    record = SimpleNamespace(
        task_type=TaskType.SERVE,
        task=SimpleNamespace(spec=SimpleNamespace(resident=resident)),
        status=TaskStatus.DISPATCHED,
        latest_update={
            "serve": {
                "mode": "proxy",
                "_relay_target": {"host": "127.0.0.1", "port": 9001},
            }
        },
    )
    return cast(TaskRuntime, SimpleNamespace(get_record=lambda task_id: record))


def test_resident_serve_task_is_not_proxiable_even_in_proxy_mode() -> None:
    with pytest.raises(HTTPException) as excinfo:
        serve_router._resolve_serve_relay_target(_runtime(resident=True), "tsk-1")
    assert excinfo.value.status_code == 404


def test_nonresident_proxy_serve_task_still_resolves() -> None:
    record, host, port = serve_router._resolve_serve_relay_target(
        _runtime(resident=False), "tsk-1"
    )
    assert (host, port) == ("127.0.0.1", 9001)
