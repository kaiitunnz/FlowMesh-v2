"""list() walks a cursor-paged listing back from the newest page."""

from typing import Any

import httpx
import pytest
import respx
from flowmesh import AsyncFlowMesh, FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.resources import _pages

from .router_app import TEST_BASE_URL, route_url

_LIMIT = 2


def _workflow(index: int) -> dict[str, Any]:
    return {
        "workflow_id": f"wfl-{index}",
        "task_ids": [],
        "submitted_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "status": "PENDING",
        "dispatched_tasks": [],
        "completed_tasks": [],
        "failed_tasks": [],
        "cancelled_tasks": [],
    }


def _task(index: int) -> dict[str, Any]:
    return {
        "task_id": f"tsk-{index}",
        "workflow_id": "wfl-1",
        "owner_id": "owner",
        "org_id": "org",
        "supplier_id": "supplier",
        "raw_yaml": "",
        "task": {},
        "status": "PENDING",
        "submitted_at": "2025-01-01T00:00:00Z",
        "submitted_ts": float(index),
        "usages": [],
        "attempts": 0,
        "max_attempts": 1,
        "load": 1,
        "depends_on": [],
        "pending_dependencies": [],
        "dependents": [],
        "completed": False,
        "failed": False,
    }


class _Server:
    """Serves ``count`` entries newest-first by a ``before`` cursor naming an index."""

    def __init__(self, entry: Any, count: int) -> None:
        self.entry = entry
        self.count = count
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        limit = int(request.url.params["limit"])
        end = int(request.url.params.get("before", self.count))
        start = max(0, end - limit)
        entries = [self.entry(index) for index in range(start, end)]
        return httpx.Response(
            200,
            json={
                "entries": entries,
                "next_cursor": str(end - 1) if entries else None,
                "prev_cursor": str(start) if entries else None,
            },
        )


@pytest.fixture(autouse=True)
def _small_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_pages, "PAGE_LIMIT", _LIMIT)


@pytest.mark.parametrize("count, requests", [(5, 3), (4, 3), (0, 1)])
@pytest.mark.parametrize(
    "route, entry, prefix",
    [("list_workflows", _workflow, "wfl"), ("list_tasks", _task, "tsk")],
)
@respx.mock
def test_sync_lists_walk_every_page(
    route: str, entry: Any, prefix: str, count: int, requests: int
) -> None:
    server = _Server(entry, count)
    respx.get(route_url(route)).mock(side_effect=server)
    client = FlowMesh(base_url=TEST_BASE_URL, api_key="k")

    if route == "list_workflows":
        ids = [w.workflow_id for w in client.workflows.list()]
    else:
        ids = [t.task_id for t in client.tasks.list()]

    assert ids == [f"{prefix}-{index}" for index in range(count)]
    # A full page is followed by the next older one; a short page is the oldest.
    assert len(server.requests) == requests


@pytest.mark.anyio
@respx.mock
async def test_async_lists_walk_every_page() -> None:
    workflows = _Server(_workflow, 5)
    tasks = _Server(_task, 5)
    respx.get(route_url("list_workflows")).mock(side_effect=workflows)
    respx.get(route_url("list_tasks")).mock(side_effect=tasks)

    async with AsyncFlowMesh(base_url=TEST_BASE_URL, api_key="k") as client:
        listed_workflows = await client.workflows.list()
        listed_tasks = await client.tasks.list()

    assert [w.workflow_id for w in listed_workflows] == [f"wfl-{i}" for i in range(5)]
    assert [t.task_id for t in listed_tasks] == [f"tsk-{i}" for i in range(5)]


@respx.mock
def test_a_paging_key_is_an_sdk_error() -> None:
    client = FlowMesh(base_url=TEST_BASE_URL, api_key="k")

    with pytest.raises(FlowMeshError, match="limit"):
        client.tasks.list(query_params=[("limit", "5")])


@respx.mock
def test_a_server_that_does_not_page_is_an_sdk_error() -> None:
    respx.get(route_url("list_workflows")).respond(json=[_workflow(0)])
    client = FlowMesh(base_url=TEST_BASE_URL, api_key="k")

    with pytest.raises(FlowMeshError, match="does not page"):
        client.workflows.list()


@respx.mock
def test_a_cursor_that_does_not_advance_is_an_sdk_error() -> None:
    respx.get(route_url("list_workflows")).respond(
        json={
            "entries": [_workflow(0), _workflow(1)],
            "next_cursor": "c",
            "prev_cursor": "c",
        }
    )
    client = FlowMesh(base_url=TEST_BASE_URL, api_key="k")

    with pytest.raises(FlowMeshError, match="same page"):
        client.workflows.list()
