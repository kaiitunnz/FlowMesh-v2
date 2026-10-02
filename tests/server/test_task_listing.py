"""Task listings page by cursor, filter only on declared fields, and leave the event
loop free while they build."""

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from flowmesh import FlowMesh
from lumid_hooks import PrincipalContext
from pydantic import TypeAdapter

from server.app_state import get_logger, get_runtime
from server.auth.security import authenticate_connection
from server.routers.v1 import tasks as tasks_router
from server.schemas.tasks import TaskPage
from server.services import log_archiver
from server.services.log_archiver import TaskLogArchiver
from server.task import runtime as runtime_module
from server.task.models import TaskInfo
from server.task.runtime import TaskRuntime
from tests.server.task.test_v2_orchestration import FakeRegistry, _live_runtime

_LOGGER = logging.getLogger("test.task_listing")
_WORKFLOWS = 1000
_STAGES = 5
_FILLER = " ".join(f"word{i:04d}" for i in range(120))


def _workflow(index: int, stages: int = _STAGES) -> str:
    body = "\n".join(f"""    - name: s-{stage}
      spec:
        data:
          type: list
          items:
            - "{index}-{stage} {_FILLER}"
            - "{_FILLER}\"""" for stage in range(stages))
    return f"""apiVersion: flowmesh/v1
kind: EchoTask
metadata:
  name: listing-{index}
  annotations:
    description: "{_FILLER}"
    schedule_hint:
      selected_worker:
        global: [wkr-a]
spec:
  taskType: echo
  stages:
{body}
"""


async def _register(runtime: TaskRuntime, payload: str) -> tuple[str, list[str]]:
    workflow_id, entries = await runtime.register(
        "owner", "org", payload, format="native"
    )
    return workflow_id, [entry.task_id for entry in entries]


class _Listing:
    def __init__(self, runtime: TaskRuntime, workflows: dict[str, list[str]]):
        self.runtime = runtime
        self.workflows = workflows

    @property
    def task_ids(self) -> set[str]:
        return {task for tasks in self.workflows.values() for task in tasks}


@pytest.fixture(scope="module")
def listing() -> _Listing:
    runtime = _live_runtime(FakeRegistry())

    async def seed() -> dict[str, list[str]]:
        workflows: dict[str, list[str]] = {}
        for index in range(_WORKFLOWS):
            workflow_id, tasks = await _register(runtime, _workflow(index))
            workflows[workflow_id] = tasks
        return workflows

    return _Listing(runtime, asyncio.run(seed()))


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="owner",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


def _app(runtime: TaskRuntime) -> FastAPI:
    app = FastAPI()
    app.include_router(tasks_router.router, prefix="/api/v1")
    app.dependency_overrides[authenticate_connection] = _principal
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[get_logger] = lambda: _LOGGER

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


@contextmanager
def _serving(app: FastAPI) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            ws="none",
            loop="asyncio",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)


def _get(app: FastAPI, path: str) -> httpx.Response:
    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get(path)

    return asyncio.run(call())


def _count_task_infos(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    built: list[str] = []

    class _Counting(TaskInfo):
        def __init__(self, **data: Any) -> None:
            built.append(data["task_id"])
            super().__init__(**data)

    monkeypatch.setattr(runtime_module, "TaskInfo", _Counting)
    return built


@pytest.mark.parametrize("listings", [1, 3])
def test_a_listing_leaves_the_event_loop_free(
    listing: _Listing, monkeypatch: pytest.MonkeyPatch, listings: int
) -> None:
    # The first build holds until /ping returns, so /ping always overlaps a
    # listing; a build on the event loop holds /ping instead.
    building = threading.Event()
    released = threading.Event()

    class _Holding(TaskInfo):
        def __init__(self, **data: Any) -> None:
            if not building.is_set():
                building.set()
                released.wait(2.0)
            super().__init__(**data)

    monkeypatch.setattr(runtime_module, "TaskInfo", _Holding)
    statuses: list[int] = []

    with _serving(_app(listing.runtime)) as url:

        def list_tasks() -> None:
            with httpx.Client(timeout=60) as client:
                response = client.get(f"{url}/api/v1/tasks?limit=1000")
                statuses.append(response.status_code)

        threads = [threading.Thread(target=list_tasks) for _ in range(listings)]
        for thread in threads:
            thread.start()
        assert building.wait(30), "the listing never started building"
        with httpx.Client(timeout=10) as client:
            started = time.perf_counter()
            assert client.get(f"{url}/ping").status_code == 200
            waited = time.perf_counter() - started
        released.set()
        for thread in threads:
            thread.join(60)

    assert statuses == [200] * listings
    assert waited < 0.25, f"/ping waited {waited:.3f}s behind the listing"


def test_a_page_serializes_off_the_event_loop(
    listing: _Listing, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    dump = TaskPage.model_dump_json

    def _recording(self: TaskPage, **kwargs: Any) -> str:
        threads.append(threading.get_ident())
        return dump(self, **kwargs)

    monkeypatch.setattr(TaskPage, "model_dump_json", _recording)

    async def call() -> tuple[int, int]:
        transport = httpx.ASGITransport(app=_app(listing.runtime))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            response = await c.get("/api/v1/tasks?limit=10")
        return response.status_code, threading.get_ident()

    status, loop_thread = asyncio.run(call())
    assert status == 200
    assert threads and loop_thread not in threads


def test_a_workflow_filter_builds_only_that_workflows_tasks(
    listing: _Listing, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_id, tasks = next(iter(listing.workflows.items()))
    built = _count_task_infos(monkeypatch)

    response = _get(_app(listing.runtime), f"/api/v1/tasks?workflow_id={workflow_id}")

    assert response.status_code == 200
    assert {t["task_id"] for t in response.json()["entries"]} == set(tasks)
    assert sorted(built) == sorted(tasks)


def test_a_repeated_workflow_filter_lists_the_union(listing: _Listing) -> None:
    (first, first_tasks), (second, second_tasks) = list(listing.workflows.items())[:2]

    response = _get(
        _app(listing.runtime),
        f"/api/v1/tasks?workflow_id={first}&workflow_id={second}",
    )

    assert {t["task_id"] for t in response.json()["entries"]} == {
        *first_tasks,
        *second_tasks,
    }


def test_declared_filters_combine(listing: _Listing) -> None:
    workflow_id, tasks = next(iter(listing.workflows.items()))
    response = _get(
        _app(listing.runtime),
        f"/api/v1/tasks?workflow_id={workflow_id}&status=PENDING&status=DONE"
        "&task_type=echo&completed=false&graph_node_name=nothing",
    )
    assert response.json()["entries"] == []

    response = _get(
        _app(listing.runtime),
        f"/api/v1/tasks?workflow_id={workflow_id}&status=PENDING&completed=false",
    )
    assert {t["task_id"] for t in response.json()["entries"]} == set(tasks)


@pytest.mark.parametrize(
    "key",
    [
        "owner",
        "raw_yaml",
        "task.spec.env.OPENAI_KEY",
        "task.spec.api.headers.Authorization",
        "task.spec.model.api_key",
        "latest_update.ssh.password",
    ],
)
def test_an_undeclared_filter_is_rejected(listing: _Listing, key: str) -> None:
    response = _get(_app(listing.runtime), f"/api/v1/tasks?{key}=guess")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"


def test_an_unbounded_listing_returns_the_newest_page(listing: _Listing) -> None:
    response = _get(_app(listing.runtime), "/api/v1/tasks")

    page = response.json()
    newest = sorted(
        listing.runtime.tasks.values(), key=lambda r: (r.submitted_ts, r.task_id)
    )[-100:]
    assert [t["task_id"] for t in page["entries"]] == [r.task_id for r in newest]
    assert _get(_app(listing.runtime), "/api/v1/tasks?limit=1001").status_code == 422


@pytest.mark.parametrize(
    "query, code",
    [
        ("before=x&after=y", "invalid_request"),
        ("before=not-a-cursor", "invalid_cursor"),
        ("after=WyJhIiwiYiJd", "invalid_cursor"),
    ],
)
def test_a_bad_cursor_is_rejected(listing: _Listing, query: str, code: str) -> None:
    response = _get(_app(listing.runtime), f"/api/v1/tasks?{query}")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == code


def test_cursors_walk_every_task_once_while_tasks_are_added() -> None:
    runtime = _live_runtime(FakeRegistry())
    for index in range(6):
        asyncio.run(_register(runtime, _workflow(index, stages=3)))
    app = _app(runtime)
    initial = set(runtime.tasks)

    seen: list[str] = []
    before: str | None = None
    while True:
        page = _get(
            app, f"/api/v1/tasks?limit=4{f'&before={before}' if before else ''}"
        )
        entries = page.json()["entries"]
        if not entries:
            break
        seen = [t["task_id"] for t in entries] + seen
        before = page.json()["prev_cursor"]
        asyncio.run(_register(runtime, _workflow(100 + len(seen), stages=2)))
    assert sorted(seen) == sorted(initial)
    assert len(seen) == len(set(seen))

    oldest = min(runtime.tasks.values(), key=lambda r: (r.submitted_ts, r.task_id))
    info = runtime.describe_task(oldest.task_id)
    assert info is not None
    forward = [oldest.task_id]
    page = _get(app, f"/api/v1/tasks?limit=5&after={tasks_router._task_cursor(info)}")
    while entries := page.json()["entries"]:
        forward.extend(t["task_id"] for t in entries)
        page = _get(app, f"/api/v1/tasks?limit=5&after={page.json()['next_cursor']}")
    assert forward == sorted(
        runtime.tasks, key=lambda t: (runtime.tasks[t].submitted_ts, t)
    )


def test_the_sdk_lists_every_task_across_pages(listing: _Listing) -> None:
    workflow_id, tasks = next(iter(listing.workflows.items()))
    with _serving(_app(listing.runtime)) as url:
        client = FlowMesh(base_url=url, api_key="k")
        listed = client.tasks.list()
        filtered = client.tasks.list(workflow_id=workflow_id)
        with pytest.raises(ValueError, match="limit"):
            client.tasks.list(query_params=[("limit", "5")])

    assert len(listed) == len(listing.task_ids) > 1000
    assert {t.task_id for t in listed} == listing.task_ids
    assert [t.submitted_ts for t in listed] == sorted(t.submitted_ts for t in listed)
    assert {t.task_id for t in filtered} == set(tasks)


def test_entries_serialize_as_the_response_model_does(listing: _Listing) -> None:
    workflow_id, tasks = next(iter(listing.workflows.items()))
    response = _get(_app(listing.runtime), f"/api/v1/tasks?workflow_id={workflow_id}")

    infos = []
    for task_id in sorted(tasks, key=lambda t: listing.runtime.tasks[t].submitted_ts):
        info = listing.runtime.describe_task(task_id)
        assert info is not None
        tasks_router._sanitize_latest_update(info)
        infos.append(info)
    expected = TypeAdapter(list[TaskInfo]).dump_python(
        infos, mode="json", by_alias=True
    )
    entries = response.json()["entries"]
    assert entries == expected
    hint = entries[0]["task"]["metadata"]["annotations"]["schedule_hint"]
    assert hint["selected_worker"] == {"global": ["wkr-a"], "selected": None}


def test_the_schema_names_the_page_model(listing: _Listing) -> None:
    schema = _app(listing.runtime).openapi()
    response = schema["paths"]["/api/v1/tasks"]["get"]["responses"]["200"]
    ref = response["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/TaskPage")


def test_a_listing_never_redacts(
    listing: _Listing, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    redact = runtime_module.redact_source_text

    def _counting(payload: str, format: str) -> str:
        calls.append(format)
        return redact(payload, format)

    monkeypatch.setattr(runtime_module, "redact_source_text", _counting)
    asyncio.run(_register(listing.runtime, _workflow(9999)))
    assert calls == ["native"]

    _get(_app(listing.runtime), "/api/v1/tasks?limit=1000")
    for record in list(listing.runtime.tasks.values())[:50]:
        record.model_dump_json()
    assert calls == ["native"]


def test_the_log_archiver_builds_no_task_info(
    listing: _Listing, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _count_task_infos(monkeypatch)
    archiver = TaskLogArchiver(cast(Any, None), listing.runtime, tmp_path, _LOGGER)
    tracked: list[str] = []
    monkeypatch.setattr(
        archiver, "_ensure_task", lambda task_id, now: tracked.append(task_id)
    )
    monkeypatch.setattr(log_archiver.time, "sleep", lambda _: None)

    archiver._tick()

    assert built == []
    assert sorted(tracked) == sorted(listing.runtime.tasks)
