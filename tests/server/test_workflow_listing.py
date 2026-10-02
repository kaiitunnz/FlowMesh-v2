"""Workflow listings page by cursor in a fixed number of Redis round trips."""

import asyncio
import logging
from typing import Any, cast

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext

from server.app_state import get_logger, get_workflow_registry
from server.auth.security import authenticate_connection
from server.config import OrchestrationConfig
from server.registries.workflow import WorkflowRegistry
from server.routers.v1 import workflows as workflows_router
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.redis_helpers import fake_redis_client
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import _WorkerRegistryStub

_LOGGER = logging.getLogger("test.workflow_listing")


def _workflow(index: int, stages: int = 2) -> str:
    body = "\n".join(f"""    - name: s-{stage}
      spec:
        data: {{type: list, items: ["{index}-{stage}"]}}""" for stage in range(stages))
    return f"""apiVersion: flowmesh/v1
kind: EchoTask
metadata: {{name: wf-{index}}}
spec:
  taskType: echo
  stages:
{body}
"""


class _Counting:
    """Counts the round trips a registry makes: each pipeline and each direct call."""

    def __init__(self, registry: WorkflowRegistry) -> None:
        self.round_trips = 0
        client = registry._rds.asyncio
        pipeline = client.control_pipeline
        set_members = client.set_members
        counter = self

        class _Pipeline:
            def __init__(self) -> None:
                self._pipe = pipeline()

            async def __aenter__(self) -> Any:
                pipe = await self._pipe.__aenter__()
                execute = pipe.execute

                async def counted() -> Any:
                    counter.round_trips += 1
                    return await execute()

                pipe.execute = counted
                return pipe

            async def __aexit__(self, *exc: Any) -> Any:
                return await self._pipe.__aexit__(*exc)

        async def counted_members(key: str) -> set[str]:
            counter.round_trips += 1
            return await set_members(key)

        cast(Any, client).control_pipeline = _Pipeline
        cast(Any, client).set_members = counted_members


class _Fabric:
    def __init__(self, workflows: int) -> None:
        self.registry = WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))
        self.runtime = TaskRuntime(
            self.registry,
            cast(Any, _WorkerRegistryStub()),
            OrchestrationConfig(),
            make_result_reader(),
            _LOGGER,
            credential_vault=InMemoryCredentialVault(),
            redrive=lambda fire, logger: StoreRedriveScheduler(
                fire, logger, run_thread=False
            ),
        )
        self.workflow_ids = [self.submit(index) for index in range(workflows)]

    def submit(self, index: int) -> str:
        workflow_id, _ = asyncio.run(
            self.runtime.register("owner", "org", _workflow(index), format="native")
        )
        return workflow_id

    def app(self) -> FastAPI:
        app = FastAPI()
        app.include_router(workflows_router.router, prefix="/api/v1")
        app.dependency_overrides[authenticate_connection] = lambda: PrincipalContext(
            principal_id="owner",
            org_id="org",
            external_id="ext",
            principal_type="user",
            scopes=[],
        )
        app.dependency_overrides[get_workflow_registry] = lambda: self.registry
        app.dependency_overrides[get_logger] = lambda: _LOGGER
        return app

    def get(self, path: str) -> httpx.Response:
        async def call() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app())
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t"
            ) as client:
                return await client.get(path)

        return asyncio.run(call())


@pytest.fixture
def fabric() -> _Fabric:
    return _Fabric(25)


@pytest.mark.parametrize("limit", [5, 25])
def test_a_page_takes_three_round_trips_whatever_its_size(
    fabric: _Fabric, limit: int
) -> None:
    counting = _Counting(fabric.registry)

    response = fabric.get(f"/api/v1/workflows?limit={limit}")

    assert len(response.json()["entries"]) == limit
    assert counting.round_trips == 3


def test_one_workflow_is_read_in_one_round_trip(fabric: _Fabric) -> None:
    counting = _Counting(fabric.registry)

    workflow = asyncio.run(fabric.registry.get_workflow_async(fabric.workflow_ids[0]))

    assert workflow is not None and workflow.workflow_id == fabric.workflow_ids[0]
    assert counting.round_trips == 1


def test_cursors_walk_every_workflow_once_while_workflows_are_added(
    fabric: _Fabric,
) -> None:
    seen: list[str] = []
    before: str | None = None
    while True:
        page = fabric.get(
            f"/api/v1/workflows?limit=4{f'&before={before}' if before else ''}"
        ).json()
        if not page["entries"]:
            break
        seen = [w["workflow_id"] for w in page["entries"]] + seen
        before = page["prev_cursor"]
        fabric.submit(100 + len(seen))

    assert seen == fabric.workflow_ids


def test_an_unbounded_listing_returns_the_newest_page() -> None:
    fabric = _Fabric(105)

    entries = fabric.get("/api/v1/workflows").json()["entries"]

    assert [w["workflow_id"] for w in entries] == fabric.workflow_ids[-100:]


def test_a_filter_scans_past_pages_it_rejects(fabric: _Fabric) -> None:
    fabric.runtime.cancel_workflow(fabric.workflow_ids[0])
    fabric.runtime.cancel_workflow(fabric.workflow_ids[1])
    counting = _Counting(fabric.registry)

    entries = fabric.get("/api/v1/workflows?limit=2&status=CANCELLED").json()["entries"]

    assert [w["workflow_id"] for w in entries] == fabric.workflow_ids[:2]
    # members, order, then one page read per two workflows scanned
    assert counting.round_trips == 2 + len(fabric.workflow_ids) // 2 + 1


def test_workflow_filters_match_declared_fields(fabric: _Fabric) -> None:
    first, second = fabric.workflow_ids[:2]

    entries = fabric.get(
        f"/api/v1/workflows?workflow_id={first}&workflow_id={second}"
    ).json()["entries"]

    assert [w["workflow_id"] for w in entries] == [first, second]
    response = fabric.get("/api/v1/workflows?owner=owner")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"


def test_the_schema_names_the_page_model(fabric: _Fabric) -> None:
    response = fabric.app().openapi()["paths"]["/api/v1/workflows"]["get"]
    schema = response["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/WorkflowPage")
