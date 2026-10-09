"""Workflow listings page by cursor over a submission index, in a number of Redis
round trips independent of how many workflows exist."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext

from server.app_state import get_logger, get_workflow_registry
from server.auth.security import authenticate_connection
from server.clients.redis import (
    WORKFLOWS_BY_SUBMISSION_KEY,
    WORKFLOWS_SET_KEY,
    workflow_key,
)
from server.config import OrchestrationConfig
from server.registries.workflow import (
    WorkflowRecord,
    WorkflowRegistry,
    WorkflowSched,
)
from server.routers.v1 import workflows as workflows_router
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from server.utils.cursors import encode_cursor
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
    """Counts the round trips a registry makes, each pipeline and each direct call,
    and the commands they carry."""

    def __init__(
        self, registry: WorkflowRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.round_trips = 0
        self.commands = 0
        self.index_reads = 0
        client = registry._rds.asyncio
        pipeline = client.control_pipeline
        set_members = client.set_members
        lex_range = client.lex_range
        counter = self

        class _Pipeline:
            def __init__(self, transaction: bool = True) -> None:
                self._pipe = pipeline(transaction)

            async def __aenter__(self) -> Any:
                pipe = await self._pipe.__aenter__()
                execute = pipe.execute

                async def counted(raise_on_error: bool = True) -> Any:
                    counter.round_trips += 1
                    counter.commands += len(pipe.command_stack)
                    return await execute(raise_on_error)

                monkeypatch.setattr(pipe, "execute", counted)
                return pipe

            async def __aexit__(self, *exc: Any) -> Any:
                return await self._pipe.__aexit__(*exc)

        async def counted_members(key: str) -> set[str]:
            counter.round_trips += 1
            counter.commands += 1
            return await set_members(key)

        async def counted_range(*args: Any, **kwargs: Any) -> list[str]:
            counter.round_trips += 1
            counter.commands += 1
            counter.index_reads += 1
            return await lex_range(*args, **kwargs)

        monkeypatch.setattr(client, "control_pipeline", _Pipeline)
        monkeypatch.setattr(client, "set_members", counted_members)
        monkeypatch.setattr(client, "lex_range", counted_range)


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

    def ids(self, path: str, **params: str) -> list[str]:
        response = self.get(path if not params else f"{path}?{_query(params)}")
        return [w["workflow_id"] for w in response.json()["entries"]]

    def get(self, path: str) -> httpx.Response:
        async def call() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app())
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t"
            ) as client:
                return await client.get(path)

        return asyncio.run(call())


def _query(params: dict[str, str]) -> str:
    return "&".join(f"{key}={value}" for key, value in params.items())


def _seed(registry: WorkflowRegistry, count: int, indexed: bool = True) -> list[str]:
    """Register ``count`` task-less workflows one second apart, oldest first; without
    ``indexed``, write them as a registry without a submission index did."""
    ids = [f"wfl-{index:05d}" for index in range(count)]
    for index, workflow_id in enumerate(ids):
        submitted_at = datetime.fromtimestamp(1_700_000_000 + index, UTC).isoformat()
        if indexed:
            registry.register_workflow(
                workflow_id, [], WorkflowSched(), submitted_at=submitted_at
            )
            continue
        record = WorkflowRecord(
            workflow_id=workflow_id, task_ids=[], submitted_at=submitted_at
        )
        registry._rds.sync.sadd(WORKFLOWS_SET_KEY, workflow_id)
        registry._rds.sync.hash_set(workflow_key(workflow_id), record.model_dump())
    return ids


@pytest.fixture
def fabric() -> _Fabric:
    return _Fabric(25)


@pytest.mark.parametrize("total", [30, 300])
@pytest.mark.parametrize("limit", [5, 25])
def test_an_unfiltered_page_takes_two_round_trips_whatever_the_total(
    total: int, limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fabric = _Fabric(0)
    ids = _seed(fabric.registry, total)
    counting = _Counting(fabric.registry, monkeypatch)

    page = fabric.get(f"/api/v1/workflows?limit={limit}").json()
    older = fabric.get(f"/api/v1/workflows?limit={limit}&before={page['prev_cursor']}")

    older_ids = [w["workflow_id"] for w in older.json()["entries"]]
    assert [w["workflow_id"] for w in page["entries"]] == ids[-limit:]
    assert older_ids == ids[-2 * limit : -limit]
    # Per page, one index range and one read of the page's five keys per workflow.
    assert counting.round_trips == 4
    assert counting.commands == 2 + 5 * (limit + len(older_ids))


def test_workflows_registered_before_the_index_list_once_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _Fabric(0)
    ids = _seed(fabric.registry, 12, indexed=False)
    assert fabric.ids("/api/v1/workflows") == []

    assert asyncio.run(fabric.registry.index_submissions_async()) == 12
    ids += [fabric.submit(index) for index in range(3)]
    counting = _Counting(fabric.registry, monkeypatch)
    assert asyncio.run(fabric.registry.index_submissions_async()) == 0
    # A covering index is recognized by its size, without reading a workflow.
    assert (counting.round_trips, counting.commands) == (1, 2)

    assert fabric.ids("/api/v1/workflows") == ids
    assert _walk(fabric, "before", limit=4) == ids
    assert _walk(fabric, "after", limit=4) == ids


def test_the_sync_and_async_clients_read_the_same_index_range() -> None:
    client = fake_redis_client(fakeredis.FakeServer())
    members = {f"{index:016d}:wfl-{index}": 0 for index in range(6)}
    pipe = client.sync.control_pipeline()
    pipe.zadd(WORKFLOWS_BY_SUBMISSION_KEY, members)
    pipe.execute()

    for args in [("-", "+", 4, False), ("+", "(0000000000000003:wfl-3", 2, True)]:
        expected = asyncio.run(
            client.asyncio.lex_range(WORKFLOWS_BY_SUBMISSION_KEY, *args)
        )
        assert client.sync.lex_range(WORKFLOWS_BY_SUBMISSION_KEY, *args) == expected
    assert client.sync.lex_range(WORKFLOWS_BY_SUBMISSION_KEY, "-", "+", 2) == [
        "0000000000000000:wfl-0",
        "0000000000000001:wfl-1",
    ]


def _walk(fabric: _Fabric, direction: str, limit: int, **filters: str) -> list[str]:
    """Every workflow a cursor walk visits, oldest first: ``before`` from the newest
    page, ``after`` from the oldest."""
    seen: list[str] = []
    cursor: str | None = None
    if direction == "after":
        cursor = encode_cursor([0, ""])
    while True:
        params = {
            "limit": str(limit),
            **filters,
            **({direction: cursor} if cursor else {}),
        }
        page = fabric.get(f"/api/v1/workflows?{_query(params)}").json()
        ids = [w["workflow_id"] for w in page["entries"]]
        if not ids:
            return seen
        if direction == "before":
            seen = ids + seen
            cursor = page["prev_cursor"]
        else:
            seen += ids
            cursor = page["next_cursor"]


def test_a_forward_walk_visits_every_workflow_once(fabric: _Fabric) -> None:
    fabric.runtime.cancel_workflow(fabric.workflow_ids[3])
    fabric.runtime.cancel_workflow(fabric.workflow_ids[17])

    assert _walk(fabric, "after", limit=4) == fabric.workflow_ids
    assert _walk(fabric, "after", limit=1, status="CANCELLED") == [
        fabric.workflow_ids[3],
        fabric.workflow_ids[17],
    ]
    assert _walk(fabric, "before", limit=1, status="CANCELLED") == [
        fabric.workflow_ids[3],
        fabric.workflow_ids[17],
    ]


def test_a_workflow_id_pushdown_reads_no_index(
    fabric: _Fabric, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = fabric.workflow_ids[:2]
    counting = _Counting(fabric.registry, monkeypatch)

    assert fabric.ids(
        f"/api/v1/workflows?workflow_id={first}&workflow_id={second}"
    ) == [
        first,
        second,
    ]
    assert counting.index_reads == 0
    # The candidates' submission times, then the page.
    assert counting.round_trips == 2


def test_one_workflow_is_read_in_one_round_trip(
    fabric: _Fabric, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = _Counting(fabric.registry, monkeypatch)

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


def test_a_filter_scans_past_entries_it_rejects_in_growing_chunks(
    fabric: _Fabric, monkeypatch: pytest.MonkeyPatch
) -> None:
    fabric.runtime.cancel_workflow(fabric.workflow_ids[0])
    fabric.runtime.cancel_workflow(fabric.workflow_ids[1])
    counting = _Counting(fabric.registry, monkeypatch)

    entries = fabric.ids("/api/v1/workflows", limit="1", status="CANCELLED")

    assert entries == [fabric.workflow_ids[1]]
    # Chunks of 1, 2, 4, 8 and 16 reach the 24th newest: an index range and a read
    # each.
    assert counting.round_trips == 10


@pytest.mark.parametrize(
    "cursor",
    [
        encode_cursor([1.5, "wfl-1"]),
        encode_cursor([True, "wfl-1"]),
        encode_cursor([-1, "wfl-1"]),
        encode_cursor([10**16, "wfl-1"]),
        encode_cursor([10**400, "wfl-1"]),
    ],
    ids=["task-cursor", "bool", "negative", "past-the-index", "overflow"],
)
def test_a_cursor_this_route_did_not_issue_is_rejected(
    fabric: _Fabric, cursor: str
) -> None:
    response = fabric.get(f"/api/v1/workflows?before={cursor}")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_cursor"


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
