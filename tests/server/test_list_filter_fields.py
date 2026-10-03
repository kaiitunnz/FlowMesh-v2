"""Every list route filters on fields its response model serves, and nothing else."""

import asyncio
import logging
import types
from collections.abc import Callable, Collection, Coroutine, Iterator
from typing import Annotated, Any, Union, cast, get_args, get_origin

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from lumid_hooks import PrincipalContext, ResourceRef
from pydantic import BaseModel
from starlette.datastructures import QueryParams

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import nodes, resident, ssh, stack, tasks, workers, workflows
from server.routers.v1._listing import PAGE_PARAMS, ListFilter
from server.task.models import TaskRecord
from server.utils.query import InvalidQuery, QueryFilter
from tests.server.task.test_v2_orchestration import FakeRegistry, _live_runtime

_LOGGER = logging.getLogger("test.list_filter_fields")
_PRINCIPAL = PrincipalContext(
    principal_id="p-1",
    org_id="org",
    external_id="ext",
    principal_type="user",
    scopes=[],
)

# The routes paging by cursor, whose own query parameters sit beside their filters.
_PAGED = {"/tasks", "/workflows"}
_ROUTES = [
    (tasks.router, "/tasks", tasks.TASK_FILTER_FIELDS),
    (workflows.router, "/workflows", workflows.WORKFLOW_FILTER_FIELDS),
    (workers.router, "/workers", workers.WORKER_FILTER_FIELDS),
    (workers.router, "/workers/cordons", workers.CORDON_FILTER_FIELDS),
    (nodes.router, "/nodes", nodes.NODE_FILTER_FIELDS),
    (nodes.router, "/nodes/workers", nodes.NODE_WORKER_FILTER_FIELDS),
    (nodes.router, "/nodes/{node_id}/workers", nodes.NODE_WORKER_FILTER_FIELDS),
    (stack.router, "/stack/workers", stack.STACK_WORKER_FILTER_FIELDS),
    (ssh.router, "/ssh/connections", ssh.SSH_CONNECTION_FILTER_FIELDS),
    (resident.router, "/resident/replicas", resident.RESIDENT_REPLICA_FILTER_FIELDS),
]


def _route(router: APIRouter, path: str) -> APIRoute:
    return next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == path and "GET" in route.methods
    )


def _model(annotation: Any) -> type[BaseModel] | None:
    """The one model an annotation holds, through Annotated and Optional."""
    if get_origin(annotation) is Annotated:
        return _model(get_args(annotation)[0])
    if get_origin(annotation) in (Union, types.UnionType):
        models = [m for arg in get_args(annotation) if (m := _model(arg)) is not None]
        return models[0] if len(models) == 1 else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _entry_model(route: APIRoute) -> type[BaseModel]:
    response = route.response_model
    if get_origin(response) is list:
        entry = _model(get_args(response)[0])
    else:
        page = _model(response)
        assert page is not None
        entry = _model(get_args(page.model_fields["entries"].annotation)[0])
    assert entry is not None
    return entry


def _unresolved(model: type[BaseModel], paths: Collection[str]) -> list[str]:
    """The paths that do not walk to a served, non-model field of ``model``."""
    missing = []
    for path in paths:
        current: type[BaseModel] | None = model
        annotation: Any = None
        for part in path.split("."):
            field = current.model_fields.get(part) if current is not None else None
            if field is None or field.exclude:
                missing.append(path)
                break
            annotation = field.annotation
            current = _model(annotation)
        else:
            if current is not None:
                missing.append(path)
    return missing


@pytest.mark.parametrize(
    "router, path, fields", _ROUTES, ids=[path for _, path, _ in _ROUTES]
)
def test_every_declared_filter_resolves_on_the_served_model(
    router: APIRouter, path: str, fields: frozenset[str]
) -> None:
    assert _unresolved(_entry_model(_route(router, path)), fields) == []


def test_a_field_the_model_lacks_does_not_resolve() -> None:
    model = _entry_model(_route(stack.router, "/stack/workers"))

    assert _unresolved(model, {"node_id", "hardware.cpu.model"}) == ["node_id"]


def test_every_task_filter_reads_under_the_runtime_lock() -> None:
    runtime = _live_runtime(FakeRegistry())
    asyncio.run(
        runtime.register(
            "owner",
            "org",
            "apiVersion: flowmesh/v1\nkind: EchoTask\nmetadata: {name: w}\n"
            "spec: {taskType: echo, data: {type: list, items: [x]}}\n",
            format="native",
        )
    )

    for key in tasks.TASK_FILTER_FIELDS:
        runtime.task_page(QueryFilter({key: frozenset({"x"})}), 10)
    assert tasks.TASK_FILTER_FIELDS - {"completed", "failed"} - {
        "depends_on",
        "pending_dependencies",
        "dependents",
    } <= set(TaskRecord.model_fields)


@pytest.mark.parametrize(
    "fields, key",
    [
        (workers.WORKER_FILTER_FIELDS, "capabilities.gpu_binding_task_types"),
        (nodes.NODE_WORKER_FILTER_FIELDS, "held_gpus"),
        (stack.STACK_WORKER_FILTER_FIELDS, "held_gpus"),
    ],
)
def test_worker_gpu_fields_are_filters(fields: frozenset[str], key: str) -> None:
    assert QueryFilter.parse(QueryParams({key: "1"}), fields) is not None


@pytest.mark.parametrize(
    "fields, key",
    [
        (tasks.TASK_FILTER_FIELDS, "task.spec.api.headers.Authorization"),
        (tasks.TASK_FILTER_FIELDS, "raw_yaml"),
        (tasks.TASK_FILTER_FIELDS, "latest_update.ssh.password"),
        (tasks.TASK_FILTER_FIELDS, "merge_key"),
        (tasks.TASK_FILTER_FIELDS, "credential_refs"),
        (tasks.TASK_FILTER_FIELDS, "failed_workers"),
        (tasks.TASK_FILTER_FIELDS, "submitted_ts"),
        (tasks.TASK_FILTER_FIELDS, "error"),
        (workers.WORKER_FILTER_FIELDS, "env.OPENAI_API_KEY"),
        (workers.WORKER_FILTER_FIELDS, "hardware.extra.token"),
        (resident.RESIDENT_REPLICA_FILTER_FIELDS, "endpoint.host"),
        (nodes.NODE_FILTER_FIELDS, "network_endpoint.host"),
    ],
)
def test_private_and_unmatchable_fields_are_not_filters(
    fields: frozenset[str], key: str
) -> None:
    with pytest.raises(InvalidQuery):
        QueryFilter.parse(QueryParams({key: "guess"}), fields)


class _DenyAll:
    name = "deny-all"

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="denied")

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return frozenset()


@pytest.fixture
def deny_all() -> Iterator[None]:
    PERMISSION_CHECKERS.append(_DenyAll())
    try:
        yield
    finally:
        PERMISSION_CHECKERS.clear()


_GATED_LISTINGS: list[
    tuple[Collection[str], Callable[[ListFilter], Coroutine[Any, Any, Any]]]
] = [
    (
        resident.RESIDENT_REPLICA_FILTER_FIELDS,
        lambda filters: resident.list_resident_replicas(
            principal=_PRINCIPAL, filters=filters, control=None, logger=_LOGGER
        ),
    ),
    (
        ssh.SSH_CONNECTION_FILTER_FIELDS,
        lambda filters: ssh.list_ssh_connections(
            principal=_PRINCIPAL, filters=filters, ssh_connections=None, logger=_LOGGER
        ),
    ),
    (
        stack.STACK_WORKER_FILTER_FIELDS,
        lambda filters: stack.list_workers(
            principal=_PRINCIPAL,
            filters=filters,
            supervisor=cast(Any, None),
            node_id="node-1",
            logger=_LOGGER,
        ),
    ),
    (
        nodes.NODE_WORKER_FILTER_FIELDS,
        lambda filters: nodes.list_node_workers(
            node_id="node-1",
            principal=_PRINCIPAL,
            filters=filters,
            node_registry=cast(Any, None),
            worker_registry=cast(Any, None),
            logger=_LOGGER,
        ),
    ),
]


@pytest.mark.parametrize(("fields", "listing"), _GATED_LISTINGS)
def test_a_gated_listing_authorizes_before_it_reads_filters(
    deny_all: None,
    fields: Collection[str],
    listing: Callable[[ListFilter], Coroutine[Any, Any, Any]],
) -> None:
    request = Request(
        {"type": "http", "method": "GET", "query_string": b"undeclared=1"}
    )
    filters = ListFilter(request, fields, ())
    with pytest.raises(HTTPException) as invalid:
        filters.parse()
    assert invalid.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as exc:
        asyncio.run(listing(filters))

    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.parametrize(("router", "path", "fields"), _ROUTES)
def test_a_list_routes_schema_declares_each_of_its_filters(
    router: APIRouter, path: str, fields: frozenset[str]
) -> None:
    app = FastAPI()
    app.include_router(router)
    operation = app.openapi()["paths"][path]["get"]
    query = {
        param["name"]: param
        for param in operation.get("parameters", [])
        if param["in"] == "query"
    }

    assert set(query) == fields | (PAGE_PARAMS if path in _PAGED else set())
    for field in fields:
        schema = query[field]["schema"]
        assert {"type": "array", "items": {"type": "string"}} in schema["anyOf"]
        assert "repeated key" in query[field]["description"]


def test_a_filter_dependency_reads_dotted_and_repeated_keys() -> None:
    app = FastAPI()
    app.include_router(nodes.router)
    seen: list[QueryFilter] = []

    async def _list_nodes_async() -> list[Any]:
        return []

    registry = types.SimpleNamespace(list_nodes_async=_list_nodes_async)
    app.dependency_overrides[nodes.authenticate_connection] = lambda: _PRINCIPAL
    app.dependency_overrides[nodes.get_node_registry] = lambda: registry
    app.dependency_overrides[nodes.get_worker_registry] = lambda: registry
    app.dependency_overrides[nodes.get_logger] = lambda: _LOGGER
    original = ListFilter.parse

    def _parse(self: ListFilter) -> QueryFilter:
        seen.append(query := original(self))
        return query

    client = TestClient(app)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ListFilter, "parse", _parse)
        ok = client.get(
            "/nodes/workers",
            params=[
                ("status", "ONLINE"),
                ("status", "DRAINING"),
                ("hardware.cpu.model", "x"),
            ],
        )
        refused = client.get("/nodes/workers", params={"undeclared": "1"})

    assert ok.status_code == status.HTTP_200_OK, ok.text
    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert seen and seen[0] == QueryFilter.parse(
        QueryParams("status=ONLINE&status=DRAINING&hardware.cpu.model=x"),
        nodes.NODE_WORKER_FILTER_FIELDS,
    )
