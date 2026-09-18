"""Span-tree assembly and the two telemetry query routes.

The store is a fake implementing the read protocol -- no ClickHouse adapter is imported
here, so what these assert is the tree the route builds from a flat, unordered row set,
not any one backend's query.
"""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from fastapi import HTTPException, status
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.registries.workflow import Workflow, WorkflowRegistry, WorkflowStatus
from server.routers.v1.traces import (
    aggregate_metric,
    build_span_tree,
    get_workflow_span_tree,
)
from server.telemetry.store import (
    AggregateBucket,
    AggregateStat,
    MetricKind,
    SpanRow,
    TelemetryStore,
)
from shared.telemetry.ids import workflow_to_trace_id_int

WORKFLOW_ID = "wfl-0102030405060708090a0b0c0d0e0f10"
TRACE_ID = format(workflow_to_trace_id_int(WORKFLOW_ID), "032x")
# A different string the trace-id derivation maps onto the same trace: the mapping
# strips every non-hex character, so many ids address one trace.
TRACE_ID_ALIAS = f"{WORKFLOW_ID}-zz"
BASE = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-1",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.span_tree")


class FakeStore:
    """A ``TelemetryStore`` whose two queries return canned rows."""

    def __init__(
        self,
        rows: list[SpanRow] | None = None,
        buckets: list[AggregateBucket] | None = None,
    ) -> None:
        self.rows = rows or []
        self.buckets = buckets or []
        self.aggregate_calls: list[dict[str, object]] = []
        self.fetched: list[str] = []

    def fetch_trace(self, workflow_id: str) -> list[SpanRow]:
        self.fetched.append(workflow_id)
        return list(self.rows)

    def aggregate(
        self,
        *,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
    ) -> list[AggregateBucket]:
        self.aggregate_calls.append(
            {"metric": metric, "group_by": group_by, "stat": stat, "kind": kind}
        )
        return list(self.buckets)


def _as_port(store: FakeStore) -> TelemetryStore:
    """Type-checked proof the fake satisfies the read port the routes depend on."""
    return store


def _workflow(workflow_id: str) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        task_ids=["tsk-1"],
        submitted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        status=WorkflowStatus.DONE,
        dispatched_tasks=[],
        completed_tasks=["tsk-1"],
        failed_tasks=[],
        cancelled_tasks=[],
    )


class FakeRegistry:
    """A registry that knows exactly the workflow ids it was seeded with."""

    def __init__(self, *workflow_ids: str) -> None:
        self.known = set(workflow_ids)

    async def get_workflow_async(self, workflow_id: str) -> Workflow | None:
        return _workflow(workflow_id) if workflow_id in self.known else None


def _as_registry(registry: FakeRegistry) -> WorkflowRegistry:
    return cast(WorkflowRegistry, registry)


def _row(
    span_id: str,
    parent_span_id: str | None,
    *,
    offset_sec: float = 0.0,
    duration_sec: float = 1.0,
    name: str | None = None,
    logical: dict[str, str] | None = None,
    physical: dict[str, str] | None = None,
    service_name: str = "flowmesh-server",
) -> SpanRow:
    start = BASE + timedelta(seconds=offset_sec)
    return SpanRow(
        trace_id=TRACE_ID,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name or f"span-{span_id}",
        kind="SPAN_KIND_INTERNAL",
        start_time=start,
        end_time=start + timedelta(seconds=duration_sec),
        duration_ns=int(duration_sec * 1e9),
        status_code="Unset",
        service_name=service_name,
        logical=logical or {},
        physical=physical or {},
    )


def _ids(nodes) -> list[str]:
    return [node.span_id for node in nodes]


def test_nests_children_under_their_parent() -> None:
    rows = [
        _row("c2", "r1", offset_sec=2),
        _row("r1", None, offset_sec=0),
        _row("c1", "r1", offset_sec=1),
        _row("g1", "c1", offset_sec=1.5),
    ]

    tree = build_span_tree(WORKFLOW_ID, rows)

    assert _ids(tree.roots) == ["r1"]
    assert _ids(tree.roots[0].children) == ["c1", "c2"]
    assert _ids(tree.roots[0].children[0].children) == ["g1"]
    assert tree.span_count == 4
    assert tree.trace_id == TRACE_ID


def test_orders_siblings_by_start_time_regardless_of_row_order() -> None:
    forward = [
        _row("r", None),
        _row("a", "r", offset_sec=3),
        _row("b", "r", offset_sec=1),
        _row("c", "r", offset_sec=2),
    ]
    reversed_rows = list(reversed(forward))

    assert _ids(build_span_tree(WORKFLOW_ID, forward).roots[0].children) == [
        "b",
        "c",
        "a",
    ]
    assert _ids(build_span_tree(WORKFLOW_ID, reversed_rows).roots[0].children) == [
        "b",
        "c",
        "a",
    ]


def test_breaks_a_start_time_tie_on_span_id() -> None:
    rows = [
        _row("r", None),
        _row("bb", "r", offset_sec=1),
        _row("aa", "r", offset_sec=1),
    ]

    assert _ids(build_span_tree(WORKFLOW_ID, rows).roots[0].children) == ["aa", "bb"]


def test_promotes_a_span_whose_parent_is_absent_to_a_root() -> None:
    rows = [_row("r", None), _row("stray", "not-in-this-set", offset_sec=5)]

    tree = build_span_tree(WORKFLOW_ID, rows)

    assert _ids(tree.roots) == ["r", "stray"]
    assert tree.span_count == 2


def test_every_span_appears_exactly_once_under_a_parent_cycle() -> None:
    rows = [
        _row("a", "c", offset_sec=1),
        _row("b", "a", offset_sec=2),
        _row("c", "b", offset_sec=3),
    ]

    tree = build_span_tree(WORKFLOW_ID, rows)

    seen: list[str] = []
    stack = list(tree.roots)
    while stack:
        node = stack.pop()
        seen.append(node.span_id)
        stack.extend(node.children)
    assert sorted(seen) == ["a", "b", "c"]
    assert tree.span_count == 3


def test_keeps_the_logical_and_physical_views_separate() -> None:
    rows = [
        _row(
            "r",
            None,
            logical={"workflow_id": WORKFLOW_ID, "activation_id": "act-1"},
            physical={"worker_id": "wkr-1", "attempt_id": "att-1"},
        )
    ]

    node = build_span_tree(WORKFLOW_ID, rows).roots[0]

    assert node.logical == {"workflow_id": WORKFLOW_ID, "activation_id": "act-1"}
    assert node.physical == {"worker_id": "wkr-1", "attempt_id": "att-1"}


def test_spans_total_duration_from_earliest_start_to_latest_end() -> None:
    rows = [
        _row("r", None, offset_sec=0, duration_sec=10),
        _row("c", "r", offset_sec=2, duration_sec=3),
    ]

    assert build_span_tree(WORKFLOW_ID, rows).total_duration_seconds == pytest.approx(
        10.0
    )


def test_an_untraced_workflow_yields_an_empty_tree_at_the_derived_trace_id() -> None:
    tree = build_span_tree(WORKFLOW_ID, [])

    assert tree.roots == []
    assert tree.span_count == 0
    assert tree.total_duration_seconds == 0.0
    assert tree.trace_id == TRACE_ID


@pytest.mark.asyncio
async def test_tree_route_returns_the_assembled_tree(principal, logger) -> None:
    store = _as_port(FakeStore([_row("r", None), _row("c", "r", offset_sec=1)]))

    tree = await get_workflow_span_tree(
        WORKFLOW_ID,
        principal=principal,
        registry=_as_registry(FakeRegistry(WORKFLOW_ID)),
        store=store,
        logger=logger,
    )

    assert _ids(tree.roots) == ["r"]
    assert _ids(tree.roots[0].children) == ["c"]


@pytest.mark.asyncio
async def test_tree_route_refuses_an_id_the_registry_does_not_know(
    principal, logger
) -> None:
    """An alias of a real workflow id derives the same trace and must not read it."""
    assert workflow_to_trace_id_int(TRACE_ID_ALIAS) == workflow_to_trace_id_int(
        WORKFLOW_ID
    )
    fake = FakeStore([_row("r", None)])

    with pytest.raises(HTTPException) as excinfo:
        await get_workflow_span_tree(
            TRACE_ID_ALIAS,
            principal=principal,
            registry=_as_registry(FakeRegistry(WORKFLOW_ID)),
            store=_as_port(fake),
            logger=logger,
        )

    assert excinfo.value.status_code == status.HTTP_404_NOT_FOUND
    assert fake.fetched == []


@pytest.mark.asyncio
async def test_aggregate_route_passes_every_argument_to_the_store(
    principal, logger
) -> None:
    fake = FakeStore(buckets=[AggregateBucket("wkr-1", "p95", 12.5, 40)])
    store = _as_port(fake)

    result = await aggregate_metric(
        metric="flowmesh.control.stage.duration",
        group_by="worker_id",
        stat="p95",
        kind="histogram",
        principal=principal,
        store=store,
        logger=logger,
    )

    assert fake.aggregate_calls == [
        {
            "metric": "flowmesh.control.stage.duration",
            "group_by": "worker_id",
            "stat": "p95",
            "kind": "histogram",
        }
    ]
    assert result.metric == "flowmesh.control.stage.duration"
    assert result.stat == "p95"
    assert [bucket.group_value for bucket in result.buckets] == ["wkr-1"]
    assert result.buckets[0].sample_count == 40


class _RecordingChecker:
    """Records what it was asked, and gates a SYSTEM resource on an admin principal."""

    name = "recording"

    def __init__(self) -> None:
        self.asked: list[tuple[str, str | None, str]] = []

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        self.asked.append((resource.kind, resource.id, action))
        if resource.kind == "system" and principal.principal_type != "admin":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")

    async def accessible_ids(self, *args: Any, **kwargs: Any) -> frozenset[str] | None:
        return None


@pytest.fixture
def recording_checker() -> Iterator[_RecordingChecker]:
    checker = _RecordingChecker()
    PERMISSION_CHECKERS.append(checker)
    try:
        yield checker
    finally:
        PERMISSION_CHECKERS.remove(checker)


@pytest.mark.asyncio
async def test_a_fleet_wide_aggregate_is_gated_on_a_system_admin_right(
    principal, logger, recording_checker
) -> None:
    """The answer spans every tenant, so a tenant-level workflow read cannot buy it."""
    fake = FakeStore(buckets=[AggregateBucket("wkr-1", "avg", 1.0, 2)])

    with pytest.raises(HTTPException) as excinfo:
        await aggregate_metric(
            metric="flowmesh.gpu.utilization",
            group_by="worker_id",
            stat="avg",
            kind="gauge",
            principal=principal,
            store=_as_port(fake),
            logger=logger,
        )

    assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
    assert recording_checker.asked == [("system", None, "admin")]
    assert fake.aggregate_calls == []


@pytest.mark.asyncio
async def test_a_workflow_span_tree_stays_a_workflow_level_read(
    principal, logger, recording_checker
) -> None:
    await get_workflow_span_tree(
        WORKFLOW_ID,
        principal=principal,
        registry=_as_registry(FakeRegistry(WORKFLOW_ID)),
        store=_as_port(FakeStore([_row("r", None)])),
        logger=logger,
    )

    assert recording_checker.asked == [("workflow", WORKFLOW_ID, "read")]


@pytest.mark.asyncio
async def test_tree_route_reports_an_unconfigured_store(principal, logger) -> None:
    with pytest.raises(HTTPException) as excinfo:
        await get_workflow_span_tree(
            WORKFLOW_ID,
            principal=principal,
            registry=_as_registry(FakeRegistry(WORKFLOW_ID)),
            store=None,
            logger=logger,
        )

    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "not configured" in excinfo.value.detail


@pytest.mark.asyncio
async def test_aggregate_route_reports_an_unconfigured_store(principal, logger) -> None:
    with pytest.raises(HTTPException) as excinfo:
        await aggregate_metric(
            metric="m",
            group_by="worker_id",
            stat="avg",
            kind="gauge",
            principal=principal,
            store=None,
            logger=logger,
        )

    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_each_node_names_the_service_that_emitted_it() -> None:
    """A trace that spans processes is only readable as one if a reader can tell
    which process each span came from, so the value must survive to the node."""
    rows = [
        _row("aa", None, service_name="flowmesh-server"),
        _row("bb", "aa", offset_sec=1, service_name="flowmesh-worker"),
    ]

    tree = build_span_tree(WORKFLOW_ID, rows)

    root = tree.roots[0]
    assert root.service_name == "flowmesh-server"
    assert [child.service_name for child in root.children] == ["flowmesh-worker"]
