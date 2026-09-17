"""Span-tree assembly and the two telemetry query routes.

The store is a fake implementing the read protocol -- no ClickHouse adapter is imported
here, so what these assert is the tree the route builds from a flat, unordered row set,
not any one backend's query.
"""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException, status
from lumid_hooks import PrincipalContext

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

    def fetch_trace(self, workflow_id: str) -> list[SpanRow]:
        return list(self.rows)

    def aggregate(
        self,
        *,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
        workflow_id: str | None = None,
    ) -> list[AggregateBucket]:
        self.aggregate_calls.append(
            {
                "metric": metric,
                "group_by": group_by,
                "stat": stat,
                "kind": kind,
                "workflow_id": workflow_id,
            }
        )
        return list(self.buckets)


def _as_port(store: FakeStore) -> TelemetryStore:
    """Type-checked proof the fake satisfies the read port the routes depend on."""
    return store


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
        WORKFLOW_ID, principal=principal, store=store, logger=logger
    )

    assert _ids(tree.roots) == ["r"]
    assert _ids(tree.roots[0].children) == ["c"]


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
        workflow_id=WORKFLOW_ID,
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
            "workflow_id": WORKFLOW_ID,
        }
    ]
    assert result.metric == "flowmesh.control.stage.duration"
    assert result.stat == "p95"
    assert [bucket.group_value for bucket in result.buckets] == ["wkr-1"]
    assert result.buckets[0].sample_count == 40


@pytest.mark.asyncio
async def test_tree_route_reports_an_unconfigured_store(principal, logger) -> None:
    with pytest.raises(HTTPException) as excinfo:
        await get_workflow_span_tree(
            WORKFLOW_ID, principal=principal, store=None, logger=logger
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
            workflow_id=None,
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
