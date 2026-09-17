"""Trace endpoints — per-task upload, workflow-level read + analyzer, span queries."""

import logging
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from shared.schemas.result import result_file_path
from shared.telemetry.ids import workflow_to_trace_id_int
from shared.utils.json import encode_jsonl_bytes, read_jsonl

from ...app_state import (
    get_logger,
    get_results_dir,
    get_telemetry_store,
    get_workflow_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...governance import ProfileSummary, analyze
from ...hooks import ResourceAction, ResourceKind
from ...registries.workflow import WorkflowRegistry
from ...schemas.common import PathResponse
from ...schemas.traces import (
    TraceAggregate,
    TraceAggregateBucket,
    TraceSpanNode,
    TraceTree,
)
from ...telemetry.store import AggregateStat, MetricKind, SpanRow, TelemetryStore

router = APIRouter(prefix="/traces", tags=["Traces"])

_TYPE_TO_FILENAME: dict[str, str] = {
    "spans": "spans.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}


def _logs_dir_for_task(results_dir: Path, task_id: str) -> Path:
    """Per-task ``logs/`` directory holding the trace JSONL artifacts."""
    return result_file_path(results_dir, task_id).parent / "logs"


def _iter_workflow_jsonl(
    results_dir: Path, task_ids: Iterable[str], filename: str
) -> Iterator[dict[str, Any]]:
    for task_id in task_ids:
        yield from read_jsonl(_logs_dir_for_task(results_dir, task_id) / filename)


async def _resolve_task_ids(workflow_id: str, registry: WorkflowRegistry) -> list[str]:
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return workflow.task_ids


@router.get(
    "/workflows/analyze/{workflow_id}",
    summary="Run the trace analyzer; return ProfileSummary",
    response_model=ProfileSummary,
)
async def analyze_workflow_trace(
    workflow_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> ProfileSummary:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    spans = list(_iter_workflow_jsonl(results_dir, task_ids, "spans.jsonl"))
    assets = list(_iter_workflow_jsonl(results_dir, task_ids, "assets.jsonl"))
    lineage = list(_iter_workflow_jsonl(results_dir, task_ids, "lineage.jsonl"))
    return analyze(spans, assets, lineage, workflow_id=workflow_id)


@router.get(
    "/workflows/{workflow_id}/{trace_type}",
    summary="Stream JSONL rows (spans / assets / lineage)",
)
async def get_workflow_trace(
    workflow_id: str,
    trace_type: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    return StreamingResponse(
        encode_jsonl_bytes(_iter_workflow_jsonl(results_dir, task_ids, filename)),
        media_type="application/x-ndjson",
    )


@router.post(
    "/tasks/{task_id}/{trace_type}",
    summary="Upload a per-task trace JSONL file (spans / assets / lineage)",
)
async def upload_task_trace(
    task_id: str,
    trace_type: str,
    file: UploadFile = File(...),
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    target_path = _logs_dir_for_task(results_dir, task_id) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("wb") as out:
            out.write(await file.read())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store trace: {exc}",
        ) from exc
    return PathResponse(ok=True, path=target_path.as_posix())


def _require_telemetry_store(store: TelemetryStore | None) -> TelemetryStore:
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Telemetry querying is not configured on this deployment; "
                "configure the telemetry store to query spans and metrics."
            ),
        )
    return store


def _node_sort_key(node: TraceSpanNode) -> tuple[datetime, str]:
    return (node.start_time, node.span_id)


def _to_node(row: SpanRow) -> TraceSpanNode:
    return TraceSpanNode(
        span_id=row.span_id,
        parent_span_id=row.parent_span_id,
        name=row.name,
        start_time=row.start_time,
        end_time=row.end_time,
        duration_seconds=row.duration_ns / 1e9,
        status=row.status_code,
        logical=dict(row.logical),
        physical=dict(row.physical),
    )


def build_span_tree(workflow_id: str, rows: list[SpanRow]) -> TraceTree:
    """Assemble a workflow's flat span rows into their parent/child hierarchy.

    A span whose parent is absent from the row set becomes a root, so a trace that is
    still exporting renders every span it has rather than dropping a subtree. Siblings
    are ordered by start time, then span id, so repeated calls render identically.
    """
    if rows:
        trace_id = rows[0].trace_id
    else:
        trace_id = format(workflow_to_trace_id_int(workflow_id), "032x")

    nodes = {row.span_id: _to_node(row) for row in rows}
    roots: list[TraceSpanNode] = []
    for node in sorted(nodes.values(), key=_node_sort_key):
        parent = nodes.get(node.parent_span_id) if node.parent_span_id else None
        if parent is None or parent is node:
            roots.append(node)
        else:
            parent.children.append(node)
    _reroot_unreachable(roots, nodes)

    total_duration = 0.0
    if rows:
        earliest = min(row.start_time for row in rows)
        latest = max(row.end_time for row in rows)
        total_duration = max((latest - earliest).total_seconds(), 0.0)

    return TraceTree(
        workflow_id=workflow_id,
        trace_id=trace_id,
        span_count=len(nodes),
        total_duration_seconds=total_duration,
        roots=roots,
    )


def _reroot_unreachable(
    roots: list[TraceSpanNode], nodes: dict[str, TraceSpanNode]
) -> None:
    """Promote to a root the earliest span of any component no root reaches.

    A malformed parent chain can form a cycle, whose spans hang off each other and off
    no root. Re-rooting each such component keeps every span reachable exactly once and
    bounds the render walk.
    """
    reached: set[str] = set()

    def _visit(node: TraceSpanNode) -> None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current.span_id in reached:
                continue
            reached.add(current.span_id)
            stack.extend(current.children)

    for root in roots:
        _visit(root)

    while len(reached) < len(nodes):
        orphan = min(
            (node for span_id, node in nodes.items() if span_id not in reached),
            key=_node_sort_key,
        )
        parent = nodes.get(orphan.parent_span_id) if orphan.parent_span_id else None
        if parent is not None:
            parent.children.remove(orphan)
        roots.append(orphan)
        _visit(orphan)


@router.get(
    "/workflows/{workflow_id}/spans/tree",
    summary="Assemble a workflow's spans into their parent/child hierarchy",
    response_model=TraceTree,
)
async def get_workflow_span_tree(
    workflow_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    store: TelemetryStore | None = Depends(get_telemetry_store),
    logger: logging.Logger = Depends(get_logger),
) -> TraceTree:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    rows = _require_telemetry_store(store).fetch_trace(workflow_id)
    return build_span_tree(workflow_id, rows)


@router.get(
    "/aggregate",
    summary="Aggregate one telemetry metric, grouped by one attribute",
    response_model=TraceAggregate,
)
async def aggregate_metric(
    metric: str = Query(description="Metric name to aggregate."),
    group_by: str = Query(description="Attribute key to group the metric by."),
    stat: AggregateStat = Query(
        default="avg", description="Statistic applied to the metric's datapoints."
    ),
    kind: MetricKind = Query(
        default="gauge", description="Metric kind selecting the store's metric table."
    ),
    workflow_id: str | None = Query(
        default=None, description="Restrict the aggregate to one workflow."
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    store: TelemetryStore | None = Depends(get_telemetry_store),
    logger: logging.Logger = Depends(get_logger),
) -> TraceAggregate:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    buckets = _require_telemetry_store(store).aggregate(
        metric=metric,
        group_by=group_by,
        stat=stat,
        kind=kind,
        workflow_id=workflow_id,
    )
    return TraceAggregate(
        metric=metric,
        group_by=group_by,
        stat=stat,
        workflow_id=workflow_id,
        buckets=[
            TraceAggregateBucket(
                group_value=bucket.group_value,
                stat=bucket.stat,
                value=bucket.value,
                sample_count=bucket.sample_count,
            )
            for bucket in buckets
        ],
    )
