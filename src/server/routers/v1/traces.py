"""Trace endpoints — per-task upload, workflow-level read + analyzer, span queries."""

import functools
import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import TypeAdapter
from starlette.types import Receive, Scope, Send

from shared.schemas.result import result_file_path
from shared.telemetry.ids import workflow_to_trace_id_int
from shared.utils.atomic import atomic_write_stream
from shared.utils.json import encode_jsonl_bytes, parse_jsonl_lines
from shared.utils.manifest import LOGS_DIR
from shared.utils.nofollow import PathRefused, open_below, open_dir

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
from ...registries.workflow import Workflow, WorkflowRegistry
from ...schemas.common import PathResponse
from ...schemas.traces import (
    TraceAggregate,
    TraceAggregateBucket,
    TraceSpanNode,
    TraceTree,
)
from ...telemetry.store import (
    AggregateStat,
    MetricKind,
    SpanRow,
    TelemetryStore,
    TelemetryStoreError,
    UnsupportedAggregateError,
)

router = APIRouter(prefix="/traces", tags=["Traces"])

_TYPE_TO_FILENAME: dict[str, str] = {
    "spans": "spans.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}


def _task_dir(results_dir: Path, task_id: str) -> Path:
    return result_file_path(results_dir, task_id).parent


# The longest trace line read; a longer one is skipped rather than held in memory.
_MAX_TRACE_LINE_BYTES = 4 << 20


class _WorkflowRows:
    """The rows of one trace file across a workflow's tasks, read one file and one
    bounded line at a time; ``close`` closes the file being read, so an aborted
    stream holds none."""

    def __init__(
        self,
        results_dir: Path,
        task_ids: Iterable[str],
        filename: str,
        logger: logging.Logger,
    ):
        self._results_dir = results_dir
        self._task_ids = task_ids
        self._filename = filename
        self._logger = logger
        self._current: BinaryIO | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for task_id in self._task_ids:
            opened = open_below(
                _task_dir(self._results_dir, task_id),
                PurePosixPath(LOGS_DIR, self._filename),
            )
            if opened is None:
                continue
            with opened as fh:
                self._current = fh
                yield from parse_jsonl_lines(self._lines(fh, task_id))
            self._current = None

    def _lines(self, fh: BinaryIO, task_id: str) -> Iterator[str]:
        while line := fh.readline(_MAX_TRACE_LINE_BYTES + 1):
            if len(line) > _MAX_TRACE_LINE_BYTES and not line.endswith(b"\n"):
                self._logger.warning(
                    "Skipping a %s line of task %s longer than %d bytes",
                    self._filename,
                    task_id,
                    _MAX_TRACE_LINE_BYTES,
                )
                while line and not line.endswith(b"\n"):
                    line = fh.readline(_MAX_TRACE_LINE_BYTES)
                continue
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._current is not None:
            self._current.close()


class _ClosingStreamingResponse(StreamingResponse):
    """A streaming response that runs ``on_close`` once it ends, sent or aborted."""

    def __init__(
        self, content: Iterator[bytes], on_close: Callable[[], None], media_type: str
    ) -> None:
        super().__init__(content, media_type=media_type)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_close()


async def _resolve_workflow(workflow_id: str, registry: WorkflowRegistry) -> Workflow:
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return workflow


async def _resolve_task_ids(workflow_id: str, registry: WorkflowRegistry) -> list[str]:
    return (await _resolve_workflow(workflow_id, registry)).task_ids


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
) -> Response:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    body = await run_in_threadpool(
        _analyze_workflow, results_dir, task_ids, workflow_id, logger
    )
    return Response(body, media_type="application/json")


_PROFILE_SUMMARY = TypeAdapter(ProfileSummary)


def _analyze_workflow(
    results_dir: Path, task_ids: list[str], workflow_id: str, logger: logging.Logger
) -> bytes:
    spans = list(_WorkflowRows(results_dir, task_ids, "spans.jsonl", logger))
    assets = list(_WorkflowRows(results_dir, task_ids, "assets.jsonl", logger))
    lineage = list(_WorkflowRows(results_dir, task_ids, "lineage.jsonl", logger))
    summary = analyze(spans, assets, lineage, workflow_id=workflow_id)
    return _PROFILE_SUMMARY.dump_json(summary, by_alias=True)


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
    rows = _WorkflowRows(results_dir, task_ids, filename, logger)
    return _ClosingStreamingResponse(
        encode_jsonl_bytes(rows), rows.close, media_type="application/x-ndjson"
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
    task_dir = _task_dir(results_dir, task_id)
    try:
        await run_in_threadpool(_store_trace, task_dir, filename, file.file)
    except PathRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid path"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store trace: {exc}",
        ) from exc
    return PathResponse(ok=True, path=(task_dir / LOGS_DIR / filename).as_posix())


def _store_trace(task_dir: Path, filename: str, source: BinaryIO) -> None:
    with open_dir(task_dir, LOGS_DIR, create=True) as logs_fd:
        atomic_write_stream(Path(filename), source, dir_fd=logs_fd)


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


@contextmanager
def _store_available() -> Iterator[None]:
    """Answer a configured-but-unreachable store with guidance, not a traceback.

    A store the deployment configured and a store it can currently reach are different
    failures, and the second one is the one an operator can act on -- most often the
    store was never deployed at all.
    """
    try:
        yield
    except UnsupportedAggregateError as exc:
        # The store answered: what was asked for is not derivable from what the metric
        # carries, so it is the request that is wrong, not the deployment.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except TelemetryStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"The configured telemetry store could not be queried: {exc}. "
                "Check that the telemetry services are deployed and reachable."
            ),
        ) from exc


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
        service_name=row.service_name,
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
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    store: TelemetryStore | None = Depends(get_telemetry_store),
    logger: logging.Logger = Depends(get_logger),
) -> TraceTree:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    # The trace id is derived from the workflow id by a lossy mapping, so many strings
    # address one trace: only an id the registry knows may be read through it.
    await _resolve_workflow(workflow_id, registry)
    with _store_available():
        rows = await run_in_threadpool(
            _require_telemetry_store(store).fetch_trace, workflow_id
        )
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
    principal: PrincipalContext = Depends(authenticate_connection),
    store: TelemetryStore | None = Depends(get_telemetry_store),
    logger: logging.Logger = Depends(get_logger),
) -> TraceAggregate:
    # The answer spans every tenant's workflows, so it is gated like the fleet-wide
    # system metrics snapshot rather than as a read of one workflow.
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )
    with _store_available():
        buckets = await run_in_threadpool(
            functools.partial(
                _require_telemetry_store(store).aggregate,
                metric=metric,
                group_by=group_by,
                stat=stat,
                kind=kind,
            )
        )
    return TraceAggregate(
        metric=metric,
        group_by=group_by,
        stat=stat,
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
