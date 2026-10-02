import asyncio
import json
import logging
from collections.abc import Collection
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi import Path as ApiPath
from fastapi import Query, Request, status
from fastapi.responses import Response, StreamingResponse

from shared.schemas.command import StopMessage
from shared.tasks import TaskType

from ...app_state import (
    get_logger,
    get_redis_client,
    get_runtime,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
    resolve_accessible_ids,
)
from ...clients.redis import RedisClient, task_log_closed_key, task_log_stream_key
from ...hooks import ResourceAction, ResourceKind
from ...registries.worker import WorkerRegistry
from ...schemas.common import OkResponse
from ...schemas.logs import LogEntry, LogEvent, LogQueryResponse
from ...schemas.tasks import TaskPage
from ...task.models import TaskOrder
from ...task.runtime import TaskInfo, TaskRuntime
from ...utils.cursors import InvalidCursor, decode_cursor, encode_cursor
from ...utils.query import QueryFilter
from ._listing import (
    PAGE_LIMIT_DEFAULT,
    PageAfter,
    PageBefore,
    PageLimit,
    page_bounds,
    query_filter,
)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _strip_private_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if not key.startswith("_")}


def _sanitize_latest_update(info: TaskInfo) -> None:
    """Remove private latest_update fields from the public task response."""
    if not isinstance(info.latest_update, dict):
        return
    latest_update = _strip_private_fields(info.latest_update)
    ssh_info = latest_update.get("ssh")
    if isinstance(ssh_info, dict):
        latest_update["ssh"] = _strip_private_fields(ssh_info)
    serve_info = latest_update.get("serve")
    if isinstance(serve_info, dict):
        latest_update["serve"] = _strip_private_fields(serve_info)
    info.latest_update = latest_update


TASK_FILTER_FIELDS = frozenset(
    {
        "task_id",
        "workflow_id",
        "status",
        "category",
        "task_type",
        "assigned_worker",
        "graph_node_name",
        "completed",
        "failed",
    }
)


@router.get(
    "",
    summary="List tasks",
    description=(
        "List tasks ordered by submission. Without a cursor, returns the newest page."
    ),
    response_description="A page of task details.",
    response_model=TaskPage,
)
async def list_tasks(
    request: Request,
    limit: PageLimit = PAGE_LIMIT_DEFAULT,
    before: PageBefore = None,
    after: PageAfter = None,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    query = query_filter(request, TASK_FILTER_FIELDS)
    after_bound, before_bound = page_bounds(after, before, _decode_task_cursor)
    allowed = await resolve_accessible_ids(
        principal, ResourceKind.TASK, ResourceAction.READ, logger
    )
    body = await asyncio.to_thread(
        _task_page_json, runtime, query, limit, after_bound, before_bound, allowed
    )
    return Response(body, media_type="application/json")


def _task_page_json(
    runtime: TaskRuntime,
    query: QueryFilter,
    limit: int,
    after: TaskOrder | None,
    before: TaskOrder | None,
    accessible: Collection[str] | None,
) -> bytes:
    tasks = runtime.task_page(query, limit, after, before, accessible)
    for task in tasks:
        _sanitize_latest_update(task)
    page = TaskPage(
        entries=tasks,
        next_cursor=_task_cursor(tasks[-1]) if tasks else None,
        prev_cursor=_task_cursor(tasks[0]) if tasks else None,
    )
    return page.model_dump_json(by_alias=True).encode()


def _task_cursor(task: TaskInfo) -> str:
    return encode_cursor([task.submitted_ts, task.task_id])


def _decode_task_cursor(cursor: str) -> TaskOrder:
    identity = decode_cursor(cursor)
    match identity:
        case [int() | float() as ts, str() as task_id] if not isinstance(ts, bool):
            return float(ts), task_id
    raise InvalidCursor(f"invalid cursor {cursor!r}")


@router.get(
    "/{task_id}",
    summary="Get a task",
    description="Get task details by ID.",
    response_description="Task details.",
)
async def get_task(
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> TaskInfo:
    await require_permission(
        principal, ResourceKind.TASK, task_id, ResourceAction.READ, logger
    )
    info = runtime.describe_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    _sanitize_latest_update(info)
    return info


@router.post(
    "/{task_id}/stop",
    summary="Stop a running task",
    description=(
        "Stop a running task. The server sends a stop command to the assigned worker, "
        "but cannot guarantee the task will stop successfully."
    ),
    response_description="Operation result.",
)
async def stop_task(
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> OkResponse:
    await require_permission(
        principal, ResourceKind.TASK, task_id, ResourceAction.CANCEL, logger
    )
    record = runtime.get_record(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
        )
    if record.task.spec.taskType not in (TaskType.SSH, TaskType.SERVE):
        # TODO: Support stopping other task types.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stopping is only supported for SSH and SERVE tasks currently",
        )
    if record.status != "DISPATCHED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task is not running",
        )
    worker_id = record.assigned_worker
    if not worker_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task has no assigned worker",
        )
    worker = await worker_registry.get_worker_async(worker_id)
    if worker is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Assigned worker not found",
        )
    await worker_registry.publish_stop_async(
        worker,
        StopMessage(
            task_id=task_id, worker_id=worker.id, dispatch_id=record.dispatch_id
        ),
    )
    return OkResponse(ok=True)


@router.get(
    "/{task_id}/logs",
    summary="Query task logs",
    description="Read recent task logs.",
    response_description="Task log entries.",
    response_model=LogQueryResponse,
)
async def get_task_logs(
    task_id: str = ApiPath(..., min_length=1),
    limit: int = Query(
        default=200,
        ge=1,
        le=10_000,
        description="Maximum number of log entries to return.",
    ),
    before: str | None = Query(
        default=None,
        description=(
            "Return entries strictly before this cursor. The cursor is an opaque "
            "string previously returned as `entries[].cursor` "
            '(example: `"1707349300000-0"`).'
        ),
    ),
    after: str | None = Query(
        default=None,
        description=(
            "Return entries strictly after this cursor. The cursor is an opaque string "
            "previously returned as `entries[].cursor` "
            '(example: `"1707349300000-0"`).'
        ),
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
) -> LogQueryResponse:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    limit = max(1, min(10_000, int(limit)))
    if before and after:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of before/after may be set",
        )

    key = task_log_stream_key(task_id)
    if not await redis.asyncio.exists_telemetry(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="log stream not found"
        )
    if after:
        raw = await redis.asyncio.xrange_telemetry(key, min_id=f"({after}", count=limit)
        ordered = raw
    else:
        max_id = f"({before}" if before else "+"
        raw = await redis.asyncio.xrevrange_telemetry(
            key, max_id=max_id, min_id="-", count=limit
        )
        ordered = list(reversed(raw))

    entries: list[LogEntry] = []
    for cursor, fields in ordered:
        payload = fields.get("payload")
        if not isinstance(payload, str) or not payload:
            continue
        workflow_id = fields.get("workflow_id", "")
        task_id_field = fields.get("task_id", "")
        event: dict[str, Any] = json.loads(payload)
        if workflow_id:
            event.setdefault("workflow_id", workflow_id)
        event.setdefault("task_id", task_id_field or task_id)
        event.setdefault("level", "INFO")
        event.setdefault("stream", "system")
        if not event.get("message"):
            event["message"] = payload
        entries.append(LogEntry(cursor=cursor, event=LogEvent.model_validate(event)))

    next_cursor = entries[-1].cursor if entries else None
    prev_cursor = entries[0].cursor if entries else None
    return LogQueryResponse(
        entries=entries, next_cursor=next_cursor, prev_cursor=prev_cursor
    )


@router.get(
    "/{task_id}/logs/stream",
    summary="Stream task logs",
    description=(
        "Stream task logs via SSE.\n\n"
        "Cursor formats:\n"
        "- `cursor` and `Last-Event-ID` are opaque stream IDs (example: "
        '`"1707349300000-0"`).\n\n'
        "Where to get cursors:\n"
        "- From SSE: each message includes an `id` field; persist the last seen `id`.\n"
        "- From query API: `GET /tasks/{task_id}/logs` returns `entries[].cursor`.\n\n"
        "End of stream:\n"
        "- When the server detects the stream has ended, it sends an `eos` event and "
        "closes the connection.\n\n"
        "Reconnection:\n"
        "- Prefer setting the standard SSE header `Last-Event-ID` on reconnect.\n"
        "- If both `cursor` and `Last-Event-ID` are set, `Last-Event-ID` takes "
        "precedence."
    ),
    response_class=StreamingResponse,
)
async def stream_task_logs(
    task_id: str = ApiPath(..., min_length=1),
    cursor: str | None = Query(
        default=None,
        description=(
            "Resume streaming strictly after this cursor. Use the last seen "
            "`entries[].cursor` from the query endpoint, or the last SSE `id` value "
            "you received."
        ),
    ),
    last_event_id: str | None = Header(
        default=None,
        alias="Last-Event-ID",
        description=(
            "SSE reconnection cursor. If set, the stream resumes strictly after this "
            "ID. When both `cursor` and `Last-Event-ID` are provided, `Last-Event-ID` "
            "takes precedence."
        ),
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
):
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    key = task_log_stream_key(task_id)
    if not await redis.asyncio.exists_telemetry(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="log stream not found"
        )
    start_id = last_event_id or cursor or "$"

    async def _gen():
        current = start_id
        try:
            # Check if the stream is already closed
            if await redis.asyncio.exists(task_log_closed_key(task_id)):
                if current == "$":
                    yield b"event: eos\ndata:\n\n"
                    return
                normalized = current.lstrip("(").strip()
                if normalized:
                    newer = await redis.asyncio.xrange_telemetry(
                        key, min_id=f"({normalized}", max_id="+", count=1
                    )
                    if not newer:
                        yield b"event: eos\ndata:\n\n"
                        return

            while True:
                rows = await redis.asyncio.xread_telemetry(
                    {key: current}, count=200, block_ms=15_000
                )
                if not rows:
                    yield b": keep-alive\n\n"
                    continue
                for _, batch in rows:
                    for stream_id, fields in batch:
                        payload = fields.get("payload")
                        if not isinstance(payload, str) or not payload:
                            current = stream_id
                            continue
                        msg = f"id: {stream_id}\nevent: log\ndata: {payload}\n\n"
                        yield msg.encode()
                        current = stream_id
                        try:
                            event = json.loads(payload)
                            if (
                                isinstance(event, dict)
                                and event.get("type") == "LOG_STREAM_CLOSED"
                            ):
                                yield (
                                    f"id: {stream_id}\nevent: eos\ndata:\n\n"
                                ).encode()
                                return
                        except json.JSONDecodeError:
                            event = None
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
