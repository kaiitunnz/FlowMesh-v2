"""The single authenticated, claim-gated task-ID serve endpoint.

An external FlowMesh principal reaches a public serve task only by its task ID, over one
FlowMesh-authenticated route. The endpoint authenticates the principal on a
non-forwarded channel (the client's ``Authorization`` neither grants access here nor is
forwarded upstream), checks the existing ``TASK`` read permission, resolves the task's
live standing serve binding, and admits the request through the same resident claim gate
as a workflow consumer. It relays the binding-derived request and the opaque response
frames; the selected replica worker's claim-gated sidecar does all engine work. No raw
resident listener, engine credential, model, worker, or routing choice is ever exposed
to the caller.
"""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as ApiPath
from fastapi import Request, status
from fastapi.responses import StreamingResponse

from ...app_state import get_gated_serve, get_logger
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...serve import GatedServe
from ...serve.service import (
    BindingNotFound,
    MethodNotAllowed,
    PathNotAllowed,
    ServeResult,
)

router = APIRouter(prefix="/serve", tags=["Serve"])

_MAX_REQUEST_BYTES = 4 * 1024 * 1024


async def _read_capped_body(request: Request) -> str:
    declared = request.headers.get("content-length")
    if (
        declared is not None
        and declared.isdigit()
        and int(declared) > _MAX_REQUEST_BYTES
    ):
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "request body too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_REQUEST_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, "request body too large"
            )
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "request body must be UTF-8"
        ) from exc


@router.api_route(
    "/tasks/{task_id}/{upstream_path:path}",
    methods=["POST"],
    summary="Gated resident serve request",
    description=(
        "Submit an inference request to a public serve task by its task ID. "
        "FlowMesh-authenticated and claim-gated; the request is admitted through "
        "resident-capacity control and served by the task's standing replica."
    ),
)
async def serve_gated(
    request: Request,
    task_id: str = ApiPath(..., min_length=1),
    upstream_path: str = ApiPath(...),
    principal: PrincipalContext = Depends(authenticate_connection),
    gated_serve: GatedServe | None = Depends(get_gated_serve),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    await require_permission(
        principal, ResourceKind.TASK, task_id, ResourceAction.READ, logger
    )
    if gated_serve is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "serve task not found")
    body = await _read_capped_body(request)
    try:
        result = gated_serve.submit(
            principal.principal_id,
            principal.org_id,
            task_id,
            request.method,
            upstream_path,
            body,
        )
    except BindingNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "serve task not found") from exc
    except MethodNotAllowed as exc:
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED, "method not allowed"
        ) from exc
    except PathNotAllowed as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found") from exc

    return await _stream(result)


async def _stream(result: ServeResult) -> StreamingResponse:
    """Read the engine response head, then stream the opaque body verbatim.

    The head carries the engine's own status and content type, set on the client
    response so an OpenAI-compatible client sees the engine envelope and a streamed body
    passes through. A failure before the head maps to a bad-gateway status; once the
    body has begun streaming the terminal disposition rides the stream itself.
    """
    events = result.events()
    first = await anext(events, None)
    if first is None or (first.terminal and first.kind == "error"):
        detail = (
            first.detail if first is not None else "resident serve produced no result"
        )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail or "resident serve error"
        )

    status_code = status.HTTP_200_OK
    media_type = "text/plain; charset=utf-8"
    leading_chunk: str | None = None
    if first.kind == "head":
        status_code = first.status
        media_type = first.content_type
    elif first.kind == "chunk":
        leading_chunk = first.payload

    async def body() -> AsyncIterator[str]:
        if leading_chunk is not None:
            yield leading_chunk
        async for event in events:
            if event.kind == "chunk":
                yield event.payload

    return StreamingResponse(body(), status_code=status_code, media_type=media_type)
