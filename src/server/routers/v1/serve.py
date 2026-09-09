"""The single authenticated, claim-gated task-ID serve endpoint.

An external FlowMesh principal reaches a public serve task only by its task ID, over one
FlowMesh-authenticated route. The endpoint authenticates the principal on a
non-forwarded channel (the client's ``Authorization`` neither grants access here nor is
forwarded upstream), checks the existing ``TASK`` read permission, resolves the task's
live standing serve binding, and admits the request through the same resident claim gate
as a workflow consumer.

The request is a transparent reverse proxy: whatever method, path, query, headers, and
body the client sends reach the engine unchanged, and the engine's own status, headers,
and body bytes come back unchanged, so any endpoint the engine serves is reachable and
FlowMesh applies no endpoint semantics of its own. The selected replica worker's
claim-gated sidecar does all engine work. The caller holds only the task ID, and the
engine listener, credential, worker, and routing stay resolved behind the binding.
"""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as ApiPath
from fastapi import Request, status
from fastapi.responses import StreamingResponse

from shared.resident.envelope import (
    TRANSPARENT_METHODS,
    EnvelopeRejected,
    freeze_request_envelope,
)

from ...app_state import get_gated_serve, get_logger
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...serve import GatedServe, ServeAccessMode
from ...serve.service import (
    BindingNotFound,
    IngressUnavailable,
    MethodNotAllowed,
    ServeResult,
    WrongIngress,
)

router = APIRouter(prefix="/serve", tags=["Serve"])

_MAX_REQUEST_BYTES = 100 * 1024 * 1024


class _ServeStreamTruncated(Exception):
    """A serve response failed after its head was sent, so the body is aborted."""


async def _read_capped_body(request: Request) -> bytes:
    """Read the raw request body, refusing one past the bound before it is buffered."""
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
    return b"".join(chunks)


@router.api_route(
    "/tasks/{task_id}/{upstream_path:path}",
    methods=list(TRANSPARENT_METHODS),
    summary="Gated resident serve request",
    description=(
        "Send a request to a public serve task by its task ID. "
        "FlowMesh-authenticated and claim-gated; the request is admitted through "
        "resident-capacity control and relayed unchanged to the task's standing "
        "replica, which returns the engine's own response."
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
        envelope = freeze_request_envelope(
            method=request.method,
            upstream_path=upstream_path,
            query=request.url.query,
            headers=list(request.headers.items()),
            body=body,
        )
    except EnvelopeRejected as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    try:
        result = gated_serve.submit(
            principal.principal_id,
            principal.org_id,
            task_id,
            envelope,
            ServeAccessMode.PROXY,
        )
    except BindingNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "serve task not found") from exc
    except WrongIngress as exc:
        # The task pins a different ingress, so it is not addressable here at all.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "serve task is not served on this ingress"
        ) from exc
    except MethodNotAllowed as exc:
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED, "method not allowed"
        ) from exc
    except IngressUnavailable as exc:
        # The task pins a gated ingress this deployment has not registered. It fails
        # closed: no other mode serves it, and no raw listener is exposed.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "serve task ingress is not available",
        ) from exc

    return await _stream(result)


async def _stream(result: ServeResult) -> StreamingResponse:
    """Read the engine response head, then stream the opaque body verbatim.

    The head carries the engine's own status and headers, set on the client response so
    the client sees the engine's envelope and a streamed body passes through. A failure
    before the head maps to a bad-gateway status; once the body has begun streaming the
    terminal disposition rides the stream itself.
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
    headers: list[tuple[str, str]] = []
    leading_chunk: bytes | None = None
    if first.kind == "head":
        status_code = first.status
        headers = list(first.headers)
    elif first.kind == "chunk":
        leading_chunk = first.payload

    async def body() -> AsyncIterator[bytes]:
        terminated = False
        try:
            if leading_chunk is not None:
                yield leading_chunk
            async for event in events:
                if event.kind == "chunk":
                    yield event.payload
                elif event.terminal:
                    terminated = True
                    if event.kind == "error":
                        # The stream lost frames or the drive failed after the head was
                        # committed: abort the response so the client sees a truncated
                        # body rather than a well-formed one silently missing bytes.
                        raise _ServeStreamTruncated(
                            event.detail or "resident serve stream lost"
                        )
        finally:
            # A client that disconnects mid-stream stops the body generator before its
            # terminal: close the client stream so a still-running drive stops teeing.
            # The credit is untouched — its own fenced terminal releases it.
            if not terminated:
                result.close_client()

    response = StreamingResponse(body(), status_code=status_code)
    if headers:
        # Set the raw header list directly: a mapping cannot express the repeated fields
        # the engine may send, and latin-1 is the byte-exact HTTP header transport, so
        # each field reaches the client as the bytes the engine wrote.
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in headers
        ]
    return response
