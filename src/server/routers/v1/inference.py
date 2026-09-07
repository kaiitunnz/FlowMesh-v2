"""The controlled external inference ingress endpoint.

An authenticated external principal posts a request against a published alias. The edge
resolves the alias under the caller's tenant, admits it through the same resident claim
gate as a workflow consumer, and streams the model's response back as opaque frames the
edge relays unparsed. The endpoint is present only when the ingress is enabled.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as ApiPath
from fastapi import Request, Response, status
from fastapi.responses import StreamingResponse

from ...app_state import get_inference_ingress, get_logger
from ...auth.security import PrincipalContext, authenticate_connection
from ...ingress import InferenceIngress, QuotaExceeded
from ...ingress.service import (
    AliasNotFound,
    NoDeputyAvailable,
    TenantNotAuthorized,
)

router = APIRouter(prefix="/inference", tags=["Inference"])

_MAX_REQUEST_BYTES = 1024 * 1024


@router.post(
    "/{alias}",
    summary="Controlled external inference request",
    description=(
        "Run one inference request against a published, tenant-authorized service "
        "alias. The response body streams the model's completion as it is produced."
    ),
)
async def inference(
    request: Request,
    alias: str = ApiPath(description="The published service-family alias to invoke."),
    principal: PrincipalContext = Depends(authenticate_connection),
    ingress: InferenceIngress | None = Depends(get_inference_ingress),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    if ingress is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "inference ingress is not enabled"
        )
    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "request body too large"
        )
    try:
        payload = body.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "request body must be UTF-8")
    try:
        result = ingress.submit(principal, alias, payload)
    except AliasNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown alias {alias!r}")
    except TenantNotAuthorized:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, f"tenant is not authorized for alias {alias!r}"
        )
    except QuotaExceeded:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "per-principal request quota exceeded"
        )
    except NoDeputyAvailable:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no worker available to serve the request",
        )

    events = result.events()
    first = await anext(events, None)
    if first is None or (first.terminal and first.kind == "error"):
        detail = (first.detail if first else None) or "resident inference failed"
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail)
    if first.terminal:
        return Response(status_code=status.HTTP_200_OK)

    async def body_stream():
        yield first.payload
        async for event in events:
            if event.kind == "chunk":
                yield event.payload

    return StreamingResponse(body_stream(), media_type="text/plain")
