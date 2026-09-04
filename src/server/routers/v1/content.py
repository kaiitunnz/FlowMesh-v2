"""The reference-backed outcome content store's HTTP surface.

A worker materializes an outcome by uploading its bytes here; the store is content-
addressed and per-tenant, so the upload returns the immutable manifest and a re-drive
under the same idempotency key resolves the first materialization. A resumed worker
hydrates the content by digest before it injects the value. The server stores opaque
bytes and never assembles them into orchestration state.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from shared.outcome import ContentStoreError, OutcomeHydrationError, OutcomeManifest

from ...app_state import get_content_store, get_logger
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...services.content_store import ServerContentStore

router = APIRouter(prefix="/content", tags=["Content"])


def _require_store(store: ServerContentStore | None) -> ServerContentStore:
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="content store not enabled"
        )
    return store


@router.put(
    "",
    summary="Materialize outcome content",
    description="Upload outcome bytes content-addressed under the caller's tenant.",
)
async def put_content(
    request: Request,
    idem: str = Query(..., description="The fabric idempotency key to bind."),
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    body = await request.body()
    media_type = request.headers.get("content-type") or "application/octet-stream"
    try:
        return _require_store(store).materialize(
            principal.org_id,
            idem,
            body,
            media_type=media_type,
            provenance=f"principal:{principal.principal_id}",
        )
    except ContentStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get(
    "/by-idem/{idem}",
    summary="Resolve materialized content by idempotency key",
    description="Return the manifest already materialized under an idempotency key.",
)
async def get_by_idem(
    idem: str,
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    manifest = _require_store(store).find(principal.org_id, idem)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no content for key"
        )
    return manifest


@router.get(
    "/{digest}",
    summary="Hydrate outcome content",
    description="Return content-addressed outcome bytes scoped to the caller's tenant.",
    response_class=Response,
)
async def get_content(
    digest: str,
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    try:
        data = _require_store(store).read(principal.org_id, digest)
    except OutcomeHydrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="content not found"
        ) from exc
    except ContentStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return Response(content=data, media_type="application/octet-stream")
