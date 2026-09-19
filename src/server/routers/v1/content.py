"""The reference-backed outcome content store's HTTP surface.

A worker materializes an outcome by uploading its bytes here; the store is content-
addressed and per-scope, so the upload returns the immutable manifest and a re-drive
under the same idempotency key resolves the first materialization. A resumed worker
hydrates the content by digest before it injects the value. The server stores opaque
bytes and never assembles them into orchestration state.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from shared.content import ContentHydrationError, ContentReference, ContentStoreError
from shared.outcome import OutcomeManifest

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


def _admitted_scope(principal: PrincipalContext, requested: str | None) -> str:
    """The scope this request acts in: its own, or another it is privileged to reach.

    A fabric component writes content on behalf of the task's tenant rather than its own
    credential, so a principal holding the deployment-wide scope may name one; anyone
    else is confined to the scope their principal is.
    """
    if not requested or requested == principal.org_id:
        return principal.org_id
    if "*" in principal.scopes:
        return requested
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="scope not permitted"
    )


@router.put(
    "",
    summary="Materialize outcome content",
    description="Upload outcome bytes content-addressed under an admitted scope.",
)
async def put_content(
    request: Request,
    idem: str = Query(..., description="The fabric idempotency key to bind."),
    scope: str = Query("", description="The authorization scope to store under."),
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
            _admitted_scope(principal, scope),
            idem,
            body,
            media_type=media_type,
            provenance=f"principal:{principal.principal_id}",
        )
    except ContentStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.put(
    "/objects",
    summary="Write an immutable object",
    description="Store bytes content-addressed under an admitted scope.",
)
async def put_object(
    request: Request,
    scope: str = Query("", description="The authorization scope to store under."),
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> ContentReference:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    body = await request.body()
    media_type = request.headers.get("content-type") or "application/octet-stream"
    try:
        return _require_store(store).write(
            _admitted_scope(principal, scope), body, media_type=media_type
        )
    except ContentStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get(
    "",
    summary="Resolve materialized content by idempotency key",
    description="Return the manifest already materialized under an idempotency key.",
)
async def get_by_idem(
    idem: str = Query(..., description="The fabric idempotency key to resolve."),
    scope: str = Query("", description="The authorization scope to resolve in."),
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    manifest = _require_store(store).find(_admitted_scope(principal, scope), idem)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no content for key"
        )
    return manifest


@router.get(
    "/{digest}",
    summary="Hydrate outcome content",
    description="Return content-addressed outcome bytes from an admitted scope.",
    response_class=Response,
)
async def get_content(
    digest: str,
    scope: str = Query("", description="The authorization scope to read in."),
    store: ServerContentStore | None = Depends(get_content_store),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    try:
        data = _require_store(store).read(_admitted_scope(principal, scope), digest)
    except ContentHydrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="content not found"
        ) from exc
    except ContentStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return Response(content=data, media_type="application/octet-stream")
