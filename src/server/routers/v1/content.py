"""The outcome-finalization index's HTTP surface.

A worker materializes an outcome into the shared content store and binds it here under
the fabric idempotency key it settles against, so a re-drive resolves the first
materialization instead of re-running a sampled producer. Only that binding crosses this
surface: the bytes are in the shared store and never pass through the server.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from shared.content import ContentReference
from shared.outcome import OutcomeManifest

from ...app_state import get_finalization_index, get_logger
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...content import FinalizationIndex
from ...hooks import ResourceAction, ResourceKind

router = APIRouter(prefix="/content", tags=["Content"])


def _require_index(index: FinalizationIndex | None) -> FinalizationIndex:
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="content store not enabled"
        )
    return index


def _admitted_scope(principal: PrincipalContext, requested: str | None) -> str:
    """The scope this request acts in: its own, or another it is privileged to reach.

    A fabric component finalizes an outcome on behalf of the task's tenant rather than
    its own credential, so a principal holding the deployment-wide scope may name one;
    anyone else is confined to the scope their principal is.
    """
    if not requested or requested == principal.org_id:
        return principal.org_id
    if "*" in principal.scopes:
        return requested
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="scope not permitted"
    )


@router.put(
    "/finalizations",
    summary="Bind an outcome finalization to its content",
    description="Record the content one idempotency key materialized.",
)
async def put_finalization(
    content: ContentReference,
    idem: str = Query(..., description="The fabric idempotency key to bind."),
    scope: str = Query("", description="The authorization scope to bind in."),
    index: FinalizationIndex | None = Depends(get_finalization_index),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    admitted = _admitted_scope(principal, scope)
    if content.authorization_scope != admitted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="content is outside the admitted scope",
        )
    return _require_index(index).record(
        admitted, idem, content, provenance=f"principal:{principal.principal_id}"
    )


@router.get(
    "/finalizations",
    summary="Resolve an outcome finalization",
    description="Return the content already materialized under an idempotency key.",
)
async def get_finalization(
    idem: str = Query(..., description="The fabric idempotency key to resolve."),
    scope: str = Query("", description="The authorization scope to resolve in."),
    index: FinalizationIndex | None = Depends(get_finalization_index),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    manifest = _require_index(index).find(_admitted_scope(principal, scope), idem)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no content for key"
        )
    return manifest
