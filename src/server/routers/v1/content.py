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


def _binding_scope(
    index: FinalizationIndex,
    principal: PrincipalContext,
    idempotency_key: str,
    asserted: str,
) -> str:
    """The scope this key's finalization binds in: the one control assigned it.

    A finalization is reported by the worker that produced the content, and a worker
    carries the scope its work was authorized under rather than choosing one. So the
    scope comes from what control recorded when it authorized this key — not from the
    request, whose own ``scope`` is an assertion this checks and never a way to widen
    what the reporter reaches. A key control assigned no scope has no binding to make.
    """
    assigned = index.assigned_scope(idempotency_key)
    if not assigned:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="no scope is assigned to this idempotency key",
        )
    if asserted and asserted != assigned:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="scope is not the one assigned to this idempotency key",
        )
    if assigned != principal.org_id and "*" not in principal.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="scope not permitted"
        )
    return assigned


@router.put(
    "/finalizations",
    summary="Bind an outcome finalization to its content",
    description="Record the content one idempotency key materialized.",
)
async def put_finalization(
    content: ContentReference,
    idem: str = Query(..., description="The fabric idempotency key to bind."),
    scope: str = Query("", description="The scope asserted for the binding."),
    index: FinalizationIndex | None = Depends(get_finalization_index),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    bound = _require_index(index)
    admitted = _binding_scope(bound, principal, idem, scope)
    if content.authorization_scope != admitted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="content is outside the admitted scope",
        )
    return bound.record(
        admitted, idem, content, provenance=f"principal:{principal.principal_id}"
    )


@router.get(
    "/finalizations",
    summary="Resolve an outcome finalization",
    description="Return the content already materialized under an idempotency key.",
)
async def get_finalization(
    idem: str = Query(..., description="The fabric idempotency key to resolve."),
    scope: str = Query("", description="The scope asserted for the lookup."),
    index: FinalizationIndex | None = Depends(get_finalization_index),
    principal: PrincipalContext = Depends(authenticate_connection),
    logger: logging.Logger = Depends(get_logger),
) -> OutcomeManifest:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    bound = _require_index(index)
    manifest = bound.find(_binding_scope(bound, principal, idem, scope), idem)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no content for key"
        )
    return manifest
