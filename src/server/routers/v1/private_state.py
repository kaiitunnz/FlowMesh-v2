import logging

from fastapi import APIRouter, Depends, Request

from ...app_state import get_logger, get_policy_surface, get_runtime
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...policy import PolicySurface
from ...schemas.private_state import SealedGenerationInfo
from ...services.private_state_inventory import sealed_state_inventory
from ...task.runtime import TaskRuntime
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/private-state", tags=["Private state"])


@router.get(
    "/generations",
    summary="List sealed private-state generations",
    description=(
        "List the sealed activation-private-state generations the live engines record, "
        "with their holder evidence and advisory state-control decision."
    ),
)
async def list_sealed_generations(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    surface: PolicySurface | None = Depends(get_policy_surface),
    logger: logging.Logger = Depends(get_logger),
) -> list[SealedGenerationInfo]:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )
    if surface is None:
        return []
    generations = [
        SealedGenerationInfo.project(entry)
        for entry in sealed_state_inventory(runtime, surface)
    ]
    return filter_models_by_queries(generations, request.query_params)
