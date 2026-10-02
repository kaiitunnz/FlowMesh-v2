import logging

from fastapi import APIRouter, Depends, Request

from ...app_state import get_logger, get_resident_control
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...resident.service import ResidentCapacityControl
from ...schemas.resident import (
    ResidentClaimInfo,
    ResidentClaimsView,
    ResidentFamilyInfo,
    ResidentReplicaCredit,
    ResidentReplicaInfo,
)
from ._listing import query_filter

router = APIRouter(prefix="/resident", tags=["Resident"])

RESIDENT_REPLICA_FILTER_FIELDS = frozenset(
    {
        "replica_id",
        "family",
        "state",
        "healthy",
        "serve_task_id",
        "worker_id",
    }
)


async def _require_admin(principal: PrincipalContext, logger: logging.Logger) -> None:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )


@router.get(
    "/families",
    summary="List resident service families",
    description="List the registered resident-capacity service families.",
)
async def list_resident_families(
    principal: PrincipalContext = Depends(authenticate_connection),
    control: ResidentCapacityControl | None = Depends(get_resident_control),
    logger: logging.Logger = Depends(get_logger),
) -> list[ResidentFamilyInfo]:
    await _require_admin(principal, logger)
    if control is None:
        return []
    return [
        ResidentFamilyInfo.project(family) for family in control.list_service_families()
    ]


@router.get(
    "/replicas",
    summary="List resident replica incarnations",
    description="List live and inert resident replica incarnations.",
)
async def list_resident_replicas(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    control: ResidentCapacityControl | None = Depends(get_resident_control),
    logger: logging.Logger = Depends(get_logger),
) -> list[ResidentReplicaInfo]:
    query = query_filter(request, RESIDENT_REPLICA_FILTER_FIELDS)
    await _require_admin(principal, logger)
    if control is None:
        return []
    replicas = [
        ResidentReplicaInfo.project(replica)
        for replica in control.list_replica_incarnations()
    ]
    return query.filter(replicas)


@router.get(
    "/claims",
    summary="List resident admission claims",
    description="List credit-bearing admission claims and per-replica held credit.",
)
async def list_resident_claims(
    principal: PrincipalContext = Depends(authenticate_connection),
    control: ResidentCapacityControl | None = Depends(get_resident_control),
    logger: logging.Logger = Depends(get_logger),
) -> ResidentClaimsView:
    await _require_admin(principal, logger)
    if control is None:
        return ResidentClaimsView()
    claims, held = control.list_credit_bearing_claims()
    return ResidentClaimsView(
        claims=[ResidentClaimInfo.project(claim) for claim in claims],
        held_credit=[
            ResidentReplicaCredit(replica_id=replica_id, held_slots=slots)
            for replica_id, slots in held.items()
        ],
    )
