import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ...app_state import (
    get_logger,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
    resolve_accessible_ids,
)
from ...hooks import ResourceAction, ResourceKind
from ...registries.worker import WorkerInfo, WorkerRegistry
from ._listing import ListFilter, filter_params

router = APIRouter(prefix="/workers", tags=["Workers"])


WORKER_FILTER_FIELDS = frozenset(
    {
        "id",
        "alias",
        "namespace",
        "cluster",
        "node_id",
        "node_alias",
        "version",
        "status",
        "stale",
        "tags",
        "cached_models",
        "cached_datasets",
        "capabilities.supported_task_types",
        "capabilities.ssh_noninteractive",
        "hardware.cpu.model",
        "hardware.gpu.driver_version",
        "hardware.gpu.cuda_version",
        "hardware.network.ip",
    }
)


@router.get(
    "",
    summary="List workers",
    description="List all registered workers with optional filtering.",
    response_description="List of workers",
)
async def list_workers(
    principal: PrincipalContext = Depends(authenticate_connection),
    filters: ListFilter = Depends(filter_params(WORKER_FILTER_FIELDS)),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[WorkerInfo]:
    query = filters.parse()
    workers = await registry.list_workers_async()
    allowed = await resolve_accessible_ids(
        principal, ResourceKind.WORKER, ResourceAction.READ, logger
    )
    if allowed is not None:
        workers = [w for w in workers if w.id in allowed]
    return query.filter(workers)


@router.get(
    "/{worker_id}",
    summary="Get a worker",
    description="Get worker information by ID.",
    response_description="Worker information",
)
async def get_worker(
    worker_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerInfo:
    await require_permission(
        principal, ResourceKind.WORKER, worker_id, ResourceAction.READ, logger
    )
    worker = await registry.get_worker_async(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="worker not found"
        )
    stale = await registry.is_worker_stale_async(worker.id)
    return WorkerInfo(**worker.model_dump(), stale=stale)
