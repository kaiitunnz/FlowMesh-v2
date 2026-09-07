"""Assemble resident-capacity control from the server runtime and configuration.

Builds the CS stores, the two admission/lifecycle actors, and the serve-substrate glue
(materialize, stop, endpoint probe), returning the wired ``ResidentCapacityControl``.
The worker-owned data path is wired separately once the network plane and worker
registry are available. The materialized serve task is owned by the resolved system
principal so an operator reads its logs through the normal owner-scoped path.
"""

import logging
from collections.abc import Callable
from typing import Any

from lumid_hooks import PrincipalContext

from shared.resident.contracts import ReplicaEndpoint
from shared.schemas.command import MediatedOpMessage

from ..config import OrchestrationConfig, ResidentCapacityConfig
from ..network.reverse_relay import RelaySessionStore
from ..network.service import NetworkPlane
from ..registries import WorkerRegistry
from ..registries.resident import ResidentRegistry
from ..task.models import TERMINAL_TASK_STATUSES
from ..task.runtime import TaskRuntime
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .materializer import materialize_resident_replica
from .policy import ResidentPolicyLimits
from .service import ResidentCapacityControl, ResidentWorkerDelivery
from .state import ReplicaIncarnation, ServiceFamily
from .stores import ResidentStores

# Yields the resolved system principal, read lazily so materialization uses the
# principal resolved during lifespan startup rather than one captured at wiring time.
SystemPrincipalProvider = Callable[[], PrincipalContext]


def build_resident_capacity(
    *,
    runtime: TaskRuntime,
    orchestration: OrchestrationConfig,
    system_principal: SystemPrincipalProvider,
    registry: ResidentRegistry,
    logger: logging.Logger,
) -> ResidentCapacityControl:
    """Wire and return resident-capacity control for the enabled resident config."""
    cfg = orchestration.resident
    stores = ResidentStores()
    limits = ResidentPolicyLimits(
        allowed_models=frozenset(cfg.allowed_models),
        max_replicas_per_family=cfg.max_replicas_per_family,
        max_concurrent_cold_starts=cfg.max_concurrent_cold_starts,
        cold_start_deadline_sec=cfg.cold_start_deadline_sec,
        selection_strategy=cfg.selection_strategy,
    )

    def persist() -> None:
        registry.save_snapshot(stores.to_snapshot())

    async def materialize(family: ServiceFamily, replica: ReplicaIncarnation) -> str:
        return await materialize_resident_replica(
            runtime, system_principal(), cfg, family, replica, logger
        )

    def stop(serve_task_id: str) -> None:
        record = runtime.get_record(serve_task_id)
        if record is not None:
            runtime.cancel_workflow(
                record.workflow_id, reason="resident replica teardown"
            )

    lifecycle = LifecycleScaleManager(
        stores,
        limits=limits,
        admission_slots=cfg.admission_slots,
        idle_retain_sec=cfg.idle_retain_sec,
        persist=persist,
        materialize_fn=materialize,
        stop_fn=stop,
    )

    def endpoint(serve_task_id: str) -> ReplicaEndpoint | None:
        record = runtime.get_record(serve_task_id)
        # A terminal or absent serve task is known-dead: report no endpoint so the
        # replica is invalidated rather than re-reported live.
        if (
            record is None
            or record.status in TERMINAL_TASK_STATUSES
            or not record.latest_update
        ):
            return None
        serve = record.latest_update.get("serve")
        if not isinstance(serve, dict):
            return None
        host, port = serve.get("host"), serve.get("port")
        if not host or not port:
            return None
        return ReplicaEndpoint(
            base_url=f"http://{host}:{port}/v1",
            model=str(serve.get("model") or ""),
            api_key=serve.get("api_key"),
        )

    sweep_interval = cfg.idle_sweep_interval_sec if cfg.idle_retain_sec > 0 else 0.0
    return ResidentCapacityControl(
        stores=stores,
        admission=AdmissionController(stores, persist),
        lifecycle=lifecycle,
        limits=limits,
        dependency_resolver=runtime.resolve_service_dependency,
        settle_cb=runtime.settle_episode_invocation,
        redispatch_cb=runtime.redispatch_episode_invocation,
        endpoint_probe=endpoint,
        logger=logger,
        poll_interval_sec=cfg.poll_interval_sec,
        idle_sweep_interval_sec=sweep_interval,
        redrive_backoff_sec=cfg.redrive_backoff_sec,
        max_transient_redrives=cfg.max_transient_redrives,
    )


def wire_worker_delivery(
    resident_control: ResidentCapacityControl,
    *,
    network: NetworkPlane,
    worker_registry: WorkerRegistry,
    runtime: TaskRuntime,
    sessions: RelaySessionStore,
    resident_cfg: ResidentCapacityConfig,
) -> None:
    """Wire the worker-owned resident data path into resident-capacity control.

    Resolves an agent task's origin worker and a worker's node through the worker
    registry, relays resident control frames over the worker attachment, and writes the
    per-session routing record the reverse-relay bridges route frames by. The origin
    worker carries the request and serves the engine stream data-direct over the fabric.
    """

    def _relay(worker_id: str, frame_kind: str, payload: dict[str, Any]) -> bool:
        worker = worker_registry.get_worker(worker_id)
        if worker is None:
            return False
        worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id, frame_kind=frame_kind, payload=payload
            ),
        )
        return True

    def _node_of_worker(worker_id: str | None) -> str | None:
        if worker_id is None:
            return None
        worker = worker_registry.get_worker(worker_id)
        return worker.node_id if worker is not None else None

    def _origin_worker_of_task(task_id: str) -> str | None:
        record = runtime.get_record(task_id)
        return record.assigned_worker if record else None

    def _serve_worker_of(replica: ReplicaIncarnation) -> str | None:
        if replica.serve_task_id is None:
            return None
        record = runtime.get_record(replica.serve_task_id)
        return record.assigned_worker if record else None

    resident_control.set_worker_delivery(
        ResidentWorkerDelivery(
            relay=_relay,
            origin_worker_of_task=_origin_worker_of_task,
            serve_worker_of=_serve_worker_of,
            node_of_worker=_node_of_worker,
            network=network,
            sessions=sessions,
            directly_routable=resident_cfg.sidecar_directly_routable,
            forward_api_key=resident_cfg.forward_api_key,
        )
    )
