"""Assemble resident-capacity control from the server runtime and configuration.

Builds the CS stores, the two admission/lifecycle actors, the inference adapter, and the
serve-substrate glue (materialize, stop, endpoint probe), returning the wired
``ResidentCapacityControl``. The materialized serve task is owned by the resolved system
principal so an operator reads its logs through the normal owner-scoped path.
"""

import logging
from collections.abc import Callable
from typing import Any

from lumid_hooks import PrincipalContext

from shared.schemas.command import CommandMessage, CommandType

from ..config import OrchestrationConfig, ResidentCapacityConfig
from ..network.service import NetworkPlane
from ..registries import WorkerRegistry
from ..registries.node import NodeRegistry
from ..registries.resident import ResidentRegistry
from ..task.models import TERMINAL_TASK_STATUSES
from ..task.runtime import TaskRuntime
from .adapter import HttpInferenceAdapter
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .materializer import materialize_resident_replica
from .native import NativeTransport, NativeTransportError
from .policy import ResidentPolicyLimits
from .service import NativeDeliveryDeps, ResidentCapacityControl
from .state import ReplicaEndpoint, ReplicaIncarnation, ServiceFamily
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
        adapter=HttpInferenceAdapter(
            timeout_sec=orchestration.gateway.timeout_sec,
            forward_api_key=cfg.forward_api_key,
        ),
        limits=limits,
        binding_resolver=runtime.gateway_binding_for,
        settle_cb=runtime.settle_episode_invocation,
        redispatch_cb=runtime.redispatch_episode_invocation,
        endpoint_probe=endpoint,
        logger=logger,
        poll_interval_sec=cfg.poll_interval_sec,
        idle_sweep_interval_sec=sweep_interval,
        redrive_backoff_sec=cfg.redrive_backoff_sec,
        max_transient_redrives=cfg.max_transient_redrives,
    )


def wire_native_delivery(
    resident_control: ResidentCapacityControl,
    *,
    network: NetworkPlane,
    worker_registry: WorkerRegistry,
    runtime: TaskRuntime,
    node_registry: NodeRegistry,
    resident_cfg: ResidentCapacityConfig,
    cmd_timeout_sec: float,
) -> None:
    """Wire native two-phase delivery into resident-capacity control.

    Resolves an origin task's node and a replica's node through the worker registry and
    carries the bootstrap/stream/cancel pokes as node commands, so a resident invocation
    is delivered data-direct over the fabric rather than relayed in-server.
    """

    def _node_of_worker(worker_id: str | None) -> str | None:
        if worker_id is None:
            return None
        worker = worker_registry.get_worker(worker_id)
        return worker.node_id if worker is not None else None

    def _origin_node_of_task(task_id: str) -> str | None:
        record = runtime.get_record(task_id)
        return _node_of_worker(record.assigned_worker) if record else None

    def _node_of_replica(replica: ReplicaIncarnation) -> str | None:
        if replica.serve_task_id is None:
            return None
        record = runtime.get_record(replica.serve_task_id)
        return _node_of_worker(record.assigned_worker) if record else None

    async def _exec_node_cmd(
        node_id: str, command: CommandType, payload: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await node_registry.exec_node_cmd(
            node_id,
            CommandMessage(command=command, payload=payload),
            timeout=cmd_timeout_sec,
        )
        if not resp.success:
            raise NativeTransportError(resp.message or "resident node command failed")
        return resp.data or {}

    resident_control.set_native_delivery(
        NativeDeliveryDeps(
            network=network,
            transport=NativeTransport(_exec_node_cmd),
            origin_node_of_task=_origin_node_of_task,
            node_of_replica=_node_of_replica,
            sidecar_bind_host=resident_cfg.sidecar_bind_host,
            directly_routable=resident_cfg.sidecar_directly_routable,
            forward_api_key=resident_cfg.forward_api_key,
            relay_only=resident_cfg.relay_only,
        )
    )
