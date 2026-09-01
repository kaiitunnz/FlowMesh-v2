"""Assemble resident-capacity control from the server runtime and configuration.

Builds the CS stores, the two admission/lifecycle actors, the inference adapter, and the
serve-substrate glue (materialize, stop, endpoint probe), returning the wired
``ResidentCapacityControl``. The materialized serve task is owned by the resolved system
principal so an operator reads its logs through the normal owner-scoped path.
"""

import logging
from collections.abc import Callable

from lumid_hooks import PrincipalContext

from ..config import OrchestrationConfig
from ..registries.resident import ResidentRegistry
from ..task.models import TERMINAL_TASK_STATUSES
from ..task.runtime import TaskRuntime
from .adapter import HttpInferenceAdapter
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .materializer import materialize_resident_replica
from .policy import ResidentPolicyLimits
from .service import ResidentCapacityControl
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

    async def stop(serve_task_id: str) -> None:
        record = runtime.get_record(serve_task_id)
        if record is not None:
            runtime.cancel_workflow(record.workflow_id, reason="resident idle teardown")

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
        endpoint_probe=endpoint,
        logger=logger,
        poll_interval_sec=cfg.poll_interval_sec,
        idle_sweep_interval_sec=sweep_interval,
    )
