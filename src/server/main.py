import argparse
import asyncio
import atexit
import importlib
import inspect
import os
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import uvicorn
from fastapi import FastAPI
from flowmesh_hook import ResourceKind
from lumid_hooks import HookBindings, ResourceRef

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "server"
    sys.modules.setdefault("server.main", sys.modules[__name__])

from shared._version import FLOWMESH_RELEASE_VERSION
from shared.schemas.command import CommandMessage, CommandType

from .auth import reconcile_resources, resolve_system_principal
from .clients import RedisClient
from .clients.redis import resident_relay_client
from .config import NodeRole, ServerConfig
from .dispatcher.factory import create_dispatcher
from .hooks import register
from .network.rendezvous import RootCursorStore, RootRendezvousBridge
from .network.reverse_relay import BinaryRedis, RelaySessionStore, RelayStreamStore
from .network.service import NetworkPlane
from .orchestration.tool_dispatch import ToolInvocationEnvelope
from .registries import WorkerRegistry, WorkflowRegistry
from .registries.node import NodeRegistry
from .registries.resident import ResidentRegistry
from .resident.native import NativeTransport, NativeTransportError
from .resident.service import NativeDeliveryDeps
from .resident.state import ReplicaIncarnation
from .resident.wiring import build_resident_capacity
from .routers import docs, health, v1
from .services.agent_model_gateway import (
    AgentModelGateway,
    ResolvedGatewayBinding,
    build_agent_model_router,
    to_gateway_binding,
)
from .services.fabric_tool_broker import FabricToolBroker
from .services.log_archiver import TaskLogArchiver
from .services.metrics import MetricsRecorder
from .services.model_secret_vault import ModelSecretVault
from .services.monitoring import EventMonitor
from .services.port_forward import PortForwardService
from .services.ssh_audit import SshAuditService
from .services.watchdog import WorkerWatchdog
from .startup import rehydrate_root_state
from .supervisor import WorkerSupervisor
from .task.runtime import TaskRuntime
from .utils.logging import get_logger

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

config = ServerConfig.from_env()
NODE_ROLE = config.identity.role
IS_ROOT_NODE = NODE_ROLE is NodeRole.ROOT

if NODE_ROLE is NodeRole.WORKER and not config.worker_management.enabled:
    raise SystemExit("Worker node role requires ENABLE_SUPERVISOR=true")

# --------------------------------------------------------------------------- #
# Shared services (all node roles)
# --------------------------------------------------------------------------- #

logger = get_logger(
    name="server",
    log_file=config.logging.file,
    max_bytes=config.logging.max_bytes,
    backup_count=config.logging.backup_count,
    level=config.logging.level,
)

# Result & metrics directories
RESULTS_DIR = config.results_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert config.metrics.dir is not None
METRICS_DIR = config.metrics.dir

# Redis connections
REDIS_CLIENT = RedisClient(
    control_url=config.redis.control_url,
    telemetry_url=config.redis.telemetry_url,
    logger=logger,
    acl_enabled=config.redis.acl_enabled,
    username=config.redis.username,
    password=config.redis.password,
    tls_ca_file=config.redis.tls_ca_file,
)

NODE_REGISTRY = NodeRegistry(REDIS_CLIENT, logger)

METRICS_RECORDER = MetricsRecorder(
    METRICS_DIR,
    logger,
    enable_density_plot=config.metrics.enable_density_plot,
    density_bucket_seconds=config.metrics.density_bucket_sec,
)

SUPERVISOR: WorkerSupervisor | None = None
if config.worker_management.enabled:
    SUPERVISOR = WorkerSupervisor(
        identity=config.identity,
        redis=config.redis,
        grpc=config.grpc,
        worker_management=config.worker_management,
        logging_config=config.logging,
        logger=logger,
        network=config.orchestration.network,
    )

# --------------------------------------------------------------------------- #
# Root node services (orchestrator)
# --------------------------------------------------------------------------- #

WORKFLOW_REGISTRY = None
WORKER_REGISTRY = None
RUNTIME = None
DISPATCHER = None
SSH_AUDIT_SERVICE = None
PORT_FORWARD_SERVICE = None
WATCHDOG = None
EVENT_MONITOR = None
LOG_ARCHIVER = None
AGENT_MODEL_GATEWAY = None
FABRIC_TOOL_BROKER = None
RESIDENT_CONTROL = None
RESIDENT_REGISTRY = None
NETWORK_PLANE = None
RESIDENT_BRIDGE = None
RESIDENT_BRIDGE_TASK = None

if IS_ROOT_NODE:
    WORKFLOW_REGISTRY = WorkflowRegistry(REDIS_CLIENT)
    WORKER_REGISTRY = WorkerRegistry(REDIS_CLIENT)
    MODEL_SECRET_VAULT = ModelSecretVault(
        REDIS_CLIENT, config.orchestration.model_secret_vault.ttl_sec, logger
    )
    RUNTIME = TaskRuntime(
        WORKFLOW_REGISTRY,
        WORKER_REGISTRY,
        config.orchestration,
        RESULTS_DIR,
        logger,
        secret_vault=MODEL_SECRET_VAULT,
    )
    AGENT_MODEL_GATEWAY = AgentModelGateway(
        RUNTIME, config.orchestration.gateway, logger
    )
    RUNTIME.set_model_settler(AGENT_MODEL_GATEWAY.settle)
    AGENT_MODEL_GATEWAY.set_facade_group_originator(RUNTIME.originate_facade_turn_group)
    AGENT_MODEL_GATEWAY.set_facade_fence(RUNTIME.has_pending_facade)
    AGENT_MODEL_GATEWAY.set_facade_resolver(RUNTIME.agent_facade_descriptors)

    def _settle_tool(task_id: str, call_correlation: str, value: str) -> None:
        assert RUNTIME is not None
        RUNTIME.settle_episode_invocation(task_id, call_correlation, value)

    FABRIC_TOOL_BROKER = FabricToolBroker(
        config.orchestration.web_search, _settle_tool, logger=logger
    )
    RUNTIME.set_tool_broker(FABRIC_TOOL_BROKER.submit)

    def _resolve_gateway_binding(task_id: str) -> ResolvedGatewayBinding | None:
        assert RUNTIME is not None
        resolved = RUNTIME.gateway_binding_for(task_id)
        if resolved is None:
            return None
        workflow_id, pinned = resolved
        return to_gateway_binding(pinned, MODEL_SECRET_VAULT, workflow_id)

    AGENT_MODEL_GATEWAY.set_binding_resolver(_resolve_gateway_binding)

    if config.orchestration.resident.enabled:
        RESIDENT_REGISTRY = ResidentRegistry(REDIS_CLIENT)

        def _resident_owner():
            return app.state.system_principal

        RESIDENT_CONTROL = build_resident_capacity(
            runtime=RUNTIME,
            orchestration=config.orchestration,
            system_principal=_resident_owner,
            registry=RESIDENT_REGISTRY,
            logger=logger,
        )
        RUNTIME.set_resident_terminal_hook(RESIDENT_CONTROL.on_invocation_terminal)

        def _model_settle(env: ToolInvocationEnvelope) -> None:
            assert RESIDENT_CONTROL is not None and AGENT_MODEL_GATEWAY is not None
            if RESIDENT_CONTROL.is_resident(env.task_id):
                RESIDENT_CONTROL.settle(env)
            else:
                AGENT_MODEL_GATEWAY.settle(env)

        RUNTIME.set_model_settler(_model_settle)

    if config.orchestration.network.enabled:
        NETWORK_PLANE = NetworkPlane(
            config.orchestration.network, NODE_REGISTRY, logger
        )
        _relay_redis = cast(
            BinaryRedis,
            resident_relay_client(
                config.redis.resident_relay_url,
                acl_enabled=config.redis.acl_enabled,
                username=config.redis.username,
                password=config.redis.password,
                tls_ca_file=config.redis.tls_ca_file,
            ),
        )
        RESIDENT_BRIDGE = RootRendezvousBridge(
            RelayStreamStore(_relay_redis),
            RelaySessionStore(_relay_redis),
            RootCursorStore(_relay_redis),
            logger=logger,
        )

    if (
        RESIDENT_CONTROL is not None
        and NETWORK_PLANE is not None
        and WORKER_REGISTRY is not None
    ):
        _resident_cfg = config.orchestration.resident
        _network = NETWORK_PLANE
        _workers = WORKER_REGISTRY
        _runtime = RUNTIME
        assert _runtime is not None
        _cmd_timeout = max(
            config.orchestration.gateway.timeout_sec,
            _resident_cfg.cold_start_deadline_sec,
        )

        async def _exec_node_cmd(
            node_id: str, command: CommandType, payload: dict[str, Any]
        ) -> dict[str, Any]:
            resp = await NODE_REGISTRY.exec_node_cmd(
                node_id,
                CommandMessage(command=command, payload=payload),
                timeout=_cmd_timeout,
            )
            if not resp.success:
                raise NativeTransportError(
                    resp.message or "resident node command failed"
                )
            return resp.data or {}

        def _node_of_worker(worker_id: str | None) -> str | None:
            if worker_id is None:
                return None
            worker = _workers.get_worker(worker_id)
            return worker.node_id if worker is not None else None

        def _origin_node_of_task(task_id: str) -> str | None:
            record = _runtime.get_record(task_id)
            return _node_of_worker(record.assigned_worker) if record else None

        def _node_of_replica(replica: ReplicaIncarnation) -> str | None:
            if replica.serve_task_id is None:
                return None
            record = _runtime.get_record(replica.serve_task_id)
            return _node_of_worker(record.assigned_worker) if record else None

        RESIDENT_CONTROL.set_native_delivery(
            NativeDeliveryDeps(
                network=_network,
                transport=NativeTransport(_exec_node_cmd),
                origin_node_of_task=_origin_node_of_task,
                node_of_replica=_node_of_replica,
                sidecar_bind_host=_resident_cfg.sidecar_bind_host,
                directly_routable=_resident_cfg.sidecar_directly_routable,
                forward_api_key=_resident_cfg.forward_api_key,
                relay_only=_resident_cfg.relay_only,
            )
        )

    DISPATCHER = create_dispatcher(
        config.dispatch,
        RUNTIME,
        WORKER_REGISTRY,
        RESULTS_DIR,
        logger=logger,
        metrics_recorder=METRICS_RECORDER,
    )

    _pf_cfg = config.port_forward
    if _pf_cfg.ssh_audit_enabled:
        SSH_AUDIT_SERVICE = SshAuditService(REDIS_CLIENT)

    if _pf_cfg.enabled:
        PORT_FORWARD_SERVICE = PortForwardService(
            redis_client=REDIS_CLIENT,
            node_registry=NODE_REGISTRY,
            worker_registry=WORKER_REGISTRY,
            ssh_audit=SSH_AUDIT_SERVICE,
            bind_host=_pf_cfg.bind_host,
            public_host=_pf_cfg.public_host,
            port_start=_pf_cfg.port_start,
            port_end=_pf_cfg.port_end,
            persistent_listeners=_pf_cfg.persistent_listeners,
            logger=logger,
        )

    WATCHDOG = WorkerWatchdog(
        REDIS_CLIENT.sync,
        WORKER_REGISTRY,
        RUNTIME,
        DISPATCHER,
        logger,
        enabled=config.watchdog.enabled,
        check_interval=config.watchdog.check_interval,
        grace_seconds=config.watchdog.grace_sec,
        rehydration_grace_seconds=config.watchdog.rehydration_grace_sec,
    )

    EVENT_MONITOR = EventMonitor(
        redis_client=REDIS_CLIENT.sync,
        logger=logger,
        runtime=RUNTIME,
        dispatcher=DISPATCHER,
        worker_registry=WORKER_REGISTRY,
        node_registry=NODE_REGISTRY,
        metrics_recorder=METRICS_RECORDER,
        watchdog=WATCHDOG,
        ssh_proxy_enabled=config.port_forward.ssh_proxy_enabled,
        serve_proxy_enabled=config.port_forward.serve_proxy_enabled,
        port_forward=PORT_FORWARD_SERVICE,
        results_dir=RESULTS_DIR,
        log_stream_ttl_sec=config.log_stream.ttl_sec,
        server_base_url=config.identity.base_url,
        on_node_removed=(
            NETWORK_PLANE.forget_node if NETWORK_PLANE is not None else None
        ),
    )

    LOG_ARCHIVER = TaskLogArchiver(
        redis=REDIS_CLIENT.sync,
        runtime=RUNTIME,
        results_dir=RESULTS_DIR,
        logger=logger,
        flush_interval_sec=config.log_stream.archive_flush_interval_sec,
        flush_max_entries=config.log_stream.archive_flush_max_entries,
    )

# --------------------------------------------------------------------------- #
# Metrics export hook
# --------------------------------------------------------------------------- #


def _export_metrics_on_exit() -> None:
    try:
        METRICS_RECORDER.finalize_density_series()
        result = METRICS_RECORDER.export_final_report()
        report = result.get("report") or {}
        summary = METRICS_RECORDER.format_report(report)
        if summary:
            logger.info(summary)
        path = result.get("path")
        if path:
            logger.info("Metrics report saved to %s", path)
    except Exception as exc:
        logger.warning("Failed to export metrics summary: %s", exc)


atexit.register(_export_metrics_on_exit)

# --------------------------------------------------------------------------- #
# Background threads (root node only)
# --------------------------------------------------------------------------- #

STOP_EVENT = threading.Event()
BACKGROUND_THREADS: list[threading.Thread] = []


def _start_root_threads() -> None:
    """Start orchestrator background threads. Only called on root nodes."""
    assert DISPATCHER is not None
    assert LOG_ARCHIVER is not None
    assert WATCHDOG is not None

    dispatcher = DISPATCHER
    dispatch_thread = threading.Thread(
        target=lambda: dispatcher.dispatch_loop(STOP_EVENT, poll_interval=1.0),
        name="dispatch-loop",
        daemon=True,
    )
    dispatch_thread.start()
    BACKGROUND_THREADS.append(dispatch_thread)

    log_archiver_thread = threading.Thread(
        target=LOG_ARCHIVER.run, args=(STOP_EVENT,), name="log-archiver", daemon=True
    )
    log_archiver_thread.start()
    BACKGROUND_THREADS.append(log_archiver_thread)

    watchdog_thread = WATCHDOG.start(STOP_EVENT)
    if watchdog_thread and watchdog_thread not in BACKGROUND_THREADS:
        BACKGROUND_THREADS.append(watchdog_thread)

    node_registry_thread = NODE_REGISTRY.start()
    BACKGROUND_THREADS.append(node_registry_thread)


def _stop_background() -> None:
    STOP_EVENT.set()
    NODE_REGISTRY.shutdown()
    if RUNTIME is not None:
        RUNTIME.shutdown()
    for thread in BACKGROUND_THREADS:
        thread.join(timeout=2.0)
    BACKGROUND_THREADS.clear()


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

openapi_tags = [
    {"name": "Health", "description": "Service health and readiness endpoints."},
    {"name": "Documentations", "description": "API documentation endpoints."},
    {"name": "Workflows", "description": "Workflow submission and lifecycle."},
    {"name": "Tasks", "description": "Task inspection and status."},
    {"name": "Results", "description": "Task results and artifact handling."},
    {"name": "Workers", "description": "Worker pool operations and metadata."},
    {"name": "Nodes", "description": "Node registry and worker control."},
    {"name": "SSH", "description": "SSH proxy endpoint for task connectivity."},
    {"name": "Serve", "description": "HTTP reverse-proxy endpoint for serve tasks."},
    {"name": "System", "description": "System metrics and admin operations."},
    {"name": "Stack", "description": "Local worker lifecycle management."},
]

app = FastAPI(
    title="FlowMesh Server",
    version=FLOWMESH_RELEASE_VERSION,
    openapi_tags=openapi_tags,
    docs_url=None,
    redoc_url=None,
)


async def _load_plugins(stack: AsyncExitStack) -> None:
    """Load FLOWMESH_PLUGINS modules and drain their `HookBindings` into the
    server's runtime registries.

    A plugin's `install()` is either:
      - a sync function returning a `HookBindings`, or
      - an `@asynccontextmanager async def` yielding a `HookBindings` (the
        ctx manager registers on enter, cleans up on exit; e.g. closes a
        SQLAlchemy engine).
    """
    for plugin_name in config.plugins:
        mod = importlib.import_module(plugin_name)
        rv = mod.install()
        if hasattr(rv, "__aenter__"):
            bindings = await stack.enter_async_context(rv)
        elif inspect.iscoroutine(rv):
            bindings = await rv
        else:
            bindings = rv
        if not isinstance(bindings, HookBindings):
            raise TypeError(
                f"{plugin_name}.install() must return HookBindings, got "
                f"{type(bindings).__name__}"
            )
        register(bindings)


async def _reconcile_resources() -> None:
    """Refresh registrar-tracked records for every live resource, then purge
    anything the sweep didn't touch. Runs once at startup after plugins load
    so registrars don't drop grants on resources that outlived their TTL.
    """
    refs: list[ResourceRef] = []

    # Nodes (always present).
    for node in await NODE_REGISTRY.list_nodes_async():
        refs.append(ResourceRef(kind=ResourceKind.NODE.value, id=node.id))

    # Workers and workflows live on the root node.
    if WORKER_REGISTRY is not None:
        for worker in await WORKER_REGISTRY.list_workers_async():
            refs.append(ResourceRef(kind=ResourceKind.WORKER.value, id=worker.id))
    if WORKFLOW_REGISTRY is not None:
        workflow_ids = await WORKFLOW_REGISTRY.get_workflow_ids_async()
        for workflow_id in workflow_ids:
            record = await WORKFLOW_REGISTRY.get_workflow_record_async(workflow_id)
            if record is None:
                continue
            refs.append(ResourceRef(kind=ResourceKind.WORKFLOW.value, id=workflow_id))
            for task_id in record.task_ids:
                refs.append(ResourceRef(kind=ResourceKind.TASK.value, id=task_id))

    logger.info("Startup reconcile: %d live resource(s)", len(refs))
    await reconcile_resources(refs, logger)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    async with AsyncExitStack() as plugin_stack:
        await _load_plugins(plugin_stack)

        # --- System principal resolution ---
        system_principal = await resolve_system_principal(
            config.identity.api_key, logger
        )
        app.state.system_principal = system_principal

        # --- Root-only startup ---
        if IS_ROOT_NODE:
            await rehydrate_root_state(RUNTIME, RESIDENT_CONTROL, RESIDENT_REGISTRY)
            if RESIDENT_BRIDGE is not None and NODE_REGISTRY is not None:
                _bridge = RESIDENT_BRIDGE
                _nodes = NODE_REGISTRY

                async def _pump_resident_bridge() -> None:
                    # Bridge each attached node's up stream to its peers' down streams;
                    # a non-blocking sweep, so poll all nodes on a short interval.
                    while True:
                        try:
                            for _node in await _nodes.list_nodes_async():
                                await _bridge.pump_node(_node.id)
                        except asyncio.CancelledError:
                            return
                        except Exception:
                            logger.exception("resident relay bridge pump failed")
                        await asyncio.sleep(0.05)

                app.state.resident_bridge_task = asyncio.create_task(
                    _pump_resident_bridge()
                )
            if PORT_FORWARD_SERVICE is not None:
                await PORT_FORWARD_SERVICE.start()
            _start_root_threads()
            if EVENT_MONITOR is not None:
                EVENT_MONITOR.start()

        # --- Supervisor (all nodes with worker management) ---
        if SUPERVISOR is not None:
            await SUPERVISOR.start(system_principal)

            def _on_node_id_change(new_node_id: str) -> None:
                app.state.node_id = new_node_id
                # Tell EventMonitor which node this server belongs to so that it can
                # wait for the supervisor's SV_UNREGISTER event on shutdown.
                if EVENT_MONITOR is not None:
                    EVENT_MONITOR.set_own_node(new_node_id)

            # Keep the node_id updated for request auth scope + shutdown
            # self-identification.
            _on_node_id_change(SUPERVISOR.node_id)
            SUPERVISOR.add_node_id_listener(_on_node_id_change)

        # --- Startup reconcile ---
        # Runs after the supervisor handshake so this node is in NODE_REGISTRY
        # and is included in the live batch.
        await _reconcile_resources()

        try:
            yield
        finally:
            # --- Supervisor shutdown ---
            if SUPERVISOR is not None:
                await SUPERVISOR.stop()
                app.state.node_id = None

            # --- Event monitor shutdown ---
            if EVENT_MONITOR is not None:
                await EVENT_MONITOR.stop()

            # --- Root-only shutdown ---
            _stop_background()
            _bridge_task = getattr(app.state, "resident_bridge_task", None)
            if _bridge_task is not None:
                _bridge_task.cancel()
                try:
                    await _bridge_task
                except (asyncio.CancelledError, Exception):
                    pass
            if RESIDENT_CONTROL is not None:
                RESIDENT_CONTROL.shutdown()
            if AGENT_MODEL_GATEWAY is not None:
                AGENT_MODEL_GATEWAY.shutdown()
            if FABRIC_TOOL_BROKER is not None:
                FABRIC_TOOL_BROKER.shutdown()
            if PORT_FORWARD_SERVICE is not None:
                await PORT_FORWARD_SERVICE.stop()


app.router.lifespan_context = _lifespan

# --------------------------------------------------------------------------- #
# App state & routers
# --------------------------------------------------------------------------- #

# Shared state (all nodes)
app.state.logger = logger
app.state.node_role = NODE_ROLE
app.state.node_registry = NODE_REGISTRY
app.state.metrics_recorder = METRICS_RECORDER
app.state.redis_client = REDIS_CLIENT
app.state.results_dir = RESULTS_DIR
app.state.supervisor = SUPERVISOR
# resolved during lifespan startup
app.state.node_id = None
app.state.system_principal = None

# Root-only state (None on worker nodes)
app.state.runtime = RUNTIME
app.state.dispatcher = DISPATCHER
app.state.workflow_registry = WORKFLOW_REGISTRY
app.state.worker_registry = WORKER_REGISTRY
app.state.watchdog = WATCHDOG
app.state.event_monitor = EVENT_MONITOR
app.state.port_forward = PORT_FORWARD_SERVICE
app.state.ssh_audit = SSH_AUDIT_SERVICE
app.state.ssh_proxy_enabled = config.port_forward.ssh_proxy_enabled and IS_ROOT_NODE
app.state.serve_proxy_enabled = config.port_forward.serve_proxy_enabled and IS_ROOT_NODE
app.state.resident_control = RESIDENT_CONTROL
app.state.network_plane = NETWORK_PLANE

# Routers — shared
app.include_router(health.router)
app.include_router(docs.router)

v1_prefix = "/api/v1"

# Routers — root only
if IS_ROOT_NODE:
    app.include_router(v1.workflows.router, prefix=v1_prefix)
    app.include_router(v1.workers.router, prefix=v1_prefix)
    app.include_router(v1.nodes.router, prefix=v1_prefix)
    app.include_router(v1.tasks.router, prefix=v1_prefix)
    app.include_router(v1.results.router, prefix=v1_prefix)
    app.include_router(v1.ssh.router, prefix=v1_prefix)
    app.include_router(v1.serve.router, prefix=v1_prefix)
    app.include_router(v1.resident.router, prefix=v1_prefix)
    app.include_router(v1.network.router, prefix=v1_prefix)
    app.include_router(v1.system.router, prefix=v1_prefix)
    app.include_router(v1.traces.router, prefix=v1_prefix)
    if AGENT_MODEL_GATEWAY is not None:
        # The agent-model gateway's Responses API surface a harness provider targets.
        app.include_router(build_agent_model_router(AGENT_MODEL_GATEWAY))

# Routers — supervisor (any node with worker management)
if config.worker_management.enabled:
    app.include_router(v1.stack.router, prefix=v1_prefix)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FlowMesh server.")
    parser.add_argument(
        "--host",
        help="Bind address (defaults to SERVER_APP_HOST env or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port (defaults to SERVER_APP_PORT, then PORT env, else 8000).",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable auto-reload.",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload.",
    )
    parser.set_defaults(reload=None)
    parser.add_argument(
        "--log-level",
        help="Uvicorn log level (defaults to LOG_LEVEL).",
    )

    args = parser.parse_args(argv)

    host_value = args.host or config.http.host
    port_value = args.port if args.port is not None else config.http.port

    reload_enabled = config.http.reload if args.reload is None else args.reload
    log_level = args.log_level or config.http.log_level

    uvicorn_app = app if not reload_enabled else "server.main:app"

    uvicorn.run(
        uvicorn_app,
        host=host_value,
        port=port_value,
        reload=reload_enabled,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
