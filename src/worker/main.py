import argparse
import logging
import signal
import socket
from collections.abc import Mapping

from shared._version import FLOWMESH_RELEASE_VERSION
from shared.network.mtls import MutualTlsMaterial, MutualTlsMaterialError
from shared.outcome import FinalizationIndexClient
from shared.schemas.worker import WorkerCapabilities
from shared.tasks.executor_key import ExecutorKey
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerHardware
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.provider import build_meter
from shared.telemetry.semconv import (
    RESOURCE_ROLE,
    SERVICE_NAME,
    SERVICE_VERSION,
    ProcessRole,
    ServiceName,
)

from .config import WorkerConfig
from .content import (
    ContentAccessRegistry,
    ContentLaneHost,
    WorkerContentCache,
    WorkerContentPlane,
)
from .executors import EXECUTOR_REGISTRY, IMPORT_ERRORS, get_executor_class_name
from .executors.base_executor import Executor
from .executors.mp_executor import MPExecutor
from .gpu_sampler import build_gpu_sampler
from .hw import collect_hw
from .lifecycle import Lifecycle
from .power import PowerMonitor
from .runner import Runner
from .supervisor_client import SupervisorClient
from .utils.logging import get_logger

_EXECUTORS_TO_WRAP = {
    ExecutorKey.DEFAULT,
    ExecutorKey.VLLM,
    ExecutorKey.VLLM_LORA,
    ExecutorKey.VLLM_EMBEDDING,
    ExecutorKey.SFT,
    ExecutorKey.LORA_SFT,
    ExecutorKey.IMAGE_CLASSIFICATION_TRAINING,
    ExecutorKey.PPO,
    ExecutorKey.DPO,
    ExecutorKey.DATA_PROFILING,
    ExecutorKey.DATA_RETRIEVAL,
    ExecutorKey.DIFFUSERS,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowMesh worker entrypoint.")
    parser.add_argument(
        "--collect-hw",
        action="store_true",
        help="Print hardware JSON and exit.",
    )
    parser.add_argument(
        "--collect-hw-prefix",
        type=str,
        default=None,
        help="Optional prefix to print before hardware JSON.",
    )
    parser.add_argument(
        "--bandwidth-bytes-per-sec",
        type=float,
        default=None,
        help="Optional network bandwidth in bytes per second.",
    )
    return parser.parse_args()


def initialize_executors(
    config: WorkerConfig,
    hardware: WorkerHardware,
    logger: logging.Logger,
    lifecycle: Lifecycle,
    registry: Mapping[ExecutorKey, type[Executor] | None] | None = None,
    import_errors: dict[str, str] | None = None,
    cuda_available: bool | None = None,
    enable_mp_executors: bool = True,
):
    """Initialize executor registry and handle graceful degradation.

    Allows dependency/GPU overrides in tests; returns (executors, default_executor).
    """

    registry = registry or EXECUTOR_REGISTRY
    import_errors = import_errors or IMPORT_ERRORS

    def check_cuda() -> bool:
        if cuda_available is not None:
            return cuda_available
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    configured_wrapped = _EXECUTORS_TO_WRAP if enable_mp_executors else set()

    def init_executor(key: ExecutorKey, *, gpu_required: bool = False):
        cls = registry.get(key)
        if cls is None:
            reason = import_errors.get(
                get_executor_class_name(key, key), "dependency missing"
            )
            logger.info("Skipping executor %s: %s", key, reason)
            return None

        if gpu_required and not check_cuda():
            logger.info("Executor %s requires a GPU; unavailable, skipping", key)
            return None

        if not cls.is_available(config):
            logger.info("Executor %s is unavailable; skipping", key)
            return None

        try:
            if key in configured_wrapped:
                return MPExecutor(cls, config, hardware)
            return cls(config, hardware, lifecycle)
        except Exception as exc:
            logger.warning("Failed to initialize executor %s: %s", key, exc)
            return None

    executors: dict[ExecutorKey, Executor] = {}
    default_executor = init_executor(ExecutorKey.DEFAULT)
    if default_executor:
        executors[ExecutorKey.DEFAULT] = default_executor

    for key in [
        ExecutorKey.ECHO,
        ExecutorKey.RAG,
        ExecutorKey.AGENT_EPISODE,
        ExecutorKey.SERVICE_LEAF,
        ExecutorKey.DEV_MODEL,
        ExecutorKey.SFT,
        ExecutorKey.LORA_SFT,
        ExecutorKey.IMAGE_CLASSIFICATION_TRAINING,
        ExecutorKey.DATA_PROFILING,
        ExecutorKey.DATA_RETRIEVAL,
        ExecutorKey.DIFFUSERS,
        ExecutorKey.API,
        ExecutorKey.SSH,
    ]:
        inst = init_executor(key)
        if inst:
            executors[key] = inst

    for key in [
        ExecutorKey.VLLM,
        ExecutorKey.VLLM_LORA,
        ExecutorKey.VLLM_EMBEDDING,
        ExecutorKey.VLLM_SERVE,
        ExecutorKey.PPO,
        ExecutorKey.DPO,
        ExecutorKey.OMNI_TEXT2IMAGE,
        ExecutorKey.OMNI_TEXT2SPEECH,
        ExecutorKey.OMNI_TEXT2AUDIO,
        ExecutorKey.OMNI_TEXT2GENERAL,
    ]:
        inst = init_executor(key, gpu_required=True)
        if inst:
            executors[key] = inst

    if not executors:
        raise SystemExit(
            "No executors available. Install at least one executor package."
        )

    if not default_executor:
        default_executor = executors.get(ExecutorKey.ECHO) or executors.get(
            ExecutorKey.API
        )
        if default_executor is None:
            raise SystemExit(
                "No suitable default executor available. "
                "Ensure the echo/api executor can be initialized."
            )
        logger.info(
            "HFTransformers unavailable; using %s as default executor (CPU-only mode)",
            type(default_executor).__name__,
        )

    return executors, default_executor


def build_capabilities(
    executors: dict[ExecutorKey, Executor],
    registry: Mapping[ExecutorKey, type[Executor] | None] | None = None,
    resident_listener_port: int = 0,
) -> WorkerCapabilities:
    registry = registry or EXECUTOR_REGISTRY
    classes = {key: cls for key in executors if (cls := registry.get(key))}
    return WorkerCapabilities(
        supported_task_types=frozenset[TaskType]().union(
            *(cls.supported_task_types for cls in classes.values())
        ),
        merge_batching_executors=frozenset(
            key for key, cls in classes.items() if cls.batches_merged_children
        ),
        resident_listener_port=resident_listener_port,
    )


def _peer_material(
    cfg: WorkerConfig, logger: logging.Logger
) -> MutualTlsMaterial | None:
    """This worker's transient copy of the node's peer TLS material, if configured.

    Mutual TLS is on unless the operator attests a trusted network, so material that is
    absent or unusable is fatal: dialing and serving in plaintext instead would carry
    resident payloads over a wire the deployment asked to protect. Absent material is a
    node misconfiguration rather than a posture, since the supervisor reads the files
    and fails on its own side before it hands this worker their bytes.
    """
    if not cfg.peer_enabled:
        return None
    if cfg.peer_disable_mtls:
        logger.warning(
            "resident peer transports are enabled without mutual TLS: this worker "
            "dials a target on an operator-attested trusted network, proving no "
            "identity to it"
        )
        return None
    if not (cfg.peer_tls_ca_b64 and cfg.peer_tls_cert_b64 and cfg.peer_tls_key_b64):
        raise MutualTlsMaterialError(
            "resident peer transports require mutual TLS material this worker was "
            "not given"
        )
    try:
        return MutualTlsMaterial.from_b64(
            ca_b64=cfg.peer_tls_ca_b64,
            cert_b64=cfg.peer_tls_cert_b64,
            key_b64=cfg.peer_tls_key_b64,
        )
    except MutualTlsMaterialError:
        logger.error("resident peer TLS material is unusable")
        raise


def _bind_peer_listener(cfg: WorkerConfig) -> socket.socket | None:
    """Bind the peer listener so its port is advertised at registration.

    The port is bound before the worker registers and served once the resident lane
    loop comes up, so the address control advertises is the one an origin reaches.
    """
    if not cfg.peer_enabled:
        return None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))  # nosec B104 - an origin dials it from off-host
    sock.listen(16)
    sock.setblocking(False)
    return sock


def _build_content_plane(
    cfg: WorkerConfig, client: SupervisorClient, logger: logging.Logger
) -> WorkerContentPlane | None:
    """The worker's content plane: the shared store, and a cache over it if one runs.

    Content lives in the shared store, so a worker always reaches it — every object it
    writes and every reference it reads resolves there. The cache is the optional half:
    a deployment that enables it also gets a local copy of what this worker wrote and
    can serve a peer from it, and one that does not goes to the store every time.
    """
    access = ContentAccessRegistry(
        cfg.object_store, logger, request_access=client.push_content_access_request
    )
    lane: ContentLaneHost | None = None
    if cfg.content_hydration_enabled:
        lane = ContentLaneHost(
            store=WorkerContentCache(
                cfg.content_dir,
                retain_sec=cfg.content_cache_ttl_sec,
                max_bytes=cfg.content_cache_max_bytes,
            ),
            push_frame=client.push_content_frame,
            request_grant=(
                lambda reference, task_id: client.push_content_hydration_request(
                    reference.model_dump(mode="json"), task_id
                )
            ),
            worker_id=client.worker_id,
            generation=client.incarnation,
            transfer_timeout_sec=cfg.content_transfer_timeout_sec,
            announce=client.push_content_holding,
            holder_report_ttl_sec=cfg.content_holder_ttl_sec,
            logger=logger,
        )
    return WorkerContentPlane(
        lane,
        access,
        announce=client.push_content_holding if lane is not None else None,
        finalizations=(
            FinalizationIndexClient(cfg.server_base_url)
            if cfg.server_base_url
            else None
        ),
        logger=logger,
    )


def main() -> None:
    args = _parse_args()
    if args.collect_hw:
        hardware = collect_hw(bandwidth_bytes_per_sec=args.bandwidth_bytes_per_sec)
        hardware_json = hardware.model_dump_json()
        if args.collect_hw_prefix:
            hardware_json = args.collect_hw_prefix + hardware_json
        print(hardware_json)
        return

    cfg = WorkerConfig.from_env()
    logger = get_logger(name="worker", level=cfg.log_level)

    supervisor_client = SupervisorClient(
        worker_token=cfg.worker_token,
        owner_principal=cfg.owner_principal,
        grpc_target=cfg.supervisor_grpc_target,
        worker_namespace=cfg.namespace,
        worker_cluster=cfg.cluster,
        worker_alias=cfg.alias,
        logger=logger,
        grpc_tls_ca_b64=cfg.supervisor_grpc_tls_ca_b64,
        grpc_keepalive_time_ms=cfg.grpc_keepalive_time_ms,
        grpc_keepalive_timeout_ms=cfg.grpc_keepalive_timeout_ms,
    )

    lifecycle = Lifecycle(
        supervisor_client,
        cfg.hb_interval_sec,
        cfg.hb_ttl_sec,
        cfg.hb_file,
        cost_per_hour=cfg.cost_per_hour,
        power_monitor=PowerMonitor(),
    )
    hardware = collect_hw(bandwidth_bytes_per_sec=cfg.network_bandwidth_bytes_per_sec)
    logger.info("Collected hardware info: %s", hardware)

    def _worker_id_or_none() -> str | None:
        try:
            return lifecycle.worker_id
        except RuntimeError:
            return None

    # No channel currently reports this worker's node id to itself (it is a
    # supervisor-side concept, assigned by the server handshake); GPU metrics
    # carry an empty flowmesh.node_id until one exists.
    gpu_sampler = build_gpu_sampler(
        build_meter(
            cfg.telemetry,
            {
                SERVICE_NAME: ServiceName.WORKER,
                SERVICE_VERSION: FLOWMESH_RELEASE_VERSION,
                RESOURCE_ROLE: ProcessRole.WORKER,
            },
        ),
        node_id=lambda: None,
        worker_id=_worker_id_or_none,
        interval_sec=cfg.telemetry.resource_sample_sec,
        enabled=cfg.telemetry.metrics_enabled
        and cfg.telemetry.emits(TelemetryLevel.COARSE),
    )

    executors, default_executor = initialize_executors(
        cfg,
        hardware,
        logger,
        lifecycle,
        enable_mp_executors=cfg.enable_mp_executors,
    )

    peer_sock = _bind_peer_listener(cfg)
    capabilities = build_capabilities(
        executors,
        resident_listener_port=(
            peer_sock.getsockname()[1] if peer_sock is not None else 0
        ),
    )
    ssh_limits = cfg.ssh_limits
    if TaskType.SSH in capabilities.supported_task_types:
        if ssh_limits is None:
            logger.warning(
                "SSH resource cap not configured; SSH sessions will be able to access "
                "full host resources of this worker."
            )
        else:
            logger.info("SSH resource cap: %s", ssh_limits.model_dump())
    lifecycle.start(
        env={},
        hardware=hardware,
        capabilities=capabilities,
        ssh_limits=ssh_limits,
        tags=cfg.tags,
    )
    gpu_sampler.start()

    lifecycle.start_content_plane(_build_content_plane(cfg, supervisor_client, logger))

    task_stream = supervisor_client.iter_tasks()
    runner = Runner(
        lifecycle,
        task_stream,
        cfg.results_dir,
        hardware,
        executors,
        default_executor,
        logger,
        executor_idle_cleanup_sec=cfg.executor_idle_cleanup_sec,
        web_search_provider=cfg.web_search_provider,
        web_search_api_key=cfg.web_search_api_key,
        model_api_key=cfg.model_api_key,
        model_egress_timeout_sec=cfg.model_egress_timeout_sec,
        peer_enabled=cfg.peer_enabled,
        peer_material=_peer_material(cfg, logger),
        peer_listener_sock=peer_sock,
        telemetry=cfg.telemetry,
    )

    # Install signal handlers to allow graceful shutdown
    def handle_exit_signal(signum: int, _) -> None:
        logger.info("Received exit signal %d; initiating shutdown", signum)
        runner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        try:
            signal.signal(sig, handle_exit_signal)
        except (ValueError, OSError):
            # Signal not supported on this platform
            logger.debug("Signal %s not supported; skipping handler installation", sig)

    try:
        runner.start()
    finally:
        gpu_sampler.shutdown()
        lifecycle.shutdown()


if __name__ == "__main__":
    main()
