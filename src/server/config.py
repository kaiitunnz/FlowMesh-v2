import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from shared.tasks.specs import ModelBindingMode
from shared.utils.parsing import parse_bool_env, parse_float_env, parse_int_env


def _default_selection_strategy() -> str:
    # Lazy import: the resident package pulls in the orchestration chain, which imports
    # this module, so referencing the canonical default at call time breaks the cycle.
    from .resident.selection import DEFAULT_SELECTION_STRATEGY

    return DEFAULT_SELECTION_STRATEGY


class NodeRole(StrEnum):
    ROOT = "root"
    WORKER = "worker"


@dataclass
class LoggingConfig:
    file: str = "server.log"
    level: str = "INFO"
    max_bytes: int = 5_242_880
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        return cls(
            file=os.getenv("LOG_FILE", "server.log"),
            level=(
                os.getenv("SERVER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
            ).upper(),
            max_bytes=parse_int_env("LOG_MAX_BYTES", 5_242_880),
            backup_count=parse_int_env("LOG_BACKUP_COUNT", 5),
        )


@dataclass
class RedisConfig:
    control_url: str = "redis://localhost:6379/0"
    telemetry_url: str = "redis://localhost:6379/0"
    acl_enabled: bool = False
    username: str = "admin"
    password: str = ""
    tls_ca_file: str | None = None

    @classmethod
    def from_env(cls) -> "RedisConfig":
        redis_url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        tls_raw = os.getenv("REDIS_TLS_CA_FILE", "").strip()
        return cls(
            control_url=os.getenv("REDIS_CONTROL_URL") or redis_url,
            telemetry_url=os.getenv("REDIS_TELEMETRY_URL") or redis_url,
            acl_enabled=parse_bool_env("REDIS_ACL_ENABLED", False),
            username=os.getenv("REDIS_USERNAME", "admin"),
            password=os.getenv("REDIS_PASSWORD", ""),
            tls_ca_file=tls_raw or None,
        )


@dataclass
class HttpConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "HttpConfig":
        port_default = parse_int_env("PORT", 8000)
        return cls(
            host=os.getenv("SERVER_APP_HOST", "0.0.0.0"),
            port=parse_int_env(
                "SERVER_APP_PORT", parse_int_env("SERVER_HTTP_PORT", port_default)
            ),
            reload=parse_bool_env("SERVER_APP_RELOAD", False),
            log_level=(
                os.getenv("SERVER_APP_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "info"
            ).lower(),
        )


@dataclass
class GrpcConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    tls_ca_file: str = ""
    tls_cert_file: str = ""
    tls_key_file: str = ""

    @classmethod
    def from_env(cls) -> "GrpcConfig":
        return cls(
            host="0.0.0.0",
            port=int(os.getenv("SERVER_GRPC_PORT") or "50051"),
            tls_ca_file=(os.getenv("SERVER_GRPC_TLS_CA_FILE") or "").strip(),
            tls_cert_file=(os.getenv("SERVER_GRPC_TLS_CERT_FILE") or "").strip(),
            tls_key_file=(os.getenv("SERVER_GRPC_TLS_KEY_FILE") or "").strip(),
        )


@dataclass
class PortForwardConfig:
    enabled: bool = True
    persistent_listeners: bool = True
    ssh_proxy_enabled: bool = True
    ssh_audit_enabled: bool = True
    serve_proxy_enabled: bool = True
    bind_host: str = "0.0.0.0"
    public_host: str = "localhost"
    port_start: int = 32000
    port_end: int = 32100

    @classmethod
    def from_env(cls) -> "PortForwardConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_SERVER_PORT_FORWARD", True),
            persistent_listeners=parse_bool_env("ENABLE_PERSISTENT_PORT_FORWARD", True),
            ssh_proxy_enabled=parse_bool_env("ENABLE_SERVER_SSH_PROXY", True),
            ssh_audit_enabled=parse_bool_env(
                "ENABLE_SERVER_SSH_CONNECTION_AUDIT", True
            ),
            serve_proxy_enabled=parse_bool_env("ENABLE_SERVER_SERVE_PROXY", True),
            bind_host=os.getenv("SERVER_PORT_FORWARD_BIND_HOST", "0.0.0.0").strip(),
            public_host=os.getenv(
                "SERVER_PORT_FORWARD_PUBLIC_HOST", "localhost"
            ).strip(),
            port_start=parse_int_env("SERVER_PORT_FORWARD_PORT_START", 32000),
            port_end=parse_int_env("SERVER_PORT_FORWARD_PORT_END", 32100),
        )


@dataclass
class IdentityConfig:
    role: NodeRole = NodeRole.ROOT
    namespace: str = "flowmesh"
    cluster: str = "cluster"
    alias: str = "node"
    tags: list[str] = field(default_factory=list)
    base_url: str = "http://localhost:8000"
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "IdentityConfig":
        raw_tags = os.getenv("NODE_TAGS") or ""
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        role_raw = (os.getenv("NODE_ROLE") or NodeRole.ROOT.value).strip().lower()
        try:
            role = NodeRole(role_raw)
        except ValueError as exc:
            raise SystemExit(
                f"NODE_ROLE must be one of: root, worker (got {role_raw!r})"
            ) from exc
        return cls(
            role=role,
            namespace=(os.getenv("NODE_NAMESPACE") or "flowmesh"),
            cluster=(os.getenv("NODE_CLUSTER") or "cluster"),
            alias=(os.getenv("NODE_ALIAS") or "node"),
            tags=tags,
            base_url=(os.getenv("FLOWMESH_BASE_URL") or "http://localhost:8000"),
            api_key=os.getenv("FLOWMESH_API_KEY", "").strip(),
        )


@dataclass
class DispatchConfig:
    mode: str = "adaptive"
    worker_selection: str = "best_fit"
    selection_jitter: float = 1e-3
    lambda_inference: float = 0.4
    lambda_training: float = 0.8
    lambda_other: float = 0.5
    enable_task_merge: bool = True
    task_merge_max_batch_size: int = 4
    enable_context_reuse: bool = True
    worker_cache_ttl_sec: int = 3600
    enable_stage_weight_stickiness: bool = False
    no_worker_grace_sec: int = 60

    @classmethod
    def from_env(cls) -> "DispatchConfig":
        return cls(
            mode=os.getenv("ORCHESTRATOR_DISPATCH_MODE", "adaptive"),
            worker_selection=os.getenv("ORCHESTRATOR_WORKER_SELECTION", "best_fit"),
            selection_jitter=parse_float_env("SCHEDULER_SELECTION_JITTER", 1e-3),
            lambda_inference=parse_float_env("SCHEDULER_LAMBDA_INFERENCE", 0.4),
            lambda_training=parse_float_env("SCHEDULER_LAMBDA_TRAINING", 0.8),
            lambda_other=parse_float_env("SCHEDULER_LAMBDA_OTHER", 0.5),
            enable_task_merge=parse_bool_env("ENABLE_TASK_MERGE", True),
            task_merge_max_batch_size=max(
                1, parse_int_env("TASK_MERGE_MAX_BATCH_SIZE", 4)
            ),
            enable_context_reuse=parse_bool_env("ENABLE_CONTEXT_REUSE", True),
            worker_cache_ttl_sec=max(0, parse_int_env("WORKER_CACHE_TTL_SEC", 3600)),
            enable_stage_weight_stickiness=parse_bool_env(
                "ENABLE_STAGE_WEIGHT_STICKINESS", False
            ),
            no_worker_grace_sec=max(0, parse_int_env("TASK_NO_WORKER_GRACE_SEC", 60)),
        )


@dataclass
class WatchdogConfig:
    enabled: bool = True
    check_interval: int = 30
    grace_sec: int = 60
    rehydration_grace_sec: int = 120

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_WORKER_WATCHDOG", True),
            check_interval=max(5, parse_int_env("WORKER_DEATH_CHECK_INTERVAL", 30)),
            grace_sec=max(0, parse_int_env("WORKER_DEATH_GRACE_SEC", 60)),
            rehydration_grace_sec=max(
                0, parse_int_env("WORKER_REHYDRATION_GRACE_SEC", 120)
            ),
        )


@dataclass
class MetricsConfig:
    dir: Path | None = None
    enable_density_plot: bool = False
    density_bucket_sec: int = 60

    @classmethod
    def from_env(cls, results_dir: Path) -> "MetricsConfig":
        metrics_env = os.getenv("SERVER_METRICS_DIR")
        metrics_dir: Path | None
        if metrics_env:
            metrics_dir = Path(metrics_env).expanduser().resolve()
        else:
            metrics_dir = results_dir.parent / "metrics"
        return cls(
            dir=metrics_dir,
            enable_density_plot=parse_bool_env(
                "SERVER_METRICS_ENABLE_DENSITY_PLOT", False
            ),
            density_bucket_sec=max(
                1, parse_int_env("SERVER_METRICS_DENSITY_BUCKET_SEC", 60)
            ),
        )


@dataclass
class WorkerManagementConfig:
    enabled: bool = True
    config_path: str = "configs/worker_config.yaml"
    heartbeat_interval: int = 30

    @classmethod
    def from_env(cls) -> "WorkerManagementConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_SUPERVISOR", True),
            config_path=os.getenv("WORKER_CONFIG_PATH", "configs/worker_config.yaml"),
            heartbeat_interval=int(os.getenv("SERVER_HEARTBEAT_INTERVAL") or "30"),
        )


@dataclass
class LogStreamConfig:
    ttl_sec: int = 3600
    archive_flush_interval_sec: float = 5.0
    archive_flush_max_entries: int = 100

    @classmethod
    def from_env(cls) -> "LogStreamConfig":
        return cls(
            ttl_sec=max(0, parse_int_env("LOG_STREAM_TTL_SEC", 3600)),
            archive_flush_interval_sec=max(
                0.1, parse_float_env("TASK_LOG_ARCHIVE_FLUSH_INTERVAL_SEC", 5.0)
            ),
            archive_flush_max_entries=max(
                1, parse_int_env("TASK_LOG_ARCHIVE_FLUSH_MAX_ENTRIES", 100)
            ),
        )


class GatewayMode(StrEnum):
    """How a mediated model invocation settles at the agent-model gateway."""

    CANNED = "canned"
    ECHO = "echo"
    OPENAI = "openai"
    PROXY = "proxy"


def _env_or_none(name: str) -> str | None:
    """The env var's non-empty, stripped value, else None."""
    return (os.getenv(name) or "").strip() or None


@dataclass
class AgentModelGatewayConfig:
    """The agent-model gateway's upstream binding for mediated model invocations.

    ``mode`` selects how a durable model invocation settles: ``canned`` and ``echo`` are
    deterministic and credential-free; ``openai`` forwards to an OpenAI-compatible
    endpoint; ``proxy`` targets an upstream Responses API, streaming a harness's own
    model turns and settling a deferred facade with a single-shot call.
    """

    mode: GatewayMode = GatewayMode.CANNED
    url: str | None = None
    model: str | None = None
    timeout_sec: float = 60.0

    @classmethod
    def from_env(cls) -> "AgentModelGatewayConfig":
        prefix = "AGENT_MODEL_GATEWAY_"
        raw_mode = (os.getenv(f"{prefix}MODE") or "canned").strip().lower()
        try:
            mode = GatewayMode(raw_mode)
        except ValueError:
            mode = GatewayMode.CANNED
        return cls(
            mode=mode,
            url=_env_or_none(f"{prefix}URL"),
            model=_env_or_none(f"{prefix}MODEL"),
            timeout_sec=parse_float_env(f"{prefix}TIMEOUT_SEC") or 60.0,
        )


# Deployment gateway modes map to the per-workflow default binding mode; the codex
# reasoning passthrough (``proxy``) defaults a new binding to an external openai one.
_DEFAULT_BINDING_MODE = {
    "canned": ModelBindingMode.CANNED,
    "echo": ModelBindingMode.ECHO,
    "openai": ModelBindingMode.OPENAI,
    "proxy": ModelBindingMode.OPENAI,
}


@dataclass
class AgentBindingConfig:
    """Deployment defaults for per-workflow agent harness and managed-model bindings.

    The model defaults read ``AGENT_MODEL_GATEWAY_*``. A workflow supplies its own
    inline credential; the deployment holds none.
    """

    default_backend: str | None = None
    default_version: str | None = None
    default_mode: ModelBindingMode | None = None
    default_url: str | None = None
    default_model: str | None = None

    @classmethod
    def from_env(cls) -> "AgentBindingConfig":
        prefix = "AGENT_MODEL_GATEWAY_"
        raw_mode = (os.getenv(f"{prefix}MODE") or "").strip().lower()
        return cls(
            default_backend=_env_or_none("AGENT_HARNESS_DEFAULT_BACKEND"),
            default_version=_env_or_none("AGENT_HARNESS_DEFAULT_VERSION"),
            default_mode=_DEFAULT_BINDING_MODE.get(raw_mode),
            default_url=_env_or_none(f"{prefix}URL"),
            default_model=_env_or_none(f"{prefix}MODEL"),
        )


@dataclass
class ModelSecretVaultConfig:
    """The durable vault backstop TTL for user-supplied model credentials.

    The TTL is sliding: a workflow that keeps resolving its credential keeps it, while
    an abandoned, idle submission expires after this window. Purge on a workflow's
    terminal transition is the primary reclaim; this backstop bounds the rest.
    """

    ttl_sec: int = 86400

    @classmethod
    def from_env(cls) -> "ModelSecretVaultConfig":
        return cls(ttl_sec=parse_int_env("AGENT_MODEL_SECRET_TTL_SEC") or 86400)


@dataclass
class ResidentCapacityConfig:
    """Resident-capacity control admission and materialization knobs.

    ``substrate`` selects the serving stand-in a materialized replica runs: ``serve`` is
    a GPU vLLM replica; ``dev_model`` is the GPU-free stand-in. ``allowed_models`` empty
    permits any plan-derived model; a non-empty list enforces an explicit catalog.
    ``admission_slots`` is the conservative safe-slot count reported per replica.
    ``forward_api_key`` is the credential the in-server adapter presents to a replica
    that reports none, so the ``dev_model`` stand-in can forward it to a keyed upstream.
    """

    enabled: bool = False
    substrate: str = "serve"
    access_mode: str = "forward"
    admission_slots: int = 8
    max_replicas_per_family: int = 1
    max_concurrent_cold_starts: int = 1
    cold_start_deadline_sec: float = 300.0
    poll_interval_sec: float = 1.0
    serve_ttl_sec: float | None = None
    allowed_models: tuple[str, ...] = ()
    forward_api_key: str | None = None
    selection_strategy: str = field(default_factory=_default_selection_strategy)
    idle_retain_sec: float = 0.0
    idle_sweep_interval_sec: float = 30.0

    @classmethod
    def from_env(cls) -> "ResidentCapacityConfig":
        prefix = "RESIDENT_"
        raw_allowed = _env_or_none(f"{prefix}ALLOWED_MODELS")
        allowed = (
            tuple(m.strip() for m in raw_allowed.split(",") if m.strip())
            if raw_allowed
            else ()
        )
        substrate = (
            os.getenv(f"{prefix}INFERENCE_SUBSTRATE") or "serve"
        ).strip().lower() or "serve"
        access = (
            os.getenv(f"{prefix}SERVE_ACCESS_MODE") or "forward"
        ).strip().lower() or "forward"
        default_strategy = _default_selection_strategy()
        strategy = (
            os.getenv(f"{prefix}SELECTION_STRATEGY") or default_strategy
        ).strip().lower() or default_strategy
        return cls(
            enabled=parse_bool_env(f"{prefix}CAPACITY_ENABLED", False),
            substrate=substrate,
            access_mode=access,
            admission_slots=max(1, parse_int_env(f"{prefix}ADMISSION_SLOTS") or 8),
            max_replicas_per_family=max(
                1, parse_int_env(f"{prefix}MAX_REPLICAS_PER_FAMILY") or 1
            ),
            max_concurrent_cold_starts=max(
                1, parse_int_env(f"{prefix}MAX_COLD_STARTS") or 1
            ),
            cold_start_deadline_sec=parse_float_env(f"{prefix}COLD_START_DEADLINE_SEC")
            or 300.0,
            poll_interval_sec=parse_float_env(f"{prefix}POLL_INTERVAL_SEC") or 1.0,
            serve_ttl_sec=parse_float_env(f"{prefix}SERVE_TTL_SEC"),
            allowed_models=allowed,
            forward_api_key=_env_or_none(f"{prefix}FORWARD_API_KEY"),
            selection_strategy=strategy,
            idle_retain_sec=parse_float_env(f"{prefix}IDLE_RETAIN_SEC") or 0.0,
            idle_sweep_interval_sec=parse_float_env(f"{prefix}IDLE_SWEEP_INTERVAL_SEC")
            or 30.0,
        )


@dataclass
class WebSearchConfig:
    """The fabric web-search tool's provider binding and bounds.

    ``provider`` selects the backend (keyless ``duckduckgo`` default); ``api_key`` is
    the deployment credential a keyed provider needs. ``max_calls`` bounds one episode's
    searches; ``result_char_cap`` bounds the injected result size.
    """

    provider: str = "duckduckgo"
    api_key: str | None = None
    max_results: int = 5
    timeout_sec: float = 20.0
    result_char_cap: int = 6000
    max_calls: int = 8
    max_parallel: int = 4

    @classmethod
    def from_env(cls) -> "WebSearchConfig":
        prefix = "WEB_SEARCH_"
        return cls(
            provider=(os.getenv(f"{prefix}PROVIDER") or "duckduckgo").strip().lower(),
            api_key=_env_or_none(f"{prefix}API_KEY"),
            max_results=parse_int_env(f"{prefix}MAX_RESULTS") or 5,
            timeout_sec=parse_float_env(f"{prefix}TIMEOUT_SEC") or 20.0,
            result_char_cap=parse_int_env(f"{prefix}RESULT_CHAR_CAP") or 6000,
            max_calls=parse_int_env(f"{prefix}MAX_CALLS") or 8,
            max_parallel=parse_int_env(f"{prefix}MAX_PARALLEL_CALLS_PER_TURN") or 4,
        )


@dataclass
class OrchestrationConfig:
    max_scope_depth: int | None = None
    max_loop_iterations: int | None = None
    max_activations: int | None = None
    max_spawns_per_turn: int | None = None
    max_spawns_per_region: int | None = None
    episode_lowering: bool = False
    agent_input_budget_bytes: int = 262_144
    gateway: AgentModelGatewayConfig = field(default_factory=AgentModelGatewayConfig)
    agent_binding: AgentBindingConfig = field(default_factory=AgentBindingConfig)
    model_secret_vault: ModelSecretVaultConfig = field(
        default_factory=ModelSecretVaultConfig
    )
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    resident: ResidentCapacityConfig = field(default_factory=ResidentCapacityConfig)

    @classmethod
    def from_env(cls) -> "OrchestrationConfig":
        return cls(
            max_scope_depth=parse_int_env("ORCHESTRATOR_MAX_SCOPE_DEPTH"),
            max_loop_iterations=parse_int_env("ORCHESTRATOR_MAX_LOOP_ITERATIONS"),
            max_activations=parse_int_env("ORCHESTRATOR_MAX_ACTIVATIONS"),
            max_spawns_per_turn=parse_int_env("ORCHESTRATOR_MAX_SPAWNS_PER_TURN"),
            max_spawns_per_region=parse_int_env("ORCHESTRATOR_MAX_SPAWNS_PER_REGION"),
            episode_lowering=parse_bool_env("ORCHESTRATOR_EPISODE_LOWERING", False),
            agent_input_budget_bytes=parse_int_env(
                "ORCHESTRATOR_AGENT_INPUT_BUDGET_BYTES"
            )
            or 262_144,
            gateway=AgentModelGatewayConfig.from_env(),
            agent_binding=AgentBindingConfig.from_env(),
            model_secret_vault=ModelSecretVaultConfig.from_env(),
            web_search=WebSearchConfig.from_env(),
            resident=ResidentCapacityConfig.from_env(),
        )


@dataclass
class ServerConfig:
    logging: LoggingConfig
    redis: RedisConfig
    http: HttpConfig
    grpc: GrpcConfig
    port_forward: PortForwardConfig
    identity: IdentityConfig
    dispatch: DispatchConfig
    watchdog: WatchdogConfig
    metrics: MetricsConfig
    worker_management: WorkerManagementConfig
    log_stream: LogStreamConfig
    orchestration: OrchestrationConfig
    results_dir: Path = Path("./results")
    plugins: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        results_dir = (
            Path(os.getenv("RESULTS_DIR", "").strip() or "./results")
            .expanduser()
            .resolve()
        )
        plugins = [
            p
            for raw in os.getenv("FLOWMESH_PLUGINS", "").split(",")
            if (p := raw.strip())
        ]
        return cls(
            logging=LoggingConfig.from_env(),
            redis=RedisConfig.from_env(),
            http=HttpConfig.from_env(),
            grpc=GrpcConfig.from_env(),
            port_forward=PortForwardConfig.from_env(),
            identity=IdentityConfig.from_env(),
            dispatch=DispatchConfig.from_env(),
            watchdog=WatchdogConfig.from_env(),
            metrics=MetricsConfig.from_env(results_dir),
            worker_management=WorkerManagementConfig.from_env(),
            log_stream=LogStreamConfig.from_env(),
            orchestration=OrchestrationConfig.from_env(),
            results_dir=results_dir,
            plugins=plugins,
        )
