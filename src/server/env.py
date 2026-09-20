import base64
import os
import tempfile
from pathlib import Path

from shared.tools.search.schema import DEFAULT_SEARCH_PROVIDER
from shared.utils import parse_bool_env, parse_float_env, parse_int_env


def _read_file_b64(path: str, what: str) -> str:
    """One operator-configured TLS file, base64-encoded for a transient copy."""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {what}: {exc}") from exc


NODE_NAMESPACE: str = os.getenv("NODE_NAMESPACE") or "flowmesh"
NODE_CLUSTER: str = os.getenv("NODE_CLUSTER") or "cluster"
NODE_ALIAS: str = os.getenv("NODE_ALIAS") or "node"
NODE_TAGS: list[str] = [
    t.strip() for t in (os.getenv("NODE_TAGS") or "").split(",") if t.strip()
]
SERVER_BIND_HOST: str = "0.0.0.0"
SERVER_LOCAL_HOST: str = "localhost"
SERVER_HOST: str = os.getenv("SERVER_HOST") or SERVER_LOCAL_HOST
SERVER_APP_PORT: int = int(
    os.getenv("SERVER_APP_PORT")
    or os.getenv("SERVER_HTTP_PORT")
    or os.getenv("PORT")
    or "8000"
)
SERVER_GRPC_PORT: int = int(os.getenv("SERVER_GRPC_PORT") or "50051")

SERVER_GRPC_TLS_CA_FILE: str = (os.getenv("SERVER_GRPC_TLS_CA_FILE") or "").strip()
SERVER_GRPC_TLS_CERT_FILE: str = (os.getenv("SERVER_GRPC_TLS_CERT_FILE") or "").strip()
SERVER_GRPC_TLS_KEY_FILE: str = (os.getenv("SERVER_GRPC_TLS_KEY_FILE") or "").strip()

SUPERVISOR_GRPC_DISABLE_SERVER_TLS: bool = parse_bool_env(
    "SUPERVISOR_GRPC_DISABLE_SERVER_TLS", False
)
SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS: bool = parse_bool_env(
    "SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", True
)
SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS: int = parse_int_env(
    "SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS", 60000
)
SUPERVISOR_GRPC_EXTERNAL_PORT: int | None = parse_int_env(
    "SUPERVISOR_GRPC_EXTERNAL_PORT"
)

if SERVER_GRPC_TLS_CERT_FILE or SERVER_GRPC_TLS_KEY_FILE:
    if not (SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE):
        raise RuntimeError(
            "SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE are required"
        )
    if not SERVER_GRPC_TLS_CA_FILE:
        raise RuntimeError("SERVER_GRPC_TLS_CA_FILE is required for server TLS")
    SERVER_GRPC_TLS_CA_B64 = _read_file_b64(
        SERVER_GRPC_TLS_CA_FILE, "server TLS CA file"
    )
else:
    SERVER_GRPC_TLS_CA_B64 = ""

FLOWMESH_BASE_URL: str = os.getenv("FLOWMESH_BASE_URL", "http://localhost:8000")
FLOWMESH_API_KEY: str = os.getenv("FLOWMESH_API_KEY", "")

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ACL_ENABLED = parse_bool_env("REDIS_ACL_ENABLED", False)
REDIS_USERNAME: str = os.getenv("REDIS_USERNAME", "admin")
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
REDIS_CONTROL_URL: str = os.getenv("REDIS_CONTROL_URL", REDIS_URL)
REDIS_TELEMETRY_URL: str = os.getenv("REDIS_TELEMETRY_URL", REDIS_URL)
REDIS_TLS_CA_FILE: str = os.getenv("REDIS_TLS_CA_FILE", "").strip()

SERVER_HEARTBEAT_INTERVAL: int = int(os.getenv("SERVER_HEARTBEAT_INTERVAL") or "30")
SERVER_HEARTBEAT_TTL: int = max(SERVER_HEARTBEAT_INTERVAL * 4, 120)
ENABLE_SSH_BY_DEFAULT: bool = parse_bool_env("ENABLE_SSH_BY_DEFAULT", False)
SSH_DEFAULT_IMAGE: str | None = os.getenv("SSH_DEFAULT_IMAGE", "").strip() or None
SSH_DEFAULT_USER: str | None = os.getenv("SSH_DEFAULT_USER", "").strip() or None
SSH_DEFAULT_TTL_SEC: float | None = parse_float_env("SSH_DEFAULT_TTL_SEC")
SSH_DEFAULT_IDLE_SEC: float | None = parse_float_env("SSH_DEFAULT_IDLE_SEC")
SSH_MAX_TTL_SEC: float | None = parse_float_env("SSH_MAX_TTL_SEC")
SSH_POLL_INTERVAL_SEC: float | None = parse_float_env("SSH_POLL_INTERVAL_SEC")
SSH_STOP_TIMEOUT_SEC: float | None = parse_float_env("SSH_STOP_TIMEOUT_SEC")
SSH_MAX_CPU: float | None = parse_float_env("SSH_MAX_CPU")
SSH_MAX_MEMORY: str | None = os.getenv("SSH_MAX_MEMORY", "").strip() or None
SSH_MAX_PIDS: int | None = parse_int_env("SSH_MAX_PIDS")
ENABLE_SSH_GPU_LIMIT: bool = parse_bool_env("ENABLE_SSH_GPU_LIMIT", False)

LOG_FILE: str = os.getenv("LOG_FILE", "server.log")
LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 5_242_880))
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 5))
LOG_LEVEL: str = (
    os.getenv("SERVER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
).upper()

FLOWMESH_REGISTRY: str = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
FLOWMESH_VERSION: str = os.getenv("FLOWMESH_VERSION", "latest")
SERVER_CUDA_PROBE_IMAGE: str = os.getenv(
    "SERVER_CUDA_PROBE_IMAGE", "nvidia/cuda:12.9.1-base-ubuntu24.04"
)
DOCKER_GPU_RUNTIME: str | None = os.getenv("DOCKER_GPU_RUNTIME", "").strip() or None

# Projected to the worker so a worker-hosted external-tool executor builds its provider
# and reads a keyed provider's credential from its own local environment.
WEB_SEARCH_PROVIDER: str = (
    os.getenv("WEB_SEARCH_PROVIDER", "").strip() or DEFAULT_SEARCH_PROVIDER
)
WEB_SEARCH_API_KEY: str = os.getenv("WEB_SEARCH_API_KEY", "")

# The deployment-global credential a worker uses to egress an external managed-model
# boundary, read only in the worker that performs the egress.
AGENT_MODEL_API_KEY: str = os.getenv("AGENT_MODEL_API_KEY", "")

# The bound a held model turn's worker waits for its one-use egress permit.
AGENT_MODEL_EGRESS_TIMEOUT_SEC: str = os.getenv("AGENT_MODEL_EGRESS_TIMEOUT_SEC", "")

WORKER_CONFIG_PATH: str = os.getenv("WORKER_CONFIG_PATH", "configs/worker_config.yaml")
CUDA_VISIBLE_DEVICES: str | None = os.getenv("CUDA_VISIBLE_DEVICES")
if CUDA_VISIBLE_DEVICES is not None:
    if CUDA_VISIBLE_DEVICES.strip().lower() == "all":
        CUDA_VISIBLE_DEVICES = None
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "").strip() or "./results"
WORKER_RESULTS_DIR: str = (
    os.getenv("WORKER_RESULTS_DIR", "").strip() or "flowmesh_results"
)
WORKER_PRIVATE_STATE_DIR: str = os.getenv("WORKER_PRIVATE_STATE_DIR", "").strip()
WORKER_CONTENT_DIR: str = os.getenv("WORKER_CONTENT_DIR", "").strip()
CONTENT_HYDRATION_ENABLED: bool = parse_bool_env("CONTENT_HYDRATION_ENABLED", False)
CONTENT_CACHE_TTL_SEC: float = parse_float_env("CONTENT_CACHE_TTL_SEC", 900.0)
CONTENT_HOLDER_TTL_SEC: float = parse_float_env("CONTENT_HOLDER_TTL_SEC", 300.0)
CONTENT_STORE_BACKEND: str = os.getenv("CONTENT_STORE_BACKEND", "s3").strip()
CONTENT_STORE_ENDPOINT_URL: str = os.getenv("CONTENT_STORE_ENDPOINT_URL", "").strip()
CONTENT_STORE_PORT: str = os.getenv("CONTENT_STORE_PORT", "").strip()
CONTENT_STORE_BUCKET: str = os.getenv("CONTENT_STORE_BUCKET", "").strip()
CONTENT_STORE_PREFIX: str = os.getenv("CONTENT_STORE_PREFIX", "").strip()
CONTENT_STORE_REGION: str = os.getenv("CONTENT_STORE_REGION", "").strip()
CONTENT_STORE_ACCESS_KEY: str = os.getenv("CONTENT_STORE_ACCESS_KEY", "").strip()
CONTENT_STORE_SECRET_KEY: str = os.getenv("CONTENT_STORE_SECRET_KEY", "").strip()
CONTENT_STORE_FILESYSTEM_ROOT: str = os.getenv(
    "CONTENT_STORE_FILESYSTEM_ROOT", ""
).strip()
CONTENT_TRANSFER_TIMEOUT_SEC: float = parse_float_env(
    "CONTENT_TRANSFER_TIMEOUT_SEC", 60.0
)
HF_CACHE_DIR: str | None = os.getenv("HF_CACHE_DIR") or None
PREDOWNLOAD_MODEL_LIST: str = os.getenv("PREDOWNLOAD_MODEL_LIST", "")
WORKER_TAGS: str = os.getenv("WORKER_TAGS", "")
WORKER_HB_DIR: str = os.getenv("WORKER_HB_DIR") or os.path.join(
    tempfile.gettempdir(), "flowmesh_worker_health"
)
WORKER_UPLOAD_RESULTS: bool = parse_bool_env("WORKER_UPLOAD_RESULTS", False)
WORKER_EXECUTOR_IDLE_CLEANUP_SEC: float = parse_float_env(
    "WORKER_EXECUTOR_IDLE_CLEANUP_SEC", 60
)
WORKER_ENABLE_DEV_MODEL: bool = parse_bool_env("WORKER_ENABLE_DEV_MODEL", False)
DEV_MODEL_FORWARD_URL: str = os.getenv("DEV_MODEL_FORWARD_URL", "")
DEV_MODEL_RESPONSE_DELAY_SEC: float = (
    parse_float_env("DEV_MODEL_RESPONSE_DELAY_SEC") or 0.0
)

VAST_SEARCH_LIMIT: int = int(os.getenv("VAST_SEARCH_LIMIT") or "10")
VAST_MAX_RETRIES: int = int(os.getenv("VAST_MAX_RETRIES") or "1")

NEBULA_API_BASE_URL: str = os.getenv("NEBULA_API_BASE_URL", "")

SERVER_METRICS_TELEMETRY_LEVEL: str = (
    (os.getenv("SERVER_METRICS_TELEMETRY_LEVEL") or "off").strip().lower()
)
SERVER_METRICS_TRACES_ENABLED: bool = parse_bool_env(
    "SERVER_METRICS_TRACES_ENABLED", True
)
SERVER_METRICS_METRICS_ENABLED: bool = parse_bool_env(
    "SERVER_METRICS_METRICS_ENABLED", True
)
SERVER_METRICS_TRACE_SAMPLE_RATIO: float = parse_float_env(
    "SERVER_METRICS_TRACE_SAMPLE_RATIO", 1.0
)
SERVER_METRICS_OTLP_ENDPOINT: str = (
    os.getenv("SERVER_METRICS_OTLP_ENDPOINT") or ""
).strip()
SERVER_METRICS_OTLP_TIMEOUT_SEC: int = parse_int_env(
    "SERVER_METRICS_OTLP_TIMEOUT_SEC", 10
)
SERVER_METRICS_RESOURCE_SAMPLE_SEC: int = parse_int_env(
    "SERVER_METRICS_RESOURCE_SAMPLE_SEC", 15
)


NETWORK_PLANE_PEER_ENABLED: bool = parse_bool_env("NETWORK_PLANE_PEER_ENABLED", False)
NETWORK_PLANE_PEER_DISABLE_MTLS: bool = parse_bool_env(
    "NETWORK_PLANE_PEER_DISABLE_MTLS", False
)


def _peer_material_b64(var: str) -> str:
    """A worker's transient copy of one peer TLS file, base64-encoded.

    The operator configures the material as files on the node; a worker runs in its own
    container, so the supervisor hands it the bytes rather than a path it cannot read.
    Material the node cannot read is fatal here rather than handed over absent, so a
    worker never dials in plaintext on a deployment that asked for mutual TLS.
    """
    if not NETWORK_PLANE_PEER_ENABLED or NETWORK_PLANE_PEER_DISABLE_MTLS:
        return ""
    path = os.getenv(var, "").strip()
    if not path:
        raise RuntimeError(f"{var} is required unless peer mutual TLS is disabled")
    return _read_file_b64(path, var)


NETWORK_PLANE_PEER_TLS_CA_B64: str = _peer_material_b64(
    "NETWORK_PLANE_PEER_TLS_CA_FILE"
)
NETWORK_PLANE_PEER_TLS_CERT_B64: str = _peer_material_b64(
    "NETWORK_PLANE_PEER_TLS_CERT_FILE"
)
NETWORK_PLANE_PEER_TLS_KEY_B64: str = _peer_material_b64(
    "NETWORK_PLANE_PEER_TLS_KEY_FILE"
)
