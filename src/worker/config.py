# worker/config.py
"""Configuration loader for the Worker process.

This module encapsulates all environment-derived configuration so the rest of the worker
code can depend on a structured config object.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.schemas.worker import SSHLimits
from shared.tools.search.schema import DEFAULT_SEARCH_PROVIDER
from shared.utils.parsing import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_mem_to_bytes,
)

from .utils.health import get_hb_config


@dataclass(frozen=True)
class WorkerConfig:
    worker_token: str
    owner_principal: dict[str, Any] | None
    server_base_url: str | None
    supervisor_grpc_target: str
    supervisor_grpc_tls_ca_b64: str | None
    results_dir: Path
    results_mount_source: str | None
    hb_interval_sec: int
    hb_ttl_sec: int
    hb_file: Path
    namespace: str
    cluster: str
    alias: str
    tags: list[str]
    log_level: str
    cost_per_hour: float
    network_bandwidth_bytes_per_sec: float | None
    executor_idle_cleanup_sec: float | None
    enable_mp_executors: bool
    enable_dev_model: bool
    dev_model_forward_url: str | None
    dev_model_response_delay_sec: float
    web_search_provider: str
    web_search_api_key: str | None
    model_api_key: str | None
    model_egress_timeout_sec: float
    docker_gpu_runtime: str | None
    ssh_limits: SSHLimits | None
    enable_ssh_gpu_limit: bool
    grpc_keepalive_time_ms: int | None = None
    grpc_keepalive_timeout_ms: int | None = None
    network_mode: str | None = None
    container_name: str | None = None
    ssh_network_name: str | None = None
    peer_enabled: bool = False
    peer_disable_mtls: bool = False
    peer_tls_ca_b64: str | None = None
    peer_tls_cert_b64: str | None = None
    peer_tls_key_b64: str | None = None

    @staticmethod
    def from_env() -> "WorkerConfig":
        worker_token = os.getenv("WORKER_TOKEN", "").strip()
        if not worker_token:
            raise SystemExit("WORKER_TOKEN is required")

        owner_principal: dict[str, Any] | None = None
        if owner_principal_json := os.getenv("WORKER_OWNER_PRINCIPAL_JSON", "").strip():
            try:
                loaded = json.loads(owner_principal_json)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(loaded, dict):
                    owner_principal = loaded

        server_base_url = os.getenv("FLOWMESH_BASE_URL", "").strip() or None

        supervisor_grpc_target = (os.getenv("SUPERVISOR_GRPC_TARGET") or "").strip()
        if not supervisor_grpc_target:
            raise SystemExit("SUPERVISOR_GRPC_TARGET is required")

        supervisor_grpc_tls_ca_b64: str | None = (
            os.getenv("SUPERVISOR_GRPC_TLS_CA_B64") or ""
        ).strip() or None

        # The node's peer material reaches a worker base64-encoded; the operator
        # configures it as files on the node it came from.
        peer_prefix = "NETWORK_PLANE_PEER_"
        peer_enabled = parse_bool_env(f"{peer_prefix}ENABLED", False)
        peer_disable_mtls = parse_bool_env(f"{peer_prefix}DISABLE_MTLS", False)
        peer_tls_ca_b64 = (os.getenv(f"{peer_prefix}TLS_CA_B64") or "").strip() or None
        peer_tls_cert_b64 = (
            os.getenv(f"{peer_prefix}TLS_CERT_B64") or ""
        ).strip() or None
        peer_tls_key_b64 = (
            os.getenv(f"{peer_prefix}TLS_KEY_B64") or ""
        ).strip() or None

        results_dir = Path(
            os.getenv("RESULTS_DIR", "").strip() or "./results"
        ).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)

        hb_interval, hb_ttl, hb_file = get_hb_config()

        namespace = os.getenv("WORKER_NAMESPACE", "flowmesh").strip()
        cluster = os.getenv("WORKER_CLUSTER", "cluster").strip()
        container_name = os.getenv("WORKER_CONTAINER_NAME", "").strip() or None
        ssh_network_name = os.getenv("SSH_NETWORK_NAME", "").strip() or None
        alias = os.getenv("WORKER_ALIAS", "").strip() or os.urandom(8).hex()
        tags = [t.strip() for t in os.getenv("WORKER_TAGS", "").split(",") if t.strip()]

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        cost_per_hour = parse_float_env("WORKER_COST_PER_HOUR", 1.0)
        if cost_per_hour < 0:
            raise SystemExit("WORKER_COST_PER_HOUR must be non-negative")

        network_bandwidth_bytes_per_sec = parse_float_env(
            "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC", None
        )
        if (
            network_bandwidth_bytes_per_sec is not None
            and network_bandwidth_bytes_per_sec <= 0
        ):
            raise SystemExit("WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC must be positive")

        enable_mp_executors = parse_bool_env("WORKER_ENABLE_MP_EXECUTORS", True)
        enable_dev_model = parse_bool_env("WORKER_ENABLE_DEV_MODEL", False)
        dev_model_forward_url = os.getenv("DEV_MODEL_FORWARD_URL", "").strip() or None
        dev_model_response_delay_sec = (
            parse_float_env("DEV_MODEL_RESPONSE_DELAY_SEC") or 0.0
        )
        web_search_provider = (
            (os.getenv("WEB_SEARCH_PROVIDER") or DEFAULT_SEARCH_PROVIDER)
            .strip()
            .lower()
        )
        web_search_api_key = os.getenv("WEB_SEARCH_API_KEY", "").strip() or None
        model_api_key = os.getenv("AGENT_MODEL_API_KEY", "").strip() or None
        model_egress_timeout_sec = parse_float_env(
            "AGENT_MODEL_EGRESS_TIMEOUT_SEC", 120.0
        )
        docker_gpu_runtime = os.getenv("DOCKER_GPU_RUNTIME", "").strip() or None
        grpc_keepalive_time_ms = parse_int_env(
            "SUPERVISOR_GRPC_KEEPALIVE_TIME_MS", 300_000
        )
        grpc_keepalive_timeout_ms = parse_int_env(
            "SUPERVISOR_GRPC_KEEPALIVE_TIMEOUT_MS", 10_000
        )
        results_mount_source = os.getenv("RESULTS_MOUNT_SOURCE", "").strip() or None
        network_mode = os.getenv("WORKER_NETWORK_MODE", "").strip() or None
        executor_idle_cleanup_sec = parse_float_env(
            "WORKER_EXECUTOR_IDLE_CLEANUP_SEC", 60
        )

        ssh_max_cpu = parse_float_env("SSH_MAX_CPU")
        if ssh_max_cpu is not None and ssh_max_cpu <= 0:
            raise SystemExit("SSH_MAX_CPU must be positive")
        ssh_max_memory_raw = os.getenv("SSH_MAX_MEMORY", "").strip() or None
        ssh_max_memory_bytes: int | None = None
        if ssh_max_memory_raw is not None:
            ssh_max_memory_bytes = parse_mem_to_bytes(ssh_max_memory_raw)
            if ssh_max_memory_bytes is None or ssh_max_memory_bytes <= 0:
                raise SystemExit(
                    f"SSH_MAX_MEMORY value {ssh_max_memory_raw!r} is not a valid "
                    "memory string (e.g. '8Gi', '512Mi', or a positive byte count)"
                )
        ssh_max_pids = parse_int_env("SSH_MAX_PIDS")
        if ssh_max_pids is not None and ssh_max_pids <= 0:
            raise SystemExit("SSH_MAX_PIDS must be positive")
        ssh_limits = (
            None
            if (
                ssh_max_cpu is None
                and ssh_max_memory_bytes is None
                and ssh_max_pids is None
            )
            else SSHLimits(
                max_cpu_cores=ssh_max_cpu,
                max_memory_bytes=ssh_max_memory_bytes,
                max_pids=ssh_max_pids,
            )
        )
        enable_ssh_gpu_limit = parse_bool_env("ENABLE_SSH_GPU_LIMIT", False)

        return WorkerConfig(
            worker_token=worker_token,
            owner_principal=owner_principal,
            server_base_url=server_base_url,
            supervisor_grpc_target=supervisor_grpc_target,
            supervisor_grpc_tls_ca_b64=supervisor_grpc_tls_ca_b64,
            peer_enabled=peer_enabled,
            peer_disable_mtls=peer_disable_mtls,
            peer_tls_ca_b64=peer_tls_ca_b64,
            peer_tls_cert_b64=peer_tls_cert_b64,
            peer_tls_key_b64=peer_tls_key_b64,
            results_dir=results_dir,
            results_mount_source=results_mount_source,
            hb_interval_sec=hb_interval,
            hb_ttl_sec=hb_ttl,
            hb_file=hb_file,
            namespace=namespace,
            cluster=cluster,
            alias=alias,
            tags=tags,
            log_level=log_level,
            cost_per_hour=cost_per_hour,
            network_bandwidth_bytes_per_sec=network_bandwidth_bytes_per_sec,
            executor_idle_cleanup_sec=executor_idle_cleanup_sec,
            enable_mp_executors=enable_mp_executors,
            enable_dev_model=enable_dev_model,
            dev_model_forward_url=dev_model_forward_url,
            dev_model_response_delay_sec=dev_model_response_delay_sec,
            web_search_provider=web_search_provider,
            web_search_api_key=web_search_api_key,
            model_api_key=model_api_key,
            model_egress_timeout_sec=model_egress_timeout_sec,
            docker_gpu_runtime=docker_gpu_runtime,
            ssh_limits=ssh_limits,
            enable_ssh_gpu_limit=enable_ssh_gpu_limit,
            grpc_keepalive_time_ms=grpc_keepalive_time_ms,
            grpc_keepalive_timeout_ms=grpc_keepalive_timeout_ms,
            network_mode=network_mode,
            container_name=container_name,
            ssh_network_name=ssh_network_name,
        )
