"""Shared dummy constructors for test modules."""

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Final
from unittest.mock import patch

from lumid_hooks import PrincipalContext

from shared.content import ContentReference, FabricObjectStore
from shared.tasks import TaskType
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    TaskEnvelopeStrict,
    WorkerHardware,
    WorkerTaskMessage,
)
from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from worker.config import ObjectStoreConfig, WorkerConfig
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import DockerSessionBackend

DEFAULT_WORKER_CONFIG: Final[WorkerConfig] = WorkerConfig(
    owner_principal=PrincipalContext(
        principal_id="test-user",
        org_id="test-org",
        external_id="test-user",
        principal_type="user",
        scopes=["*"],
    ).model_dump(),
    worker_token="test",
    private_state_dir=Path("/tmp/test-private-state"),
    content_dir=Path("/tmp/test-content"),
    content_hydration_enabled=False,
    content_cache_ttl_sec=0.0,
    content_cache_max_bytes=0,
    content_holder_ttl_sec=300.0,
    object_store=ObjectStoreConfig.from_env(Path("/tmp/test-results")),
    content_transfer_timeout_sec=60.0,
    server_base_url=None,
    supervisor_grpc_target="localhost:50051",
    supervisor_grpc_tls_ca_b64=None,
    results_dir=Path("/tmp/test-results"),
    results_mount_source=None,
    hb_interval_sec=30,
    hb_ttl_sec=120,
    hb_file=Path("/tmp/test-hb"),
    namespace="test",
    cluster="test",
    alias="test-worker",
    tags=[],
    log_level="WARNING",
    cost_per_hour=0.0,
    network_bandwidth_bytes_per_sec=None,
    executor_idle_cleanup_sec=None,
    enable_mp_executors=False,
    enable_dev_model=False,
    dev_model_forward_url=None,
    dev_model_response_delay_sec=0.0,
    web_search_provider="duckduckgo",
    web_search_api_key=None,
    model_api_key=None,
    model_egress_timeout_sec=120.0,
    docker_gpu_runtime=None,
    ssh_limits=None,
    enable_ssh_gpu_limit=False,
    telemetry=TelemetryConfig(
        level=TelemetryLevel.OFF,
        traces_enabled=True,
        metrics_enabled=True,
        sample_ratio=1.0,
        otlp_endpoint=None,
    ),
)


def make_worker_config(**overrides: Any) -> WorkerConfig:
    """Build a WorkerConfig with sensible test defaults.

    Override any field via kwargs, e.g.:
        make_worker_config(results_dir=tmp_path / "out", enable_mp_executors=True)
    """
    return replace(DEFAULT_WORKER_CONFIG, **overrides)


def make_live_worker_config(tmp_path: Path, **overrides: Any) -> WorkerConfig:
    """Build a WorkerConfig configured like a live worker — paths under
    ``tmp_path``, MP executors enabled, INFO logs, fixed alias.

    Use this for tests that exercise executor lifecycle (running, heartbeats,
    artifact persistence). For lightweight construction-only tests, call
    ``make_worker_config()`` directly.
    """
    return make_worker_config(
        results_dir=tmp_path / "worker-results",
        hb_file=tmp_path / "worker.hb",
        alias="worker-1",
        log_level="INFO",
        cost_per_hour=1.0,
        executor_idle_cleanup_sec=60.0,
        enable_mp_executors=True,
        **overrides,
    )


def make_ssh_executor(config: WorkerConfig, **kwargs: Any) -> SSHExecutor:
    """Build an SSH executor on the Docker session backend, whether or not a Docker
    daemon is reachable from the test."""
    with patch.object(DockerSessionBackend, "is_available", return_value=True):
        return SSHExecutor(config, **kwargs)


def make_worker_task_message(
    spec: Any,
    task_type: TaskType | None = None,
    task_id: str = "tsk-test",
    workflow_id: str = "wfl-test",
    owner_id: str = "usr-test",
    content_scope: str = "local",
    assigned_worker: str = "wrk-test",
    dispatched_at: str = "2026-04-28T00:00:00Z",
    api_version: str = "flowmesh/v1",
    kind: str = "Task",
    **overrides: Any,
) -> WorkerTaskMessage:
    """Wrap a spec in a WorkerTaskMessage with sensible test defaults."""
    return WorkerTaskMessage(
        task_id=task_id,
        workflow_id=workflow_id,
        owner_id=owner_id,
        content_scope=content_scope,
        assigned_worker=assigned_worker,
        dispatched_at=dispatched_at,
        task_type=task_type,
        task=TaskEnvelopeStrict(apiVersion=api_version, kind=kind, spec=spec),
        **overrides,
    )


def make_worker_hardware(devices: list[GpuInfo] | None = None) -> WorkerHardware:
    """Build a WorkerHardware with sensible test defaults."""
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=2, model="x"),
        memory=MemoryInfo(total_bytes=1024**3),
        gpu=GpuPlatformInfo(
            driver_version=None,
            cuda_version=None,
            devices=devices or [],
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


class FakeContentPlane:
    """A worker content plane over one store, failing its reads while told to."""

    def __init__(self, store: FabricObjectStore) -> None:
        self.store = store
        self.error: Exception | None = None
        self.reads: list[ContentReference] = []
        self.on_read: Callable[[], None] | None = None

    def for_task(self, task_id: str) -> FabricObjectStore:
        return self.store

    def hydrate(self, task_id: str, reference: ContentReference) -> bytes:
        self.reads.append(reference)
        if self.on_read is not None:
            self.on_read()
        if self.error is not None:
            raise self.error
        return self.store.hydrate(reference)


def no_mediated_op(timeout: float) -> None:
    """Stand in for an empty mediated-op queue: wait briefly, then return nothing."""
    time.sleep(min(timeout, 0.01))
    return None
