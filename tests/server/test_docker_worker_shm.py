"""A Docker GPU worker gets a /dev/shm large enough for multi-GPU vLLM engines."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from docker.errors import NotFound

from server.hooks import PrincipalContext
from server.supervisor.adapters import docker as docker_adapter
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
)
from server.supervisor.resource_manager import GpuArch


def _started_with(worker_type: WorkerType) -> dict[str, Any]:
    client = MagicMock()
    client.containers.get.side_effect = NotFound("absent")
    gpu = worker_type is WorkerType.GPU
    adapter = DockerWorkerAdapter(
        token=WorkerTokenType("worker-token"),
        alias="worker_0",
        container_name="worker_0",
        cuda_devices=[0, 1] if gpu else None,
        gpu_arch=GpuArch.HOPPER if gpu else None,
        config=DockerWorkerConfig(
            worker_type=worker_type,
            results_dir="/results",
            hf_cache_dir="/hf-cache",
            enable_ssh=False,
        ),
        docker_client=client,
        owner=PrincipalContext(
            principal_id="test-user",
            org_id="test-org",
            external_id="test-user",
            principal_type="user",
            scopes=[],
        ),
    )
    adapter._hardware = {}
    assert adapter._start() is True
    return dict(client.containers.run.call_args.kwargs)


def test_a_gpu_worker_starts_with_the_gpu_shm_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_adapter.env, "DOCKER_GPU_RUNTIME", None)

    assert _started_with(WorkerType.GPU)["shm_size"] == "8g"


def test_a_cpu_worker_keeps_the_docker_default_shm() -> None:
    assert "shm_size" not in _started_with(WorkerType.CPU)
