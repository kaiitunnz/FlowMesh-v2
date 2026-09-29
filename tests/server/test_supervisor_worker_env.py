"""A worker's environment carries only the credentials the worker reads."""

import pytest

from server.hooks import PrincipalContext
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
)
from server.supervisor.adapters.vastai import VastAIWorkerAdapter, VastAIWorkerConfig

_OWNER = PrincipalContext(
    principal_id="test-user",
    org_id="test-org",
    external_id="test-user",
    principal_type="user",
    scopes=[],
)


def _docker_worker() -> DockerWorkerAdapter:
    worker = object.__new__(DockerWorkerAdapter)
    worker.config = DockerWorkerConfig(worker_type=WorkerType.CPU)
    worker.token = "worker-token"  # type: ignore[assignment]
    worker.owner = _OWNER
    worker.container_name = "worker-cpu-0"
    return worker


def _vastai_worker() -> VastAIWorkerAdapter:
    worker = object.__new__(VastAIWorkerAdapter)
    worker.config = VastAIWorkerConfig()
    worker.token = "worker-token"  # type: ignore[assignment]
    worker.owner = _OWNER
    return worker


@pytest.mark.parametrize("make_worker", [_docker_worker, _vastai_worker])
@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "GOOGLE_API_KEY"])
def test_worker_env_omits_unread_provider_keys(make_worker, name: str) -> None:
    assert name not in make_worker()._base_environment()
