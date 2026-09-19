"""SERVER_METRICS_* telemetry vars must reach a worker through the env allowlist.

A var read only by ``src/server/config.py`` never arrives at a worker process; it must
also be threaded through ``WorkerAdapter._base_environment`` (and survive every
adapter's override of it) or worker-side telemetry silently never turns on.
"""

import pytest

from server import env
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


def _set_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "SERVER_METRICS_TELEMETRY_LEVEL", "fine")
    monkeypatch.setattr(env, "SERVER_METRICS_TRACES_ENABLED", True)
    monkeypatch.setattr(env, "SERVER_METRICS_METRICS_ENABLED", False)
    monkeypatch.setattr(env, "SERVER_METRICS_TRACE_SAMPLE_RATIO", 0.5)
    monkeypatch.setattr(env, "SERVER_METRICS_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setattr(env, "SERVER_METRICS_OTLP_TIMEOUT_SEC", 20)
    monkeypatch.setattr(env, "SERVER_METRICS_RESOURCE_SAMPLE_SEC", 30)


def _assert_telemetry_env(environment: dict[str, str]) -> None:
    assert environment["SERVER_METRICS_TELEMETRY_LEVEL"] == "fine"
    assert environment["SERVER_METRICS_TRACES_ENABLED"] == "1"
    assert environment["SERVER_METRICS_METRICS_ENABLED"] == "0"
    assert environment["SERVER_METRICS_TRACE_SAMPLE_RATIO"] == "0.5"
    assert environment["SERVER_METRICS_OTLP_ENDPOINT"] == "http://collector:4317"
    assert environment["SERVER_METRICS_OTLP_TIMEOUT_SEC"] == "20"
    assert environment["SERVER_METRICS_RESOURCE_SAMPLE_SEC"] == "30"


def test_telemetry_env_reaches_docker_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_telemetry_env(monkeypatch)

    _assert_telemetry_env(_docker_worker()._base_environment())


def test_telemetry_env_reaches_vastai_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_telemetry_env(monkeypatch)

    _assert_telemetry_env(_vastai_worker()._base_environment())


def test_telemetry_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "SERVER_METRICS_TELEMETRY_LEVEL", "off")
    monkeypatch.setattr(env, "SERVER_METRICS_OTLP_ENDPOINT", "")

    environment = _docker_worker()._base_environment()

    assert environment["SERVER_METRICS_TELEMETRY_LEVEL"] == "off"
    assert environment["SERVER_METRICS_OTLP_ENDPOINT"] == ""
