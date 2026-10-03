import uuid
from pathlib import Path

import pytest

from worker.config import WorkerConfig


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "localhost:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.setenv("WORKER_TOKEN", uuid.uuid4().hex)
    monkeypatch.setenv("WORKER_ALIAS", "w")
    monkeypatch.delenv("SERVE_DEFAULT_TTL_SEC", raising=False)
    monkeypatch.delenv("SERVE_MAX_TTL_SEC", raising=False)


def test_serve_ttl_defaults() -> None:
    config = WorkerConfig.from_env()
    assert (config.serve_default_ttl_sec, config.serve_max_ttl_sec) == (3600.0, 86400.0)


def test_serve_ttl_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVE_DEFAULT_TTL_SEC", "600")
    monkeypatch.setenv("SERVE_MAX_TTL_SEC", "7200")
    config = WorkerConfig.from_env()
    assert (config.serve_default_ttl_sec, config.serve_max_ttl_sec) == (600.0, 7200.0)
