import base64
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from server.config import ServerConfig
from worker.config import WorkerConfig

_SERVER_CA = b"-----BEGIN CERTIFICATE-----\nserver\n"
_OTLP_CA = b"-----BEGIN CERTIFICATE-----\nroot\n"


@pytest.fixture(autouse=True)
def _telemetry_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "coarse")


@pytest.fixture
def ca_files(tmp_path: Path) -> tuple[Path, Path]:
    server_ca = tmp_path / "server-ca.pem"
    otlp_ca = tmp_path / "root-ca.pem"
    server_ca.write_bytes(_SERVER_CA)
    otlp_ca.write_bytes(_OTLP_CA)
    return server_ca, otlp_ca


def test_the_server_verifies_an_https_collector_with_the_server_ca(
    monkeypatch: pytest.MonkeyPatch, ca_files: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://localhost:4317")
    monkeypatch.setenv("SERVER_GRPC_TLS_CA_FILE", str(ca_files[0]))
    monkeypatch.delenv("SERVER_METRICS_OTLP_CA_FILE", raising=False)

    assert ServerConfig.from_env().telemetry.otlp_ca_pem == _SERVER_CA


def test_the_otlp_ca_file_overrides_the_server_ca(
    monkeypatch: pytest.MonkeyPatch, ca_files: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://root:4317")
    monkeypatch.setenv("SERVER_GRPC_TLS_CA_FILE", str(ca_files[0]))
    monkeypatch.setenv("SERVER_METRICS_OTLP_CA_FILE", str(ca_files[1]))

    assert ServerConfig.from_env().telemetry.otlp_ca_pem == _OTLP_CA


def test_a_plaintext_collector_reads_no_ca(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("SERVER_GRPC_TLS_CA_FILE", str(tmp_path / "missing.pem"))

    assert ServerConfig.from_env().telemetry.otlp_ca_pem is None


def test_a_missing_collector_ca_fails_with_its_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://localhost:4317")
    monkeypatch.setenv("SERVER_GRPC_TLS_CA_FILE", str(tmp_path / "missing.pem"))

    with pytest.raises(RuntimeError, match="OTLP collector CA"):
        ServerConfig.from_env()


@pytest.mark.parametrize(
    ("level", "traces", "metrics"),
    [("off", "true", "true"), ("coarse", "false", "false")],
)
def test_a_server_exporting_nothing_reads_no_collector_ca(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    level: str,
    traces: str,
    metrics: str,
) -> None:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", level)
    monkeypatch.setenv("SERVER_METRICS_TRACES_ENABLED", traces)
    monkeypatch.setenv("SERVER_METRICS_METRICS_ENABLED", metrics)
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://localhost:4317")
    monkeypatch.setenv("SERVER_GRPC_TLS_CA_FILE", str(tmp_path / "missing.pem"))

    assert ServerConfig.from_env().telemetry.otlp_ca_pem is None


@pytest.mark.parametrize(("level", "imports"), [("off", True), ("coarse", False)])
def test_the_supervisor_reads_the_collector_ca_only_to_export(
    tmp_path: Path, level: str, imports: bool
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "SERVER_METRICS_TELEMETRY_LEVEL": level,
        "SERVER_METRICS_OTLP_ENDPOINT": "https://root:4317",
        "SERVER_METRICS_OTLP_CA_FILE": str(tmp_path / "not-copied-yet.pem"),
    }
    result = subprocess.run(  # nosec B603 - argv list, no shell
        [sys.executable, "-c", "import server.env"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is imports, result.stderr


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "localhost:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.setenv("WORKER_TOKEN", uuid.uuid4().hex)
    monkeypatch.setenv("WORKER_ALIAS", "w0")
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://root:4317")
    monkeypatch.setenv(
        "SUPERVISOR_GRPC_TLS_CA_B64", base64.b64encode(_SERVER_CA).decode()
    )
    monkeypatch.delenv("SERVER_METRICS_OTLP_CA_B64", raising=False)


@pytest.mark.usefixtures("worker_env")
def test_a_worker_verifies_its_collector_with_the_supervisor_ca() -> None:
    assert WorkerConfig.from_env().telemetry.otlp_ca_pem == _SERVER_CA


@pytest.mark.usefixtures("worker_env")
def test_a_worker_prefers_the_forwarded_otlp_ca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SERVER_METRICS_OTLP_CA_B64", base64.b64encode(_OTLP_CA).decode()
    )

    assert WorkerConfig.from_env().telemetry.otlp_ca_pem == _OTLP_CA


@pytest.mark.usefixtures("worker_env")
def test_a_worker_exporting_nothing_reads_no_collector_ca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "off")
    monkeypatch.setenv("SERVER_METRICS_OTLP_CA_B64", "not base64!")

    assert WorkerConfig.from_env().telemetry.otlp_ca_pem is None
