"""Tests for server environment configuration."""

from pathlib import Path

import pytest

from server.config import MetricsConfig, PortForwardConfig
from shared.telemetry.config import TelemetryLevel


def test_port_forward_config_enables_capabilities_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENABLE_PERSISTENT_PORT_FORWARD", raising=False)
    monkeypatch.delenv("ENABLE_SERVER_SSH_PROXY", raising=False)

    config = PortForwardConfig.from_env()

    assert config.persistent_listeners is True
    assert config.ssh_proxy_enabled is True


def test_port_forward_config_reads_persistent_listener_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_PERSISTENT_PORT_FORWARD", "false")

    config = PortForwardConfig.from_env()

    assert config.persistent_listeners is False


@pytest.mark.parametrize("ssh_proxy", ["false", "true"])
def test_port_forward_config_reads_ssh_proxy_capability(
    monkeypatch: pytest.MonkeyPatch,
    ssh_proxy: str,
) -> None:
    monkeypatch.setenv("ENABLE_SERVER_SSH_PROXY", ssh_proxy)

    config = PortForwardConfig.from_env()

    assert config.ssh_proxy_enabled is (ssh_proxy == "true")


def test_metrics_config_telemetry_defaults_to_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SERVER_METRICS_TELEMETRY_LEVEL", raising=False)

    config = MetricsConfig.from_env(tmp_path)

    assert config.telemetry.level is TelemetryLevel.OFF
    assert config.telemetry.traces_enabled is True
    assert config.telemetry.metrics_enabled is True
    assert config.telemetry.sample_ratio == 1.0
    assert config.telemetry.otlp_endpoint is None
    assert config.otlp_timeout_sec == 10
    assert config.resource_sample_sec == 15


@pytest.mark.parametrize("level", ["coarse", "fine", "full"])
def test_metrics_config_reads_telemetry_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, level: str
) -> None:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", level)
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("SERVER_METRICS_TRACE_SAMPLE_RATIO", "0.25")

    config = MetricsConfig.from_env(tmp_path)

    assert config.telemetry.level is TelemetryLevel(level)
    assert config.telemetry.otlp_endpoint == "http://collector:4317"
    assert config.telemetry.sample_ratio == 0.25


def test_metrics_config_rejects_unknown_telemetry_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "verbose")

    with pytest.raises(SystemExit):
        MetricsConfig.from_env(tmp_path)
