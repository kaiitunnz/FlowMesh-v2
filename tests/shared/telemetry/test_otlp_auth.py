"""OTLP export presents the collector's bearer token and verifies a TLS collector."""

import logging

import grpc
import pytest

from shared.telemetry.config import TelemetryConfig
from shared.telemetry.provider import otlp_exporter_kwargs

_ENDPOINT_VARS = ("SERVER_METRICS_OTLP_ENDPOINT", "SERVER_METRICS_OTLP_TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENDPOINT_VARS:
        monkeypatch.delenv(name, raising=False)


def test_the_token_is_parsed_and_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://root:4317")
    monkeypatch.setenv("SERVER_METRICS_OTLP_TOKEN", " otlp-SECRET ")

    config = TelemetryConfig.from_env()

    assert config.otlp_token is not None
    assert config.otlp_token.get_secret_value() == "otlp-SECRET"
    assert "otlp-SECRET" not in repr(config)


def test_a_blank_token_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_TOKEN", "  ")

    assert TelemetryConfig.from_env().otlp_token is None


def test_the_token_rides_as_a_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("SERVER_METRICS_OTLP_TOKEN", "otlp-SECRET")

    kwargs = otlp_exporter_kwargs(TelemetryConfig.from_env())

    assert kwargs["headers"] == (("authorization", "Bearer otlp-SECRET"),)
    assert "credentials" not in kwargs


def test_no_token_sends_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "http://localhost:4317")

    assert "headers" not in otlp_exporter_kwargs(TelemetryConfig.from_env())


def test_an_https_collector_is_verified_with_the_given_ca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", "https://root:4317")
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "coarse")

    kwargs = otlp_exporter_kwargs(TelemetryConfig.from_env(lambda: b"-----CA-----"))

    assert isinstance(kwargs["credentials"], grpc.ChannelCredentials)


@pytest.mark.parametrize(
    ("endpoint", "warns"),
    [
        ("http://10.0.0.5:4317", True),
        ("http://root.example:4317", True),
        ("http://localhost:4317", False),
        ("http://127.0.0.1:4317", False),
        ("https://10.0.0.5:4317", False),
    ],
)
def test_a_token_sent_off_host_over_plaintext_warns_without_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    endpoint: str,
    warns: bool,
) -> None:
    monkeypatch.setenv("SERVER_METRICS_OTLP_ENDPOINT", endpoint)
    monkeypatch.setenv("SERVER_METRICS_OTLP_TOKEN", "otlp-SECRET")

    with caplog.at_level(logging.WARNING, logger="shared.telemetry.config"):
        TelemetryConfig.from_env()

    assert ("plaintext" in caplog.text) is warns
    assert "otlp-SECRET" not in caplog.text
