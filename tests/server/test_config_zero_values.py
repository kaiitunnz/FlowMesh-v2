"""A configured zero reaches the config it names.

Each knob below has a meaning at zero — no cached secret, the minimum slot or buffer
its floor allows, no retained warmth, an uncached route, no backoff, no result — so
reading it must return the zero the deployment set rather than the built-in default.
"""

from collections.abc import Callable
from typing import Any

import pytest

from server.config import (
    ModelSecretVaultConfig,
    NetworkPlaneConfig,
    OrchestrationConfig,
    ResidentCapacityConfig,
    WebSearchConfig,
)


def test_model_secret_ttl_honors_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODEL_SECRET_TTL_SEC", "0")

    assert ModelSecretVaultConfig.from_env().ttl_sec == 0


def test_agent_input_budget_honors_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_INPUT_BUDGET_BYTES", "0")

    assert OrchestrationConfig.from_env().agent_input_budget_bytes == 0


@pytest.mark.parametrize(
    ("env_name", "attribute"),
    [
        ("WEB_SEARCH_MAX_RESULTS", "max_results"),
        ("WEB_SEARCH_RESULT_CHAR_CAP", "result_char_cap"),
        ("WEB_SEARCH_MAX_PARALLEL_CALLS_PER_TURN", "max_parallel"),
    ],
)
def test_web_search_caps_honor_zero(
    monkeypatch: pytest.MonkeyPatch, env_name: str, attribute: str
) -> None:
    monkeypatch.setenv(env_name, "0")

    assert getattr(WebSearchConfig.from_env(), attribute) == 0


@pytest.mark.parametrize(
    ("env_name", "attribute"),
    [
        ("NETWORK_PLANE_POSITIVE_TTL_SEC", "positive_ttl_sec"),
        ("NETWORK_PLANE_NEGATIVE_TTL_SEC", "negative_ttl_sec"),
        ("NETWORK_PLANE_BACKOFF_BASE_SEC", "backoff_base_sec"),
        ("NETWORK_PLANE_BACKOFF_MAX_SEC", "backoff_max_sec"),
        ("NETWORK_PLANE_ROUTE_TTL_SEC", "route_ttl_sec"),
    ],
)
def test_network_plane_timings_honor_zero(
    monkeypatch: pytest.MonkeyPatch, env_name: str, attribute: str
) -> None:
    monkeypatch.setenv(env_name, "0")

    assert getattr(NetworkPlaneConfig.from_env(), attribute) == 0.0


def test_resident_idle_retain_honors_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_IDLE_RETAIN_SEC", "0")

    assert ResidentCapacityConfig.from_env().idle_retain_sec == 0.0


def test_resident_idle_sweep_interval_honors_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESIDENT_IDLE_SWEEP_INTERVAL_SEC", "0")

    assert ResidentCapacityConfig.from_env().idle_sweep_interval_sec == 0.0


@pytest.mark.parametrize(
    ("env_name", "attribute", "floor"),
    [
        ("RESIDENT_ADMISSION_SLOTS", "admission_slots", 1),
        ("RESIDENT_ADAPTER_SLOTS", "adapter_slots", 1),
        ("RESIDENT_MAX_TRANSIENT_REDRIVES", "max_transient_redrives", 1),
    ],
)
def test_resident_floored_counts_settle_at_their_floor_for_zero(
    monkeypatch: pytest.MonkeyPatch, env_name: str, attribute: str, floor: int
) -> None:
    monkeypatch.setenv(env_name, "0")

    assert getattr(ResidentCapacityConfig.from_env(), attribute) == floor


def test_relay_buffer_bytes_settles_at_its_floor_for_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NETWORK_PLANE_RELAY_BUFFER_BYTES", "0")

    assert NetworkPlaneConfig.from_env().relay_buffer_bytes == 1024


@pytest.mark.parametrize(
    ("read_config", "env_name", "attribute", "default"),
    [
        (
            ModelSecretVaultConfig.from_env,
            "AGENT_MODEL_SECRET_TTL_SEC",
            "ttl_sec",
            86400,
        ),
        (WebSearchConfig.from_env, "WEB_SEARCH_MAX_RESULTS", "max_results", 5),
        (
            NetworkPlaneConfig.from_env,
            "NETWORK_PLANE_ROUTE_TTL_SEC",
            "route_ttl_sec",
            30.0,
        ),
        (
            ResidentCapacityConfig.from_env,
            "RESIDENT_ADMISSION_SLOTS",
            "admission_slots",
            8,
        ),
    ],
)
def test_an_unset_or_empty_knob_keeps_its_default(
    monkeypatch: pytest.MonkeyPatch,
    read_config: Callable[[], Any],
    env_name: str,
    attribute: str,
    default: float,
) -> None:
    monkeypatch.delenv(env_name, raising=False)
    assert getattr(read_config(), attribute) == default

    monkeypatch.setenv(env_name, "")
    assert getattr(read_config(), attribute) == default
