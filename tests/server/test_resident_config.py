"""Resident-capacity config resolves its knobs from the RESIDENT_* environment."""

import pytest

from server.config import OrchestrationConfig, ResidentCapacityConfig
from server.resident.selection import DEFAULT_SELECTION_STRATEGY

_KEYS = (
    "RESIDENT_SELECTION_STRATEGY",
    "RESIDENT_IDLE_RETAIN_SEC",
    "RESIDENT_IDLE_SWEEP_INTERVAL_SEC",
    "RESIDENT_CAPACITY_ENABLED",
    "NETWORK_PLANE_ENABLED",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_disable_idle_teardown_and_use_the_canonical_strategy():
    cfg = ResidentCapacityConfig.from_env()
    assert cfg.selection_strategy == DEFAULT_SELECTION_STRATEGY
    assert ResidentCapacityConfig().selection_strategy == DEFAULT_SELECTION_STRATEGY
    assert cfg.idle_retain_sec == 0.0
    assert cfg.idle_sweep_interval_sec == 30.0


def test_env_overrides_strategy_and_idle_knobs(monkeypatch):
    monkeypatch.setenv("RESIDENT_SELECTION_STRATEGY", "Least-Load")
    monkeypatch.setenv("RESIDENT_IDLE_RETAIN_SEC", "120")
    monkeypatch.setenv("RESIDENT_IDLE_SWEEP_INTERVAL_SEC", "15")
    cfg = ResidentCapacityConfig.from_env()
    assert cfg.selection_strategy == "least-load"
    assert cfg.idle_retain_sec == 120.0
    assert cfg.idle_sweep_interval_sec == 15.0


def test_resident_capacity_requires_the_network_plane(monkeypatch):
    monkeypatch.setenv("RESIDENT_CAPACITY_ENABLED", "true")
    monkeypatch.setenv("NETWORK_PLANE_ENABLED", "false")
    with pytest.raises(ValueError, match="NETWORK_PLANE_ENABLED"):
        OrchestrationConfig.from_env()


def test_resident_capacity_with_the_network_plane_resolves(monkeypatch):
    monkeypatch.setenv("RESIDENT_CAPACITY_ENABLED", "true")
    monkeypatch.setenv("NETWORK_PLANE_ENABLED", "true")
    cfg = OrchestrationConfig.from_env()
    assert cfg.resident.enabled and cfg.network.enabled
