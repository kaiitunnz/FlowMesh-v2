"""Resident-capacity config resolves its knobs from the RESIDENT_* environment."""

import pytest

from server.config import ResidentCapacityConfig
from server.resident.selection import DEFAULT_SELECTION_STRATEGY

_KEYS = (
    "RESIDENT_SELECTION_STRATEGY",
    "RESIDENT_IDLE_RETAIN_SEC",
    "RESIDENT_IDLE_SWEEP_INTERVAL_SEC",
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
