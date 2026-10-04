"""Route-substrate ids and the network-plane config edge."""

from server.config import NetworkPlaneConfig
from shared.utils.ids import (
    PREFIX_RELAY_SESSION,
    PREFIX_ROUTE_ORIGIN,
    new_relay_session_id,
    new_route_origin_id,
)


def test_route_origin_id_is_prefixed_and_unique() -> None:
    first, second = new_route_origin_id(), new_route_origin_id()
    assert first.startswith(f"{PREFIX_ROUTE_ORIGIN}-")
    assert first != second


def test_relay_session_id_is_prefixed_and_unique() -> None:
    first, second = new_relay_session_id(), new_relay_session_id()
    assert first.startswith(f"{PREFIX_RELAY_SESSION}-")
    assert first != second


def test_config_defaults_to_enabled(monkeypatch) -> None:
    for name in ("NETWORK_PLANE_ENABLED", "NETWORK_PLANE_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    cfg = NetworkPlaneConfig.from_env()
    assert cfg.enabled is True
    assert cfg.endpoint_url is None


def test_config_parses_env(monkeypatch) -> None:
    monkeypatch.setenv("NETWORK_PLANE_ENABLED", "true")
    monkeypatch.setenv("NETWORK_PLANE_ENDPOINT_URL", "127.0.0.1:41000")
    monkeypatch.setenv("NETWORK_PLANE_POSITIVE_TTL_SEC", "12")
    cfg = NetworkPlaneConfig.from_env()
    assert cfg.enabled is True
    assert cfg.endpoint_url == "127.0.0.1:41000"
    assert cfg.positive_ttl_sec == 12.0
