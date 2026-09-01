"""Route-substrate ids, the network-plane config edge, and the echo wire format."""

import asyncio

from server.config import NetworkPlaneConfig
from server.network import wire
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


def test_config_defaults_to_disabled(monkeypatch) -> None:
    for name in (
        "NETWORK_PLANE_ENABLED",
        "NETWORK_PLANE_ENDPOINT_URL",
        "NETWORK_PLANE_PROTOCOLS",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = NetworkPlaneConfig.from_env()
    assert cfg.enabled is False
    assert cfg.endpoint_url is None
    assert cfg.protocols == ("echo",)


def test_config_parses_env(monkeypatch) -> None:
    monkeypatch.setenv("NETWORK_PLANE_ENABLED", "true")
    monkeypatch.setenv("NETWORK_PLANE_ENDPOINT_URL", "127.0.0.1:41000")
    monkeypatch.setenv("NETWORK_PLANE_SIDECAR_URL", "127.0.0.1:41001")
    monkeypatch.setenv("NETWORK_PLANE_PROTOCOLS", "echo, relay")
    monkeypatch.setenv("NETWORK_PLANE_POSITIVE_TTL_SEC", "12")
    monkeypatch.setenv("NETWORK_PLANE_RELAY_BUFFER_BYTES", "8192")
    cfg = NetworkPlaneConfig.from_env()
    assert cfg.enabled is True
    assert cfg.endpoint_url == "127.0.0.1:41000"
    assert cfg.sidecar_url == "127.0.0.1:41001"
    assert cfg.protocols == ("echo", "relay")
    assert cfg.positive_ttl_sec == 12.0
    assert cfg.relay_buffer_bytes == 8192


def test_wire_frame_round_trip() -> None:
    async def run() -> bytes:
        reader = asyncio.StreamReader()
        payload = b"a framed payload"
        # Build the bytes a writer would produce, then read them back.
        buffer = len(payload).to_bytes(4, "big") + payload
        reader.feed_data(buffer)
        reader.feed_eof()
        return await wire.read_frame(reader)

    assert asyncio.run(run()) == b"a framed payload"
