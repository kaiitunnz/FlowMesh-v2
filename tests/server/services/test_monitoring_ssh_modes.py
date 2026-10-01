"""Choosing the mode an SSH session is served in.

`forward` falls back to `proxy`. A relayed session no mode can carry — the proxy
disabled, or no network plane to relay it — is reported as `direct` at the address
it listens on, with its scope; only a session with no address fails.
"""

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from server.services.monitoring import EventMonitor


@dataclass
class _Monitor:
    monitor: EventMonitor
    dispatcher: MagicMock
    relay: MagicMock | None


def _make_monitor(
    *,
    relay: bool = True,
    ssh_proxy_enabled: bool = True,
    port_forward: MagicMock | None = None,
) -> _Monitor:
    runtime, dispatcher = MagicMock(), MagicMock()
    runtime.get_record.return_value = None
    ssh_relay = MagicMock() if relay else None
    monitor = EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.ssh_modes"),
        runtime=runtime,
        dispatcher=dispatcher,
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        ssh_proxy_enabled=ssh_proxy_enabled,
        ssh_relay=ssh_relay,
        port_forward=port_forward,
        server_base_url="http://server.example.com:8000",
    )
    return _Monitor(monitor, dispatcher, ssh_relay)


def _payload(mode: str, **over: Any) -> dict[str, Any]:
    ssh = {
        "session_id": "ssn-1",
        "mode": mode,
        "username": "flowmesh",
        "host": "127.0.0.1",
        "port": 2222,
        "directHost": "127.0.0.1",
        "directPort": 2222,
        "directScope": "loopback",
        "workerId": "wkr-1",
    }
    ssh.update(over)
    return {"ssh": ssh}


def _update(world: _Monitor, mode: str, **over: Any) -> dict[str, Any]:
    payload = _payload(mode, **over)
    return world.monitor._handle_ssh_task_update("tsk-1", "wkr-1", payload)["ssh"]


def test_a_proxy_session_stays_proxy_where_the_relay_runs() -> None:
    assert _update(_make_monitor(), "proxy")["mode"] == "proxy"


@pytest.mark.parametrize(
    "world",
    [_make_monitor(relay=False), _make_monitor(ssh_proxy_enabled=False)],
    ids=["no-network-plane", "proxy-disabled"],
)
def test_an_unservable_proxy_session_degrades_to_its_own_address(
    world: _Monitor,
) -> None:
    ssh = _update(world, "proxy")
    assert (ssh["mode"], ssh["host"], ssh["port"]) == ("direct", "127.0.0.1", 2222)
    assert ssh["directScope"] == "loopback" and ssh["workerId"] == "wkr-1"
    assert "directHost" not in ssh and "directPort" not in ssh
    world.dispatcher.fail_task.assert_not_called()


def test_a_forward_session_without_a_listener_falls_back_to_proxy() -> None:
    assert _update(_make_monitor(port_forward=None), "forward")["mode"] == "proxy"


@pytest.mark.parametrize("proxy", [True, False])
def test_a_failed_forward_registration_falls_back(proxy: bool) -> None:
    port_forward = MagicMock()
    port_forward.register_port_forward.side_effect = RuntimeError("no ports left")
    world = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=proxy)
    assert _update(world, "forward")["mode"] == ("proxy" if proxy else "direct")


def test_a_registered_forward_session_reports_the_listener() -> None:
    port_forward = MagicMock()
    port_forward.register_port_forward.return_value = {
        **_payload("forward")["ssh"],
        "host": "server.example.com",
        "port": 32001,
    }
    ssh = _update(_make_monitor(port_forward=port_forward), "forward")
    assert (ssh["host"], ssh["port"]) == ("server.example.com", 32001)
    assert (ssh["directHost"], ssh["directPort"]) == ("127.0.0.1", 2222)


def test_a_session_with_no_address_fails_its_task() -> None:
    world = _make_monitor(relay=False)
    payload = world.monitor._handle_ssh_task_update(
        "tsk-1", "wkr-1", _payload("proxy", host=None)
    )
    assert "ssh" not in payload
    world.dispatcher.fail_task.assert_called_once()


def test_a_direct_session_passes_through() -> None:
    monitor = _make_monitor().monitor
    payload = _payload("direct", directScope="network")
    assert monitor._handle_ssh_task_update("tsk-1", "wkr-1", payload) == payload


def test_releasing_a_task_ends_its_relayed_connections() -> None:
    world = _make_monitor()
    world.monitor._release_task("tsk-1")
    assert world.relay is not None
    world.relay.close_task.assert_called_once_with("tsk-1")
