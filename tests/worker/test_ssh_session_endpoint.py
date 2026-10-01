"""A relayed session's port is published for the worker's relay lane, and only a
relayed session's."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from tests.worker.factories import make_live_worker_config, make_ssh_executor
from worker.ssh_relay import SshEndpointRegistry


def _ready(tmp_path: Path, mode: str) -> tuple[SshEndpointRegistry, dict]:
    registry = SshEndpointRegistry()
    lifecycle = MagicMock(ssh_endpoints=registry)
    executor = make_ssh_executor(make_live_worker_config(tmp_path), lifecycle=lifecycle)
    session = MagicMock()
    session.wait_ready.return_value = 2222
    session.login_user.return_value = "flowmesh"
    executor._wait_session_ready(
        session,
        "ssn-1",
        cast(Any, SimpleNamespace(task_id="tsk-1", assigned_worker="wkr-1")),
        cast(Any, SimpleNamespace(access_mode=mode, ttl_sec=60)),
    )
    update = lifecycle.notify_task_update.call_args.args[1]["ssh"]
    return registry, update


@pytest.mark.parametrize("mode", ["proxy", "forward"])
def test_a_relayed_session_publishes_its_port_and_its_own_route(
    tmp_path: Path, mode: str
) -> None:
    registry, update = _ready(tmp_path, mode)
    assert registry.resolve("ssn-1") == 2222
    assert not any(key.startswith("_") for key in update)
    # The Docker backend publishes the port on every host address.
    assert update["directScope"] == "network"
    assert (update["directHost"], update["directPort"]) == (update["host"], 2222)
    assert update["workerId"] == "wkr-1"


def test_a_direct_session_publishes_nothing(tmp_path: Path) -> None:
    registry, _ = _ready(tmp_path, "direct")
    assert registry.resolve("ssn-1") is None


def test_withdrawing_tells_every_listener_and_forgets_the_port() -> None:
    registry = SshEndpointRegistry()
    seen: list[str] = []
    registry.add_withdraw_listener(seen.append)
    registry.publish("ssn-1", 2222)
    registry.withdraw("ssn-1")
    registry.withdraw("ssn-never")
    assert registry.resolve("ssn-1") is None
    assert seen == ["ssn-1", "ssn-never"]


def test_a_direct_session_is_advertised_at_the_configured_host(tmp_path: Path) -> None:
    executor = make_ssh_executor(
        make_live_worker_config(tmp_path, ssh_direct_host="ssh.example.com"),
        lifecycle=MagicMock(ssh_endpoints=SshEndpointRegistry()),
    )
    assert executor.backend.session_address("direct") == "ssh.example.com"
