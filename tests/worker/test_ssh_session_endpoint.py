"""A relayed session's port is published for the worker's relay lane, and only a
relayed session's."""

import asyncio
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.network.byte_stream import ByteStreamChannel
from shared.network.relay_frame import SSH_FRAME_KIND, RelayFrame, RelayFrameKind
from shared.network.session import RelaySessionRole
from tests.worker.factories import make_live_worker_config, make_ssh_executor
from worker.ssh_relay import SshEndpointRegistry, SshRelayLane


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


def test_a_lane_whose_frames_cannot_leave_stops_within_its_timeout() -> None:
    """A worker shutting down gives the lane what is left of its deadline, and a
    push waiting on a reconnecting event stream must not stretch it."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(5)
    released = threading.Event()
    registry = SshEndpointRegistry()
    registry.publish("ssn-1", listener.getsockname()[1])

    def push(wire: dict[str, Any]) -> None:
        # Only its window grants leave; the cancels of a stop wait.
        if RelayFrame.from_wire(wire).kind is not RelayFrameKind.WINDOW:
            released.wait(30)

    lane = SshRelayLane(registry=registry, push_frame=push)
    opening: list[RelayFrame] = []

    async def capture(frame: RelayFrame) -> None:
        opening.append(frame)

    async def open_frame() -> None:
        sink = MagicMock(send=capture)
        await ByteStreamChannel("rly-1", RelaySessionRole.ORIGIN, sink).send_open(
            "ssn-1"
        )

    asyncio.run(open_frame())
    lane.start()
    try:
        lane.route(SSH_FRAME_KIND, opening[0].to_wire())
        listener.accept()[0].close()
        started = time.monotonic()
        lane.stop(0.3)
        assert time.monotonic() - started < 0.5
    finally:
        released.set()
        listener.close()


def test_a_lane_stopped_with_no_time_left_still_ends_its_connections() -> None:
    """A worker's shutdown may have spent its whole deadline by the time it stops the
    lane, and a connection left open would hold its session's sshd."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(5)
    registry = SshEndpointRegistry()
    registry.publish("ssn-1", listener.getsockname()[1])
    lane = SshRelayLane(registry=registry, push_frame=lambda wire: None)
    openings: list[RelayFrame] = []

    async def capture(frame: RelayFrame) -> None:
        openings.append(frame)

    async def open_frames() -> None:
        sink = MagicMock(send=capture)
        for session_id in ("rly-1", "rly-2", "rly-3"):
            await ByteStreamChannel(
                session_id, RelaySessionRole.ORIGIN, sink
            ).send_open("ssn-1")

    asyncio.run(open_frames())
    lane.start()
    accepted: list[socket.socket] = []
    try:
        for frame in openings:
            lane.route(SSH_FRAME_KIND, frame.to_wire())
            accepted.append(listener.accept()[0])

        lane.stop(0)

        for conn in accepted:
            conn.settimeout(5)
            assert conn.recv(1) == b""
    finally:
        for conn in accepted:
            conn.close()
        listener.close()
