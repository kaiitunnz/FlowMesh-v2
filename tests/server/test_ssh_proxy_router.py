"""The SSH WebSocket proxy relays only a running task's relayed session, and always
releases its relay connection when the client goes."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.routers.v1 import ssh as ssh_router
from server.task.models import TaskStatus
from shared.network.byte_stream import StreamClosed

PREFIX = "/api/v1"


class _Channel:
    """Echoes what the client sends until the relay is aborted."""

    def __init__(self) -> None:
        self.received: asyncio.Queue[bytes] = asyncio.Queue()
        self.aborted = False

    async def send(self, data: bytes) -> None:
        await self.received.put(data)

    async def recv(self) -> bytes | None:
        return await self.received.get()

    async def abort(self) -> None:
        self.aborted = True


def _record(**ssh: Any) -> MagicMock:
    record = MagicMock()
    record.task_id = "tsk-abc"
    record.status = TaskStatus.DISPATCHED
    record.assigned_worker = "wkr-1"
    record.workflow_id = "wfl-1"
    record.latest_update = {"ssh": {"mode": "proxy", "session_id": "ssn-1", **ssh}}
    return record


def _make_app(record: MagicMock, relay: Any) -> FastAPI:
    runtime = MagicMock()
    runtime.get_record.return_value = record
    worker_registry = MagicMock()
    worker_registry.get_worker_async = AsyncMock(
        return_value=SimpleNamespace(id="wkr-1", node_id="nde-1")
    )
    app = FastAPI()
    app.state.runtime = runtime
    app.state.logger = logging.getLogger("test.ssh_proxy_router")
    app.state.ssh_proxy_enabled = True
    app.state.worker_registry = worker_registry
    app.state.ssh_connections = None
    app.state.ssh_relay = relay
    app.include_router(ssh_router.router, prefix=PREFIX)
    return app


def _relay(channel: _Channel) -> MagicMock:
    relay = MagicMock()
    relay.open = AsyncMock(return_value=channel)
    relay.release = AsyncMock()
    return relay


def test_bytes_are_relayed_and_the_connection_is_released_on_disconnect() -> None:
    channel = _Channel()
    relay = _relay(channel)
    client = TestClient(_make_app(_record(), relay))
    with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as websocket:
        websocket.send_bytes(b"keystroke")
        assert websocket.receive_bytes() == b"keystroke"
    target = relay.open.call_args.args[0]
    assert (target.worker_id, target.node_id, target.endpoint_id) == (
        "wkr-1",
        "nde-1",
        "ssn-1",
    )
    relay.release.assert_awaited_once_with(channel, abort=True)


def test_a_forward_session_is_reachable_through_the_proxy() -> None:
    relay = _relay(_Channel())
    client = TestClient(_make_app(_record(mode="forward"), relay))
    with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as websocket:
        websocket.send_bytes(b"x")
        assert websocket.receive_bytes() == b"x"


@pytest.mark.parametrize(
    "record",
    [
        _record(mode="direct"),
        _record(session_id=None),
        MagicMock(
            task_id="tsk-abc",
            status=TaskStatus.DONE,
            assigned_worker="wkr-1",
            latest_update={"ssh": {"mode": "proxy", "session_id": "ssn-1"}},
        ),
    ],
    ids=["direct-mode", "no-session", "not-running"],
)
def test_a_session_the_relay_cannot_reach_is_refused_without_opening(
    record: MagicMock,
) -> None:
    relay = _relay(_Channel())
    client = TestClient(_make_app(record, relay))
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as ws:
            ws.receive_bytes()
    assert excinfo.value.code == 1011
    relay.open.assert_not_awaited()


def test_a_server_without_the_network_plane_refuses_the_proxy() -> None:
    client = TestClient(_make_app(_record(), None))
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as ws:
            ws.receive_bytes()
    assert excinfo.value.code == 4403


def test_an_aborted_relay_closes_the_websocket() -> None:
    channel = _Channel()

    async def closed() -> bytes | None:
        raise StreamClosed("relay stream cancelled")

    channel.recv = closed  # type: ignore[method-assign]
    relay = _relay(channel)
    client = TestClient(_make_app(_record(), relay))
    with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as websocket:
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_bytes()
    relay.release.assert_awaited_once()


def test_a_relayed_connection_is_listed_while_it_lasts() -> None:
    channel = _Channel()
    connections = MagicMock()
    connections.register_connection = AsyncMock()
    connections.unregister_connection = AsyncMock()
    app = _make_app(_record(username="flowmesh"), _relay(channel))
    app.state.ssh_connections = connections
    with TestClient(app).websocket_connect(
        f"{PREFIX}/ssh/tasks/tsk-abc/proxy"
    ) as websocket:
        websocket.send_bytes(b"x")
        assert websocket.receive_bytes() == b"x"
        info = connections.register_connection.call_args.args[0]
        assert (info.access_mode, info.task_id, info.session_id, info.username) == (
            "proxy",
            "tsk-abc",
            "ssn-1",
            "flowmesh",
        )
    connections.unregister_connection.assert_awaited_once_with(info.connection_id)
