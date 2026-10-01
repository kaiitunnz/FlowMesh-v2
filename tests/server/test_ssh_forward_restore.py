"""A forward SSH session is served again on its published port after a root restart,
and each of its connections is relayed to its worker's current node."""

import asyncio
import socket
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from server.services.port_forward import PortForwardService, forward_sessions
from server.ssh import SshRelayTarget
from server.task.models import TaskStatus
from tests.server.test_ssh_forward_ports import _free_port_range, _make_service


def _session(task_id: str, port: int) -> tuple[str, str, str, dict[str, Any]]:
    endpoint = {
        "session_id": f"ssn-{task_id}",
        "username": "flowmesh",
        "mode": "forward",
        "host": "lum.id",
        "port": port,
    }
    return (task_id, "wfl-1", "wkr-1", endpoint)


def _port_of(svc: PortForwardService, task_id: str) -> int | None:
    return next((p for p, t in svc._port_to_task.items() if t == task_id), None)


@pytest.mark.anyio
@pytest.mark.parametrize("persistent", [True, False])
async def test_a_restored_session_keeps_its_published_port(persistent: bool) -> None:
    start, end = _free_port_range(3)
    svc = _make_service(start, end, persistent_listeners=persistent)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", end)])

        assert _port_of(svc, "tsk-1") == end
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_a_session_whose_port_was_taken_stays_unreachable() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end, persistent_listeners=False)
    taken = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    taken.bind(("127.0.0.1", end))
    taken.listen()
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", end)])

        assert _port_of(svc, "tsk-1") is None
        assert "tsk-1" not in svc._sessions
    finally:
        await svc.stop()
        taken.close()


@pytest.mark.anyio
async def test_a_connection_is_relayed_to_the_workers_current_node() -> None:
    start, end = _free_port_range(1)
    svc = _make_service(start, end)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", start)])
        registry = cast(Any, svc._worker_registry)
        registry.get_worker_async = AsyncMock(
            return_value=SimpleNamespace(node_id="nde-rebound")
        )
        opened: list[SshRelayTarget] = []

        def record_open(target: SshRelayTarget) -> None:
            opened.append(target)
            raise RuntimeError("no relay in this test")

        cast(Any, svc._relay).open = AsyncMock(side_effect=record_open)

        _reader, writer = await asyncio.open_connection("127.0.0.1", start)
        for _ in range(100):
            if opened:
                break
            await asyncio.sleep(0.01)
        writer.close()

        assert [t.node_id for t in opened] == ["nde-rebound"]
    finally:
        await svc.stop()


def test_only_running_tasks_forward_sessions_are_restored() -> None:
    def record(task_id: str, status: str, mode: str) -> Any:
        return SimpleNamespace(
            task_id=task_id,
            workflow_id="wfl-1",
            assigned_worker="wkr-1",
            status=status,
            latest_update={"ssh": {"mode": mode, "port": 1}},
        )

    records = [
        record("tsk-forward", TaskStatus.DISPATCHED, "forward"),
        record("tsk-proxy", TaskStatus.DISPATCHED, "proxy"),
        record("tsk-done", TaskStatus.DONE, "forward"),
    ]

    sessions = forward_sessions(records)

    assert [task_id for task_id, *_ in sessions] == ["tsk-forward"]
