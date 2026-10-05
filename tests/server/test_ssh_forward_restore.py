"""A forward SSH session is served again on its published port after a root restart,
and each of its connections is relayed to its worker's current node."""

import asyncio
import socket
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from server.services.port_forward import PortForwardService
from server.ssh import SshRelayTarget
from server.task.models import TaskRecord, TaskStatus
from shared.tasks import TaskEnvelopeTemplate
from tests.server.test_ssh_forward_ports import _free_port_range, _make_service


def _session(
    task_id: str,
    port: int,
    status: str = TaskStatus.DISPATCHED,
    mode: str = "forward",
) -> TaskRecord:
    endpoint = {
        "session_id": f"ssn-{task_id}",
        "username": "flowmesh",
        "mode": mode,
        "host": "lum.id",
        "port": port,
    }
    return TaskRecord(
        task_id=task_id,
        workflow_id="wfl-1",
        owner_id="admin",
        raw_yaml="",
        task=TaskEnvelopeTemplate.model_validate(
            {"apiVersion": "flowmesh/v1", "kind": "Task", "spec": {"taskType": "ssh"}}
        ),
        status=status,
        assigned_worker="wkr-1",
        latest_update={"ssh": endpoint},
    )


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


@pytest.mark.anyio
async def test_only_a_running_task_s_forward_session_is_restored() -> None:
    start, end = _free_port_range(3)
    svc = _make_service(start, end, persistent_listeners=False)
    await svc.start()
    try:
        await svc.restore_sessions(
            [
                _session("tsk-forward", start),
                _session("tsk-proxy", start + 1, mode="proxy"),
                _session("tsk-done", end, status=TaskStatus.DONE),
            ]
        )

        assert set(svc._port_to_task.values()) == {"tsk-forward"}
    finally:
        await svc.stop()
