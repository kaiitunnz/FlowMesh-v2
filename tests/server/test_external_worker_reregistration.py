"""An external worker whose supervisor restarts re-registers as a new incarnation and
gives up the dispatch it was running."""

import asyncio
import logging
import socket
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import fakeredis
import pytest

from server.clients.redis import (
    WORKERS_SET_KEY,
    SyncRedisClient,
    node_dispatch_channel,
)
from server.hooks import PrincipalContext
from server.supervisor.adapters.external import mint_external_token
from server.supervisor.manager import WorkerManager
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import GrpcServer
from server.supervisor.services.relay_service import RelayService
from server.supervisor.services.task_listener import TaskListener
from shared.schemas.command import TaskMessage
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.supervisor_client import SupervisorClient

_SECRET = "external-secret"
_ALIAS = "ext-0"
_NODE = "nod-1"
_LOGGER = logging.getLogger("test.external_reregistration")


class _RecordingRelay(RelayService):
    """Keeps what the supervisor relays to the root, in order."""

    def __init__(self) -> None:
        super().__init__(SyncRedisClient.__new__(SyncRedisClient), _LOGGER)
        self.events: list[dict[str, Any]] = []

    def add_event(self, event_data: Any) -> None:
        self.events.append(dict(event_data))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _redis(server: fakeredis.FakeServer) -> SyncRedisClient:
    client = SyncRedisClient.__new__(SyncRedisClient)
    client._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    client._telemetry = client._control
    client.logger = _LOGGER
    return client


class _Supervisor:
    """One supervisor process's gRPC side, as a restart would build it afresh."""

    def __init__(self, redis: SyncRedisClient, port: int) -> None:
        self.listener = TaskListener(redis, _NODE, _LOGGER)
        registry = WorkerRegistry(on_worker_id_released=self.listener.remove_worker)
        manager = WorkerManager(
            system_principal=MagicMock(spec=PrincipalContext),
            config_path="unused",
            registry=registry,
            logger=_LOGGER,
        )
        manager._is_started = True
        manager._default_worker_config = {}
        self.relay = _RecordingRelay()
        self.server = GrpcServer(
            "127.0.0.1",
            port,
            registry,
            redis,
            _NODE,
            "box",
            self.listener,
            self.relay,
            manager,
            _LOGGER,
        )

    async def start(self) -> None:
        self.listener.start()
        await self.server.start()

    async def stop(self) -> None:
        # A crash: the worker's streams are cut, not drained.
        await self.server.stop(grace=0)
        self.listener.stop()


def _client(port: int) -> SupervisorClient:
    return SupervisorClient(
        worker_token=mint_external_token(_SECRET, _ALIAS),
        owner_principal=None,
        grpc_target=f"127.0.0.1:{port}",
        worker_namespace="ns",
        worker_cluster="cl",
        worker_alias=_ALIAS,
        logger=_LOGGER,
    )


def _register(client: SupervisorClient) -> None:
    client.register(
        WorkerStatus.IDLE,
        "2026-10-02T00:00:00Z",
        1,
        {},
        make_worker_hardware(),
        WorkerCapabilities(supported_task_types=frozenset({TaskType.ECHO})),
        None,
        [],
        1.0,
    )


async def _until(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_supervisor_restart_re_admits_the_worker_and_abandons_its_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    redis = _redis(fakeredis.FakeServer())
    port = _free_port()
    first = _Supervisor(redis, port)
    await first.start()

    client = _client(port)
    abandoned: list[str | None] = []
    client.on_reregistered(abandoned.append)
    await asyncio.to_thread(_register, client)
    old_id, old_incarnation = client.worker_id, client.incarnation
    await asyncio.to_thread(client.start)
    try:
        tasks = iter(client.iter_tasks())
        message = make_worker_task_message(
            {"taskType": "echo"},
            task_type=TaskType.ECHO,
            task_id="tsk-1",
            assigned_worker=old_id,
            dispatch_id="dsp-1",
        )
        redis.publish_control(
            node_dispatch_channel(_NODE),
            TaskMessage(
                worker_id=old_id,
                payload=message.model_dump(mode="json", exclude_none=True),
            ).model_dump_json(),
        )
        received = await asyncio.to_thread(next, tasks)
        assert received.dispatch_id == "dsp-1"

        await first.stop()
        second = _Supervisor(redis, port)
        await second.start()
        try:
            await _until(lambda: bool(abandoned))
            assert abandoned == ["dsp-1"]
            new_id = client.worker_id
            assert new_id != old_id
            assert client.incarnation > old_incarnation

            # The abandoned run's reports, then a heartbeat that must follow them.
            await asyncio.to_thread(client.task_failed, "tsk-1", "boom")
            await asyncio.to_thread(client.set_status, WorkerStatus.BUSY, None, "dsp-1")
            await asyncio.to_thread(client.heartbeat)
            relayed = second.relay.events
            await _until(lambda: any(e.get("type") == "HEARTBEAT" for e in relayed))
        finally:
            await second.stop()
    finally:
        await asyncio.to_thread(client.shutdown)

    assert not [e for e in relayed if str(e.get("type", "")).startswith("TASK_")]
    assert {e["worker_id"] for e in relayed} == {new_id}
    status = next(e for e in relayed if e.get("type") == "STATUS")
    assert status.get("dispatch_id") is None


@pytest.mark.asyncio
async def test_a_worker_the_root_reaped_registers_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    redis = _redis(fakeredis.FakeServer())
    port = _free_port()
    supervisor = _Supervisor(redis, port)
    await supervisor.start()
    client = _client(port)
    await asyncio.to_thread(_register, client)
    old_id = client.worker_id
    await asyncio.to_thread(client.start)
    try:
        redis.srem(WORKERS_SET_KEY, old_id)

        await asyncio.to_thread(supervisor.server.reconcile_workers)

        await _until(lambda: client.worker_id != old_id)
        assert redis.set_members(WORKERS_SET_KEY) == {client.worker_id}
    finally:
        await asyncio.to_thread(client.shutdown)
        await supervisor.stop()
