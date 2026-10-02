"""An external worker whose supervisor restarts re-registers as a new incarnation and
gives up the dispatch it was running."""

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import fakeredis
import pytest

from server.clients.redis import (
    WORKERS_SET_KEY,
    RedisClient,
    SyncRedisClient,
    node_dispatch_channel,
)
from server.hooks import PrincipalContext
from server.registries.worker import WorkerRegistry as WorkerRecords
from server.supervisor.adapters.external import mint_external_token
from server.supervisor.manager import WorkerManager
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import GrpcServer
from server.supervisor.services.relay_service import RelayService
from server.supervisor.services.task_listener import TaskListener
from shared.content import BACKEND_FILESYSTEM, ObjectStoreConfig
from shared.schemas.command import TaskMessage
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks import TaskType
from tests.server.redis_helpers import fake_redis_client
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.content import (
    ContentAccessRegistry,
    ContentLaneHost,
    WorkerContentCache,
    WorkerContentPlane,
)
from worker.executors.base_executor import EchoExecutor
from worker.lifecycle import Lifecycle
from worker.resident.lane_host import ResidentLaneHost
from worker.runner import Runner
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


class _Supervisor:
    """One supervisor process's gRPC side, as a restart would build it afresh."""

    def __init__(self, redis_client: RedisClient, port: int) -> None:
        redis = redis_client.sync
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
            WorkerRecords(redis_client),
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


# Well under a lane's 30 s rebind bound, so a re-registration whose event thread waits
# on a rebind fails the test.
_REREGISTERED_WITHIN_SEC = 10.0


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
    redis_client = fake_redis_client(fakeredis.FakeServer())
    redis = redis_client.sync
    port = _free_port()
    first = _Supervisor(redis_client, port)
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
        second = _Supervisor(redis_client, port)
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
    redis_client = fake_redis_client(fakeredis.FakeServer())
    redis = redis_client.sync
    port = _free_port()
    supervisor = _Supervisor(redis_client, port)
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


def _event_stream_registers_first(
    client: SupervisorClient,
    monkeypatch: pytest.MonkeyPatch,
    while_registering: Callable[[], Any] | None = None,
) -> list[str]:
    """Make the event-stream thread win the re-registration, running
    ``while_registering`` as it registers; return who registered."""
    reregister = client._reregister
    retry_register = client._retry_register_grpc
    registered_by: list[str] = []

    def task_thread_waits(seen_gen: int) -> int:
        if threading.current_thread().name == "SupervisorTaskStream":
            deadline = time.monotonic() + 30
            while client._register_generation == seen_gen:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
        return reregister(seen_gen)

    def recording_retry() -> tuple[str, int] | None:
        registered_by.append(threading.current_thread().name)
        if while_registering is not None:
            while_registering()
        return retry_register()

    monkeypatch.setattr(client, "_reregister", task_thread_waits)
    monkeypatch.setattr(client, "_retry_register_grpc", recording_retry)
    return registered_by


def _lifecycle(client: SupervisorClient, tmp_path: Path) -> Lifecycle:
    lifecycle = Lifecycle(client, 30, 120, tmp_path / "worker.hb", 1.0)
    client.on_reregistered(lifecycle._on_reregistered)
    return lifecycle


def _runner(lifecycle: Lifecycle, tmp_path: Path) -> Runner:
    executor = EchoExecutor(make_worker_config())
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=_LOGGER,
    )
    lifecycle.set_abandon_handler(runner.abandon_running)
    return runner


def _relayed(supervisor: _Supervisor, kind: str, worker_id: str) -> bool:
    return any(
        e.get("type") == kind and e.get("worker_id") == worker_id
        for e in supervisor.relay.events
    )


@pytest.mark.asyncio
async def test_a_worker_holding_content_moves_to_its_new_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    redis_client = fake_redis_client(fakeredis.FakeServer())
    port = _free_port()
    first = _Supervisor(redis_client, port)
    await first.start()

    client = _client(port)
    registered_by = _event_stream_registers_first(client, monkeypatch)
    await asyncio.to_thread(_register, client)
    old_id = client.worker_id
    cache = WorkerContentCache(tmp_path / "content", retain_sec=900.0)
    cache.write("local", b"held across the restart")
    lane = ContentLaneHost(
        store=cache,
        push_frame=client.push_content_frame,
        request_grant=lambda reference, task_id: None,
        worker_id=old_id,
        generation=client.incarnation,
        announce=client.push_content_holding,
    )
    lifecycle = _lifecycle(client, tmp_path)
    lifecycle.content_plane = WorkerContentPlane(
        lane,
        ContentAccessRegistry(
            ObjectStoreConfig(
                backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
            )
        ),
    )
    await asyncio.to_thread(client.start)
    lane.start()
    try:
        await first.stop()
        second = _Supervisor(redis_client, port)
        await second.start()
        try:
            await _until(lambda: client.worker_id != old_id)
            new_id = client.worker_id
            await _until(
                lambda: _relayed(second, "REGISTER", new_id), _REREGISTERED_WITHIN_SEC
            )
            await _until(
                lambda: _relayed(second, "CONTENT_HOLDING", new_id),
                _REREGISTERED_WITHIN_SEC,
            )
            await asyncio.to_thread(client.heartbeat)
            await _until(
                lambda: _relayed(second, "HEARTBEAT", new_id), _REREGISTERED_WITHIN_SEC
            )
        finally:
            await second.stop()
    finally:
        lane.stop()
        await asyncio.to_thread(client.shutdown)
    assert registered_by == ["SupervisorEventStream"]


@pytest.mark.asyncio
async def test_a_worker_streaming_a_resident_session_moves_to_its_new_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    redis_client = fake_redis_client(fakeredis.FakeServer())
    port = _free_port()
    first = _Supervisor(redis_client, port)
    await first.start()

    client = _client(port)
    host = ResidentLaneHost(
        push_frame=client.push_resident_frame,
        report_ack=lambda ack: None,
        report_outcome=lambda outcome: None,
        content_store_for=lambda task_id: None,
        peek_request=lambda task_id, call: None,
        delete_request=lambda task_id, call: None,
    )

    # A session streaming as the worker re-registers: the lane loop blocks on its send,
    # as a frame sink does, until the event stream is back.
    async def stream_one_frame() -> None:
        client.push_resident_frame({"kind": "resident_data"})

    registered_by = _event_stream_registers_first(
        client, monkeypatch, lambda: host._call(stream_one_frame)
    )
    await asyncio.to_thread(_register, client)
    old_id = client.worker_id
    lifecycle = _lifecycle(client, tmp_path)
    runner = _runner(lifecycle, tmp_path)
    host.start()
    runner._resident_host = host
    await asyncio.to_thread(client.start)
    try:
        await first.stop()
        second = _Supervisor(redis_client, port)
        await second.start()
        try:
            await _until(lambda: client.worker_id != old_id)
            new_id = client.worker_id
            await _until(
                lambda: _relayed(second, "REGISTER", new_id), _REREGISTERED_WITHIN_SEC
            )
            await asyncio.to_thread(client.heartbeat)
            await _until(
                lambda: _relayed(second, "HEARTBEAT", new_id), _REREGISTERED_WITHIN_SEC
            )
        finally:
            await second.stop()
    finally:
        host.stop()
        await asyncio.to_thread(client.shutdown)
    assert registered_by == ["SupervisorEventStream"]


@pytest.mark.asyncio
async def test_a_worker_the_root_reaped_registers_its_new_id_with_the_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    redis_client = fake_redis_client(fakeredis.FakeServer())
    redis = redis_client.sync
    port = _free_port()
    supervisor = _Supervisor(redis_client, port)
    await supervisor.start()
    client = _client(port)
    await asyncio.to_thread(_register, client)
    old_id = client.worker_id
    await asyncio.to_thread(client.start)
    try:
        await _until(lambda: _relayed(supervisor, "REGISTER", old_id))
        redis.srem(WORKERS_SET_KEY, old_id)
        await asyncio.to_thread(supervisor.server.reconcile_workers)
        await _until(lambda: client.worker_id != old_id)
        new_id = client.worker_id

        await _until(lambda: _relayed(supervisor, "REGISTER", new_id))
    finally:
        await asyncio.to_thread(client.shutdown)
        await supervisor.stop()
