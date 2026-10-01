"""A relayed SSH connection end to end: the root origin, the root bridge, a node's
attachment and bridge, and the worker's lane, carrying bytes to a loopback listener
the worker published."""

import asyncio
import contextlib
import hashlib
import logging
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import (
    SSH_RELAY_KEYSPACE,
    RelaySessionStore,
    RelayStreamStore,
)
from server.network.worker_bridge import RelayWorkerBridge
from server.services.port_forward import PortForwardService
from server.ssh import SSH_EDGE_STREAM_ID, SshRelayOrigin, SshRelayTarget
from server.supervisor.services.reverse_relay_attachment import ReverseRelayAttachment
from shared.network.byte_stream import StreamClosed
from shared.network.relay_frame import RelayFrame
from worker.executors.ssh_session.base import (
    count_established_connections,
    read_local_proc_net_tcp,
)
from worker.ssh_relay import SshEndpointRegistry, SshRelayLane

from ._relay_fakes import FakeBinaryRedis

NODE = "nde-1"
WORKER = "wkr-1"
ENDPOINT = "ssn-1"
TARGET = SshRelayTarget(
    task_id="tsk-1", worker_id=WORKER, node_id=NODE, endpoint_id=ENDPOINT
)

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


@dataclass
class _Listener:
    """A loopback server standing in for a session's sshd."""

    port: int
    accepted: int = 0
    closed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Fabric:
    redis: FakeBinaryRedis
    origin: SshRelayOrigin
    registry: SshEndpointRegistry
    lane: SshRelayLane

    def new_origin(self) -> SshRelayOrigin:
        return SshRelayOrigin(self.redis, refresh_interval_sec=3600)


@contextlib.asynccontextmanager
async def _fabric() -> AsyncIterator[_Fabric]:
    loop = asyncio.get_running_loop()
    redis = FakeBinaryRedis()
    registry = SshEndpointRegistry()
    node_bridge: RelayWorkerBridge | None = None

    def push(wire: dict[str, Any]) -> None:
        assert node_bridge is not None
        asyncio.run_coroutine_threadsafe(
            node_bridge.publish_up(RelayFrame.from_wire(wire)), loop
        )

    lane = SshRelayLane(registry=registry, push_frame=push)

    async def enqueue_local(worker_id: str, payload: dict[str, Any]) -> bool:
        if worker_id != WORKER:
            return False
        lane.route(payload["frame_kind"], payload["payload"])
        return True

    node_bridge = RelayWorkerBridge(
        redis,
        NODE,
        enqueue_local,
        keyspace=SSH_RELAY_KEYSPACE,
        frame_kind="ssh_frame",
    )
    attachment = ReverseRelayAttachment(
        redis, NODE, node_bridge, owner="test", keyspace=SSH_RELAY_KEYSPACE, poll_ms=5
    )
    root_bridge = RootRendezvousBridge(
        RelayStreamStore(redis, SSH_RELAY_KEYSPACE),
        RelaySessionStore(redis, SSH_RELAY_KEYSPACE),
        RootCursorStore(redis, SSH_RELAY_KEYSPACE),
    )

    async def pump() -> None:
        while True:
            await root_bridge.pump_ready([NODE, SSH_EDGE_STREAM_ID], 1000)

    origin = SshRelayOrigin(redis, refresh_interval_sec=3600)
    lane.start()
    attachment.start(loop)
    pump_task = asyncio.create_task(pump())
    await origin.start()
    try:
        yield _Fabric(redis, origin, registry, lane)
    finally:
        await origin.stop()
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task
        await attachment.stop()
        lane.stop()


@contextlib.asynccontextmanager
async def _listener(handler: Handler) -> AsyncIterator[_Listener]:
    state = _Listener(port=0)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        state.accepted += 1
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            state.closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    state.port = server.sockets[0].getsockname()[1]
    try:
        yield state
    finally:
        server.close()
        await server.wait_closed()


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()


async def _digest_after_eof(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Read to EOF, then answer with the digest of everything read."""
    digest = hashlib.sha256()
    while data := await reader.read(65536):
        digest.update(data)
    writer.write(digest.hexdigest().encode())
    await writer.drain()


async def _hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.read()


def _run(coro: Awaitable[None]) -> None:
    asyncio.run(asyncio.wait_for(coro, 20))  # type: ignore[arg-type]


def test_bytes_round_trip_and_a_clean_close_needs_no_cancel() -> None:
    async def run() -> None:
        async with _fabric() as fabric, _listener(_echo) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            channel = await fabric.origin.open(TARGET)
            await channel.send(b"keystroke")
            assert await channel.recv() == b"keystroke"
            await channel.send_eof()
            assert await channel.recv() is None
            assert not channel.closed
            await fabric.origin.release(channel, abort=False)
            await asyncio.wait_for(sshd.closed.wait(), 5)

    _run(run())


def test_a_half_close_keeps_the_other_direction_and_orders_behind_the_data() -> None:
    """The client's EOF arrives after every byte it sent, and the reply still flows."""

    async def run() -> None:
        payload = bytes(range(256)) * 8192  # 2 MiB, several windows
        async with _fabric() as fabric, _listener(_digest_after_eof) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            channel = await fabric.origin.open(TARGET)
            await channel.send(payload)
            await channel.send_eof()
            reply = b""
            while (data := await channel.recv()) is not None:
                reply += data
            assert reply == hashlib.sha256(payload).hexdigest().encode()

    _run(run())


def test_an_unpublished_endpoint_is_refused_without_a_connection() -> None:
    async def run() -> None:
        async with _fabric() as fabric, _listener(_echo) as sshd:
            fabric.registry.publish("ssn-other", sshd.port)
            channel = await fabric.origin.open(TARGET)
            with pytest.raises(StreamClosed, match="refused"):
                await channel.recv()
            assert sshd.accepted == 0

    _run(run())


def test_withdrawing_the_endpoint_ends_its_connections_at_both_ends() -> None:
    async def run() -> None:
        async with _fabric() as fabric, _listener(_echo) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            channel = await fabric.origin.open(TARGET)
            await channel.send(b"x")
            assert await channel.recv() == b"x"
            await asyncio.to_thread(fabric.registry.withdraw, ENDPOINT)
            with pytest.raises(StreamClosed):
                await channel.recv()
            await asyncio.wait_for(sshd.closed.wait(), 5)
            assert fabric.registry.resolve(ENDPOINT) is None

    _run(run())


def test_closing_a_task_ends_its_connections_at_both_ends() -> None:
    async def run() -> None:
        async with _fabric() as fabric, _listener(_hold) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            channel = await fabric.origin.open(TARGET)
            await channel.send(b"x")
            while sshd.accepted == 0:
                await asyncio.sleep(0.005)
            fabric.origin.close_task(TARGET.task_id)
            with pytest.raises(StreamClosed):
                await channel.recv()
            await asyncio.wait_for(sshd.closed.wait(), 5)

    _run(run())


def test_a_restarted_root_reaps_the_connections_its_predecessor_left_open() -> None:
    """A lane connection outliving its root keeps the session's sshd busy, so the
    session's idle timer could never fire; the reap lets it."""

    async def run() -> None:
        async with _fabric() as fabric, _listener(_hold) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            orphan = fabric.new_origin()
            orphan._loop = asyncio.get_running_loop()
            channel = await orphan.open(TARGET)
            # The orphan's own attachment never runs: its process is gone.
            await channel.send(b"x")

            def established() -> int:
                text = read_local_proc_net_tcp()
                assert text is not None
                return count_established_connections(text, sshd.port)

            while established() == 0:
                await asyncio.sleep(0.005)
            assert await fabric.new_origin().reap_orphans() >= 1
            await asyncio.wait_for(sshd.closed.wait(), 5)
            while established() > 0:
                await asyncio.sleep(0.005)

    _run(run())


@contextlib.asynccontextmanager
async def _forward(fabric: _Fabric) -> AsyncIterator[tuple[PortForwardService, int]]:
    worker = MagicMock(node_id=NODE)
    workers = MagicMock()
    workers.get_worker_async = AsyncMock(return_value=worker)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    service = PortForwardService(
        relay=fabric.origin,
        worker_registry=workers,
        ssh_connections=None,
        bind_host="127.0.0.1",
        public_host="127.0.0.1",
        port_start=port,
        port_end=port,
        persistent_listeners=False,
        logger=logging.getLogger("test.ssh_forward_relay"),
    )
    await service.start()
    try:
        endpoint = await service._register_task_async(
            TARGET.task_id, "wfl-1", WORKER, {"session_id": ENDPOINT, "mode": "forward"}
        )
        yield service, int(endpoint["port"])
    finally:
        await service.stop()


def test_a_forward_client_half_closes_and_still_reads_the_reply() -> None:
    async def run() -> None:
        payload = b"scp-bytes" * 100_000
        async with _fabric() as fabric, _listener(_digest_after_eof) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            async with _forward(fabric) as (_, port):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(payload)
                await writer.drain()
                writer.write_eof()
                reply = await asyncio.wait_for(reader.read(), 10)
                assert reply == hashlib.sha256(payload).hexdigest().encode()
                writer.close()

    _run(run())


def test_unregistering_a_forward_task_ends_its_live_connections() -> None:
    async def run() -> None:
        async with _fabric() as fabric, _listener(_echo) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            async with _forward(fabric) as (service, port):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"x")
                await writer.drain()
                assert await asyncio.wait_for(reader.read(1), 5) == b"x"
                await service._unregister_task_async(TARGET.task_id)
                assert await asyncio.wait_for(reader.read(), 5) == b""
                await asyncio.wait_for(sshd.closed.wait(), 5)
                writer.close()

    _run(run())


def test_a_reforwarded_opening_frame_of_an_ended_session_opens_nothing() -> None:
    """A bridge restart can re-forward a session's first frame after it ended; a
    connection it opened would keep the session from ever idling out."""

    async def run() -> None:
        async with _fabric() as fabric, _listener(_echo) as sshd:
            fabric.registry.publish(ENDPOINT, sshd.port)
            sent: list[RelayFrame] = []
            sink = fabric.origin._sink
            original = sink.send

            async def capture(frame: RelayFrame) -> None:
                sent.append(frame)
                await original(frame)

            sink.send = capture  # type: ignore[method-assign]
            channel = await fabric.origin.open(TARGET)
            await channel.send_eof()
            assert await channel.recv() is None
            await fabric.origin.release(channel, abort=False)
            await asyncio.wait_for(sshd.closed.wait(), 5)

            await sink.send(sent[0])
            await asyncio.sleep(0.3)
            assert sshd.accepted == 1

    _run(run())
