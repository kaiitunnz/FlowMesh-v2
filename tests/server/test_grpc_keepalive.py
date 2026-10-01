"""The supervisor keeps pinging an idle worker stream, so a half-open one ends."""

import asyncio
import contextlib

import fakeredis
import grpc
import grpc.aio
import pytest
from google.protobuf.empty_pb2 import Empty

from server.supervisor.adapters.external import mint_external_token
from server.supervisor.services import grpc_server as grpc_server_module
from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc
from tests.server.test_external_worker_reregistration import (
    _ALIAS,
    _SECRET,
    _free_port,
    _redis,
    _Supervisor,
)


class _FreezableProxy:
    """Forwards a TCP connection until frozen, then holds both ends open silently, as
    a path that dropped without either end noticing."""

    def __init__(self, target_port: int) -> None:
        self._target_port = target_port
        self._pumps: list[asyncio.Task[None]] = []
        self._writers: list[asyncio.StreamWriter] = []
        self._server: asyncio.Server | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        return int(self._server.sockets[0].getsockname()[1])

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        up_reader, up_writer = await asyncio.open_connection(
            "127.0.0.1", self._target_port
        )
        self._writers += [writer, up_writer]
        self._pumps += [
            asyncio.ensure_future(_pump(reader, up_writer)),
            asyncio.ensure_future(_pump(up_reader, writer)),
        ]

    def freeze(self) -> None:
        for pump in self._pumps:
            pump.cancel()

    async def close(self) -> None:
        self.freeze()
        for writer in self._writers:
            writer.close()
        if self._server is not None:
            self._server.close()


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()


@pytest.mark.asyncio
async def test_a_half_open_task_stream_ends_at_the_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", _SECRET)
    monkeypatch.setattr(grpc_server_module, "_GRPC_KEEPALIVE_TIME_MS", 200)
    monkeypatch.setattr(grpc_server_module, "_GRPC_KEEPALIVE_TIMEOUT_MS", 300)
    port = _free_port()
    supervisor = _Supervisor(_redis(fakeredis.FakeServer()), port)
    await supervisor.start()
    proxy = _FreezableProxy(port)
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{await proxy.start()}")
    metadata = (("authorization", f"Bearer {mint_external_token(_SECRET, _ALIAS)}"),)
    stub = supervisor_pb2_grpc.SupervisorStub(channel)
    try:
        registered = await stub.RegisterWorker(
            supervisor_pb2.RegisterRequest(), metadata=metadata
        )
        call = stub.StreamTasks(Empty(), metadata=metadata)
        reading = asyncio.ensure_future(call.read())
        listener = supervisor.listener
        while registered.worker_id not in listener._attached:
            await asyncio.sleep(0.05)
        # Idle past the pings a stream without data would get by default.
        await asyncio.sleep(2)

        proxy.freeze()

        async def detached() -> None:
            while registered.worker_id in listener._attached:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(detached(), timeout=10)
        reading.cancel()
    finally:
        await proxy.close()
        await channel.close(grace=0)
        await supervisor.stop()
