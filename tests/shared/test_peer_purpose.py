"""A peer connection's purpose is fixed once, by its accept exchange.

A listener hands a connection only to the handler of a purpose it serves, with what it
verified about the dialer; a dialer uses a non-resident connection only once the target
has echoed its purpose.
"""

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator

import pytest

from shared.network.frame_stream import (
    AcceptFrame,
    AcceptStatus,
    PeerPurpose,
    read_stream_frame,
    write_accept,
    write_relay_frame,
)
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.mtls_listener import MutualTlsFrameListener, PeerContext
from shared.network.peer_dial import PeerAcceptError, open_accepted_connection
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from tests.support.certs import TestCa as _TestCa
from tests.support.certs import new_ca

_BUDGET = 5.0


def _frame() -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        correlation_id="chg-1",
        operation_id="",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=b"fetch",
    )


class _Recording:
    def __init__(self, seen: list[tuple[PeerContext, RelayFrame]], ctx: PeerContext):
        self._seen = seen
        self._ctx = ctx

    async def on_frame(self, frame: RelayFrame) -> None:
        self._seen.append((self._ctx, frame))

    def close(self) -> None:
        return None


@contextlib.asynccontextmanager
async def _listener(
    material: MutualTlsMaterial | None,
    purposes: frozenset[PeerPurpose],
    seen: list[tuple[PeerContext, RelayFrame]],
) -> AsyncIterator[str]:
    listener = MutualTlsFrameListener(
        material=material,
        handler=lambda sink, ctx: _Recording(seen, ctx),
        admits=bool,
        purposes=purposes,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    await listener.start_on_socket(sock)
    try:
        yield f"127.0.0.1:{listener.port}"
    finally:
        await listener.stop()


async def _received(seen: list, count: int = 1) -> None:
    for _ in range(200):
        if len(seen) >= count:
            return
        await asyncio.sleep(0.01)


@pytest.fixture(scope="module")
def ca() -> _TestCa:
    return new_ca()


def test_a_content_connection_reaches_its_handler_with_the_dialers_names(
    ca: _TestCa,
) -> None:
    seen: list[tuple[PeerContext, RelayFrame]] = []

    async def run() -> None:
        target = ca.material("node-b", "127.0.0.1")
        async with _listener(target, frozenset({PeerPurpose.CONTENT}), seen) as ep:
            _, writer = await open_accepted_connection(
                ep,
                client_context(ca.material("node-a", "10.0.0.7")),
                _BUDGET,
                PeerPurpose.CONTENT,
            )
            await write_relay_frame(writer, _frame())
            await _received(seen)
            writer.close()

    asyncio.run(run())
    ((ctx, frame),) = seen
    assert ctx.purpose is PeerPurpose.CONTENT
    assert ctx.names.covers("10.0.0.7") and ctx.names.covers("node-a")
    assert frame.payload == b"fetch"


def test_a_listener_refuses_a_purpose_it_does_not_serve_before_any_frame(
    ca: _TestCa,
) -> None:
    seen: list[tuple[PeerContext, RelayFrame]] = []

    async def run() -> None:
        target = ca.material("node-b", "127.0.0.1")
        origin = client_context(ca.material("node-a"))
        async with _listener(target, frozenset({PeerPurpose.RESIDENT}), seen) as ep:
            with pytest.raises(PeerAcceptError) as refused:
                await open_accepted_connection(ep, origin, _BUDGET, PeerPurpose.CONTENT)
            assert refused.value.outcome is None
        async with _listener(target, frozenset({PeerPurpose.CONTENT}), seen) as ep:
            with pytest.raises(PeerAcceptError) as refused:
                await open_accepted_connection(ep, origin, _BUDGET)
            assert refused.value.outcome is None

    asyncio.run(run())
    assert seen == []


def test_a_request_naming_no_purpose_is_a_resident_one(ca: _TestCa) -> None:
    seen: list[tuple[PeerContext, RelayFrame]] = []

    async def run() -> None:
        target = ca.material("node-b", "127.0.0.1")
        async with _listener(target, frozenset({PeerPurpose.RESIDENT}), seen) as ep:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1",
                int(ep.rpartition(":")[2]),
                ssl=client_context(ca.material("node-a")),
                server_hostname="127.0.0.1",
            )
            await write_accept(writer, AcceptStatus.REQUEST)
            answer = await read_stream_frame(reader)
            assert answer == AcceptFrame(AcceptStatus.ACCEPTED, PeerPurpose.RESIDENT)
            await write_relay_frame(writer, _frame())
            await _received(seen)
            writer.close()

    asyncio.run(run())
    assert [ctx.purpose for ctx, _ in seen] == [PeerPurpose.RESIDENT]


def test_an_acceptance_without_the_purpose_echo_is_never_used() -> None:
    """A listener that predates purposes would carry the session as a resident one."""

    async def run() -> None:
        async def old_listener(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await read_stream_frame(reader)
            await write_accept(writer, AcceptStatus.ACCEPTED)
            await reader.read()

        server = await asyncio.start_server(old_listener, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            with pytest.raises(PeerAcceptError) as unserved:
                await open_accepted_connection(
                    f"127.0.0.1:{port}", None, _BUDGET, PeerPurpose.CONTENT
                )
            assert unserved.value.outcome is None
            _, writer = await open_accepted_connection(
                f"127.0.0.1:{port}", None, _BUDGET
            )
            writer.close()
        finally:
            server.close()

    asyncio.run(run())


def test_a_resident_request_carries_no_purpose_on_the_wire() -> None:
    async def run() -> AcceptFrame:
        requests: list[AcceptFrame] = []

        async def listener(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            request = await read_stream_frame(reader)
            assert isinstance(request, AcceptFrame)
            requests.append(request)
            await write_accept(writer, AcceptStatus.ACCEPTED)
            await reader.read()

        server = await asyncio.start_server(listener, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            _, writer = await open_accepted_connection(
                f"127.0.0.1:{port}", None, _BUDGET
            )
            writer.close()
        finally:
            server.close()
        return requests[0]

    assert asyncio.run(run()) == AcceptFrame(AcceptStatus.REQUEST)


def test_a_dialers_common_name_is_not_among_the_names_it_covers(ca: _TestCa) -> None:
    seen: list[tuple[PeerContext, RelayFrame]] = []

    async def run() -> None:
        target = ca.material("node-b", "127.0.0.1")
        async with _listener(target, frozenset({PeerPurpose.CONTENT}), seen) as ep:
            _, writer = await open_accepted_connection(
                ep,
                client_context(ca.named_material("10.0.0.7", "node-a.example")),
                _BUDGET,
                PeerPurpose.CONTENT,
            )
            await write_relay_frame(writer, _frame())
            await _received(seen)
            writer.close()

    asyncio.run(run())
    ((ctx, _),) = seen
    assert "10.0.0.7" in ctx.identities
    assert not ctx.names.covers("10.0.0.7")
    assert ctx.names.covers("node-a.example")


def test_an_unannounced_session_never_reaches_a_content_listener(ca: _TestCa) -> None:
    seen: list[tuple[PeerContext, RelayFrame]] = []

    async def run() -> None:
        target = ca.material("node-b", "127.0.0.1")
        async with _listener(target, frozenset({PeerPurpose.CONTENT}), seen) as ep:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1",
                int(ep.rpartition(":")[2]),
                ssl=client_context(ca.material("node-a")),
                server_hostname="127.0.0.1",
            )
            await write_relay_frame(writer, _frame())
            assert await reader.read() == b""
            writer.close()

    asyncio.run(run())
    assert seen == []
