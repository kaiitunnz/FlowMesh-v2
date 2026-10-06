"""A target accepts a dialed peer connection before the dialer sends a session on it.

Under TLS 1.3 the dialer's handshake completes before the target has checked the
dialer's certificate or its own capacity, so only the target's answer tells the dialer
whether anything it writes will be read.
"""

import asyncio
import contextlib
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from shared.network.frame_stream import (
    AcceptFrame,
    AcceptStatus,
    FrameStreamError,
    read_relay_frame,
    read_stream_frame,
    write_accept,
    write_relay_frame,
)
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.mtls_listener import MutualTlsFrameListener
from shared.network.peer_dial import (
    PeerAcceptError,
    classify_peer_error,
    open_accepted_connection,
    open_peer_connection,
)
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.schemas.network import RouteObservationOutcome
from tests.support.certs import TestCa as _TestCa
from tests.support.certs import new_ca

_BUDGET = 5.0


def _frame() -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        correlation_id="inv-1",
        operation_id="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=b"request",
    )


class _Recording:
    """A connection handler recording the relay frames the listener hands it."""

    def __init__(self, received: list[RelayFrame]) -> None:
        self._received = received

    async def on_frame(self, frame: RelayFrame) -> None:
        self._received.append(frame)

    def close(self) -> None:
        return None


@contextlib.asynccontextmanager
async def _listener(
    material: MutualTlsMaterial | None,
    received: list[RelayFrame],
    *,
    admits: Callable[[frozenset[str]], bool] = bool,
    max_connections: int = 64,
) -> AsyncIterator[str]:
    listener = MutualTlsFrameListener(
        material=material,
        handler=lambda sink: _Recording(received),
        admits=admits,
        max_connections=max_connections,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    await listener.start_on_socket(sock)
    try:
        yield f"127.0.0.1:{listener.port}"
    finally:
        await listener.stop()


def _foreign_leaf(ca: _TestCa) -> MutualTlsMaterial:
    """The deployment's CA bundle beside a leaf another CA issued."""
    leaf = new_ca("another-ca").material("node-origin", "127.0.0.1")
    return MutualTlsMaterial(
        ca_pem=ca.material("x").ca_pem, cert_pem=leaf.cert_pem, key_pem=leaf.key_pem
    )


async def _dial(endpoint: str, origin: MutualTlsMaterial | None) -> BaseException:
    with pytest.raises((OSError, ValueError)) as raised:
        _, writer = await open_accepted_connection(
            endpoint, client_context(origin) if origin else None, _BUDGET
        )
        writer.transport.abort()
    return raised.value


@pytest.fixture(scope="module")
def ca() -> _TestCa:
    return new_ca()


def test_an_accept_step_round_trips_and_is_never_a_relay_frame() -> None:
    async def run() -> None:
        reader = asyncio.StreamReader()

        class _Writer:
            def write(self, data: bytes) -> None:
                reader.feed_data(data)

            async def drain(self) -> None:
                return None

        await write_accept(_Writer(), AcceptStatus.BUSY)
        assert await read_stream_frame(reader) == AcceptFrame(AcceptStatus.BUSY)
        await write_accept(_Writer(), AcceptStatus.REFUSED)
        with pytest.raises(FrameStreamError):
            await read_relay_frame(reader)

    asyncio.run(run())


def test_an_accepted_connection_carries_the_session(ca: _TestCa) -> None:
    received: list[RelayFrame] = []

    async def run() -> None:
        async with _listener(ca.material("node-target", "127.0.0.1"), received) as ep:
            _, writer = await open_accepted_connection(
                ep, client_context(ca.material("node-origin")), _BUDGET
            )
            await write_relay_frame(writer, _frame())
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.02)
            writer.close()

    asyncio.run(run())

    assert [f.payload for f in received] == [b"request"]


def test_a_dialer_that_sends_no_accept_request_is_served_as_before(
    ca: _TestCa,
) -> None:
    received: list[RelayFrame] = []

    async def run() -> None:
        async with _listener(ca.material("node-target", "127.0.0.1"), received) as ep:
            _, writer = await open_peer_connection(
                ep, client_context(ca.material("node-origin")), _BUDGET
            )
            await write_relay_frame(writer, _frame())
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.02)
            writer.close()

    asyncio.run(run())

    assert [f.payload for f in received] == [b"request"]


def test_a_dialer_the_target_does_not_admit_is_refused_before_it_writes(
    ca: _TestCa,
) -> None:
    received: list[RelayFrame] = []

    async def run() -> BaseException:
        async with _listener(ca.material("node-target", "127.0.0.1"), received) as ep:
            return await _dial(ep, ca.anonymous_material())

    exc = asyncio.run(run())

    assert isinstance(exc, PeerAcceptError)
    assert classify_peer_error(exc) is RouteObservationOutcome.TLS_FAILURE
    assert received == []


def test_a_leaf_another_ca_issued_fails_as_a_tls_failure(ca: _TestCa) -> None:
    received: list[RelayFrame] = []

    async def run() -> BaseException:
        async with _listener(ca.material("node-target", "127.0.0.1"), received) as ep:
            return await _dial(ep, _foreign_leaf(ca))

    exc = asyncio.run(run())

    # The target's handshake rejects the leaf and closes without an alert the dialer
    # could read, so the close itself is the TLS evidence.
    assert isinstance(exc, PeerAcceptError)
    assert classify_peer_error(exc) is RouteObservationOutcome.TLS_FAILURE
    assert received == []


def test_a_target_at_its_connection_cap_answers_busy_with_no_path_evidence(
    ca: _TestCa,
) -> None:
    received: list[RelayFrame] = []

    async def run() -> BaseException:
        async with _listener(
            ca.material("node-target", "127.0.0.1"), received, max_connections=0
        ) as ep:
            return await _dial(ep, ca.material("node-origin"))

    exc = asyncio.run(run())

    assert isinstance(exc, PeerAcceptError)
    assert exc.outcome is None
    assert received == []


async def _raw_target(serve: Callable[..., Awaitable[None]]) -> asyncio.Server:
    return await asyncio.start_server(serve, "127.0.0.1", 0)


def test_a_listener_that_reads_relay_frames_only_fails_fast_as_a_route_failure() -> (
    None
):
    async def old_listener(reader, writer) -> None:
        # A listener that predates the accept step fails to decode the request as a
        # relay frame and closes.
        with contextlib.suppress(FrameStreamError):
            await read_relay_frame(reader)
        writer.close()

    async def run() -> tuple[BaseException, float]:
        server = await _raw_target(old_listener)
        port = server.sockets[0].getsockname()[1]
        async with server:
            started = time.monotonic()
            exc = await _dial(f"127.0.0.1:{port}", None)
            return exc, time.monotonic() - started

    exc, elapsed = asyncio.run(run())

    assert isinstance(exc, PeerAcceptError)
    assert classify_peer_error(exc) is RouteObservationOutcome.ROUTE_FAILURE
    assert elapsed < _BUDGET / 5


def test_a_target_that_never_answers_times_out_within_the_connect_budget() -> None:
    async def silent(reader, writer) -> None:
        with contextlib.suppress(OSError, asyncio.IncompleteReadError):
            await reader.read()
        writer.close()

    async def run() -> BaseException:
        server = await _raw_target(silent)
        port = server.sockets[0].getsockname()[1]
        async with server:
            with pytest.raises(TimeoutError) as raised:
                await open_accepted_connection(f"127.0.0.1:{port}", None, 0.3)
            return raised.value

    exc = asyncio.run(run())

    assert classify_peer_error(exc) is RouteObservationOutcome.TIMEOUT
