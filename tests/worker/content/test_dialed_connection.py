"""One dialed content connection carries one transfer, and how it ends is classified.

The holder closes the connection after the transfer it carried, and the requester
reports what the path did: a connection that ends before its transfer does is a network
failure, classified by how it ended.
"""

import asyncio
import ssl
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentPeerRoute
from shared.network.mtls_listener import ConnectionFrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.network.session import FramedRelaySession, RelaySessionRole
from shared.schemas.network import RouteObservationOutcome, Transport
from worker.content import ContentHolder, WorkerContentCache
from worker.content import holder as holder_module
from worker.content.peer import PeerConnection


class _Writer:
    def __init__(self) -> None:
        self.closed = False

    def write(self, data: bytes) -> None:
        return None

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Dialed(ConnectionFrameSink):
    """A dialed connection the holder answers over, recording its close."""

    def __init__(self) -> None:
        self.writer = _Writer()
        super().__init__(self.writer)  # type: ignore[arg-type]


def _fetch(session_id: str) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id=session_id,
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=b"not a fetch",
    )


@pytest.mark.asyncio
async def test_the_holder_closes_a_dialed_connection_after_its_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holder_module, "_DIALED_CLOSE_GRACE_SEC", 0.05)
    holder = ContentHolder(
        store=WorkerContentCache(tmp_path),
        sink=_Dialed(),
        holder_id="wkr-1",
        generation=1,
    )
    dialed = _Dialed()
    await holder.on_frame(_fetch("rly-1"), dialed)
    async with asyncio.timeout(5.0):
        while not dialed.writer.closed:
            await asyncio.sleep(0.01)
    await holder.aclose()


async def _ended_with(
    feed: Any,
) -> list[RouteObservationOutcome]:
    reader = asyncio.StreamReader()
    observed: list[RouteObservationOutcome] = []
    connection = PeerConnection(
        reader,
        _Writer(),  # type: ignore[arg-type]
        ContentPeerRoute(transport=Transport.WORKER_DIRECT, endpoint="127.0.0.1:1"),
        on_loss=observed.append,
        on_frame=lambda: None,
    )
    session = FramedRelaySession(
        session_id="rly-1", role=RelaySessionRole.ORIGIN, sink=connection
    )
    connection.attach(session)
    feed(reader)
    async with asyncio.timeout(5.0):
        await session.wait_cancelled()
    return observed


@pytest.mark.asyncio
async def test_a_connection_that_ends_before_its_transfer_is_a_route_failure() -> None:
    assert await _ended_with(lambda reader: reader.feed_eof()) == [
        RouteObservationOutcome.ROUTE_FAILURE
    ]


@pytest.mark.asyncio
async def test_a_connection_that_breaks_mid_frame_is_a_route_failure() -> None:
    def broken(reader: asyncio.StreamReader) -> None:
        reader.feed_data(b"\x00\x00")
        reader.feed_eof()

    assert await _ended_with(broken) == [RouteObservationOutcome.ROUTE_FAILURE]


@pytest.mark.asyncio
async def test_a_connection_lost_to_tls_is_a_tls_failure() -> None:
    def tls(reader: asyncio.StreamReader) -> None:
        reader.set_exception(ssl.SSLError("bad record mac"))

    assert await _ended_with(tls) == [RouteObservationOutcome.TLS_FAILURE]
