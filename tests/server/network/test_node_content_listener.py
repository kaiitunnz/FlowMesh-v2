"""A node's peer listener hands each dialed session to the uplink of its purpose.

A content session reaches the content uplink only when control routed it to this node,
for a dialer whose certificate covers the host control named for its requester.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from server.supervisor.services.peer_listener import NodePeerListener
from shared.network.frame_stream import PeerPurpose, write_relay_frame
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.peer_dial import PeerAcceptError, open_accepted_connection
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from tests.support.certs import TestCa as _TestCa
from tests.support.certs import new_ca

_REQUESTER_HOST = "10.0.0.7"


class _Uplink:
    """One namespace's bridge: control's routing records, and what reached it."""

    def __init__(self, node_id: str, records: dict[str, dict[str, str]]) -> None:
        self.node_id = node_id
        self._records = records
        self.frames: list[RelayFrame] = []
        self.bound: list[str] = []
        self.sinks: dict[str, Any] = {}

    async def session(self, session_id: str) -> dict[str, str]:
        return self._records.get(session_id, {})

    def bind_peer(self, session_id: str, sink: Any) -> None:
        self.bound.append(session_id)
        self.sinks[session_id] = sink

    def release_peer(self, session_id: str, sink: Any) -> None:
        return None

    async def on_frame(self, frame: RelayFrame) -> None:
        self.frames.append(frame)


def _frame(session_id: str) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id=session_id,
        correlation_id="chg-1",
        operation_id="",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=b"fetch",
    )


@contextlib.asynccontextmanager
async def _listener(
    ca: _TestCa, bridges: dict[PeerPurpose, _Uplink]
) -> AsyncIterator[str]:
    listener = NodePeerListener(
        endpoint="127.0.0.1:0",
        material=ca.material("node-b", "127.0.0.1"),
        bridges=cast(Any, bridges),
    )
    await listener.start()
    try:
        yield f"127.0.0.1:{listener.port}"
    finally:
        await listener.stop()


async def _send(
    endpoint: str,
    dialer: MutualTlsMaterial,
    purpose: PeerPurpose,
    session_id: str,
) -> None:
    reader, writer = await open_accepted_connection(
        endpoint, client_context(dialer), 5.0, purpose
    )
    await write_relay_frame(writer, _frame(session_id))
    with contextlib.suppress(asyncio.TimeoutError, ConnectionError):
        await asyncio.wait_for(reader.read(), 0.5)
    writer.close()


@pytest.fixture(scope="module")
def ca() -> _TestCa:
    return new_ca()


def _records() -> dict[str, dict[str, str]]:
    return {
        "rly-here": {"target_node": "nde-b", "origin_host": _REQUESTER_HOST},
        "rly-elsewhere": {"target_node": "nde-c", "origin_host": _REQUESTER_HOST},
    }


def test_a_content_session_reaches_only_the_content_uplink(ca: _TestCa) -> None:
    content = _Uplink("nde-b", _records())
    resident = _Uplink("nde-b", {})

    async def run() -> None:
        bridges = {PeerPurpose.CONTENT: content, PeerPurpose.RESIDENT: resident}
        async with _listener(ca, bridges) as endpoint:
            requester = ca.material("node-a", _REQUESTER_HOST)
            await _send(endpoint, requester, PeerPurpose.CONTENT, "rly-here")
            await _send(endpoint, requester, PeerPurpose.RESIDENT, "rly-resident")

    asyncio.run(run())
    assert [frame.session_id for frame in content.frames] == ["rly-here"]
    assert [frame.session_id for frame in resident.frames] == ["rly-resident"]


@pytest.mark.parametrize(
    ("session_id", "dialer"),
    [
        ("rly-elsewhere", "requester"),
        ("rly-unknown", "requester"),
        ("rly-here", "other-node"),
        ("rly-here", "common-name-only"),
    ],
)
def test_a_content_session_not_routed_to_this_dialer_reaches_nothing(
    ca: _TestCa, session_id: str, dialer: str
) -> None:
    content = _Uplink("nde-b", _records())
    material = {
        "requester": ca.material("node-a", _REQUESTER_HOST),
        "other-node": ca.material("node-c", "10.0.0.9"),
        "common-name-only": ca.named_material(_REQUESTER_HOST, "node-c.example"),
    }[dialer]

    async def run() -> None:
        async with _listener(ca, {PeerPurpose.CONTENT: content}) as endpoint:
            await _send(endpoint, material, PeerPurpose.CONTENT, session_id)

    asyncio.run(run())
    assert content.frames == [] and content.bound == []


def test_a_node_without_a_content_uplink_refuses_a_content_dialer(
    ca: _TestCa,
) -> None:
    async def run() -> None:
        resident = _Uplink("nde-b", {})
        async with _listener(ca, {PeerPurpose.RESIDENT: resident}) as endpoint:
            with pytest.raises(PeerAcceptError) as refused:
                await open_accepted_connection(
                    endpoint,
                    client_context(ca.material("node-a", _REQUESTER_HOST)),
                    5.0,
                    PeerPurpose.CONTENT,
                )
            assert refused.value.outcome is None

    asyncio.run(run())


def test_a_holder_that_ends_ends_the_content_connections_to_it_at_once(
    ca: _TestCa,
) -> None:
    records = {
        "rly-to-h": {
            "target_node": "nde-b",
            "target_worker": "wkr-h",
            "origin_host": _REQUESTER_HOST,
        },
        "rly-to-g": {
            "target_node": "nde-b",
            "target_worker": "wkr-g",
            "origin_host": _REQUESTER_HOST,
        },
    }
    content = _Uplink("nde-b", records)
    resident = _Uplink("nde-b", {})

    async def run() -> tuple[bytes, bool, bool]:
        listener = NodePeerListener(
            endpoint="127.0.0.1:0",
            material=ca.material("node-b", "127.0.0.1"),
            bridges=cast(
                Any, {PeerPurpose.CONTENT: content, PeerPurpose.RESIDENT: resident}
            ),
        )
        await listener.start()
        endpoint = f"127.0.0.1:{listener.port}"
        requester = client_context(ca.material("node-a", _REQUESTER_HOST))
        try:
            lost_reader, lost = await open_accepted_connection(
                endpoint, requester, 5.0, PeerPurpose.CONTENT
            )
            kept_reader, kept = await open_accepted_connection(
                endpoint, requester, 5.0, PeerPurpose.CONTENT
            )
            res_reader, res = await open_accepted_connection(
                endpoint, requester, 5.0, PeerPurpose.RESIDENT
            )
            await write_relay_frame(lost, _frame("rly-to-h"))
            await write_relay_frame(kept, _frame("rly-to-g"))
            await write_relay_frame(res, _frame("rly-to-h"))
            while len(content.frames) < 2 or not resident.frames:
                await asyncio.sleep(0.01)

            await asyncio.to_thread(listener.release_worker, "wkr-h")

            ended = await asyncio.wait_for(lost_reader.read(), 2.0)
            kept_open = not await _closes(kept_reader)
            resident_open = not await _closes(res_reader)
            for writer in (lost, kept, res):
                writer.close()
            return ended, kept_open, resident_open
        finally:
            await listener.stop()

    ended, kept_open, resident_open = asyncio.run(run())
    assert ended == b""
    assert kept_open and resident_open


def test_a_content_session_to_a_worker_that_ended_reaches_nothing(
    ca: _TestCa,
) -> None:
    records = {
        "rly-late": {
            "target_node": "nde-b",
            "target_worker": "wkr-h",
            "origin_host": _REQUESTER_HOST,
        }
    }
    content = _Uplink("nde-b", records)

    async def run() -> None:
        listener = NodePeerListener(
            endpoint="127.0.0.1:0",
            material=ca.material("node-b", "127.0.0.1"),
            bridges=cast(Any, {PeerPurpose.CONTENT: content}),
        )
        await listener.start()
        try:
            await asyncio.to_thread(listener.release_worker, "wkr-h")
            await asyncio.sleep(0.05)
            await _send(
                f"127.0.0.1:{listener.port}",
                ca.material("node-a", _REQUESTER_HOST),
                PeerPurpose.CONTENT,
                "rly-late",
            )
        finally:
            await listener.stop()

    asyncio.run(run())
    assert content.frames == [] and content.bound == []


async def _closes(reader: asyncio.StreamReader) -> bool:
    try:
        return await asyncio.wait_for(reader.read(), 0.3) == b""
    except TimeoutError:
        return False


def test_a_dialer_that_stops_reading_never_holds_the_worker_answering_it(
    ca: _TestCa,
) -> None:
    """The worker's frames answer from its event stream, which must keep moving."""
    content = _Uplink("nde-b", _records())

    async def run() -> tuple[float, bytes]:
        async with _listener(ca, {PeerPurpose.CONTENT: content}) as endpoint:
            reader, writer = await open_accepted_connection(
                endpoint,
                client_context(ca.material("node-a", _REQUESTER_HOST)),
                5.0,
                PeerPurpose.CONTENT,
            )
            await write_relay_frame(writer, _frame("rly-here"))
            for _ in range(200):
                if "rly-here" in content.sinks:
                    break
                await asyncio.sleep(0.01)
            sink = content.sinks["rly-here"]
            answer = RelayFrame(
                kind=RelayFrameKind.DATA,
                session_id="rly-here",
                correlation_id="chg-1",
                operation_id="",
                direction=RelayDirection.TARGET_TO_ORIGIN,
                seq=1,
                payload=b"x" * (64 * 1024),
            )
            started = time.monotonic()
            async with asyncio.timeout(5.0):
                for _ in range(256):
                    await sink.send(answer)
            elapsed = time.monotonic() - started
            with contextlib.suppress(ConnectionError):
                async with asyncio.timeout(5.0):
                    while await reader.read(1 << 20):
                        pass
            tail = await reader.read()
            writer.close()
            return elapsed, tail

    elapsed, tail = asyncio.run(run())
    # 16 MiB answered to a dialer reading nothing: queued at once, and the dialer is
    # dropped past its bound rather than holding the sender.
    assert elapsed < 1.0
    assert tail == b""
