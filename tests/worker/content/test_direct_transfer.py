"""A content transfer carried over a connection the requester dials to the holder.

The holder's content listener and the requester's dialer run for real over loopback
mutual TLS; the root relay both ends fall back to is an in-process pair of sinks.
"""

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from shared.content import (
    ContentCarriagePlan,
    ContentHydrationError,
    ContentHydrationGrant,
    ContentPeerRoute,
    ContentReference,
)
from shared.content.wire import CHUNK_BYTES
from shared.network.frame_stream import PeerPurpose
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.mtls_listener import MutualTlsFrameListener
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.schemas.network import Transport
from shared.utils.ids import new_hydration_grant_id, new_relay_session_id
from tests.support.certs import TestCa as _TestCa
from tests.support.certs import new_ca
from worker.content import ContentHolder, ContentHydrationClient, WorkerContentCache
from worker.content.lane_host import _HolderConnection
from worker.content.peer import ContentPeerDialer

_REQUESTER_HOST = "10.0.0.7"
_STALL_SEC = 5.0


class _ToPeer:
    """One direction of the root relay, delivering straight into the peer."""

    def __init__(self) -> None:
        self.peer: Callable[[RelayFrame], Awaitable[None]] | None = None
        self.frames: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.frames.append(frame)
        if self.peer is not None:
            await self.peer(frame)


class _SlowFile:
    """A held copy that reads one chunk at a time, slowly."""

    def __init__(self, fh: BinaryIO, reading: threading.Event) -> None:
        self._fh = fh
        self._reading = reading

    def read(self, size: int = -1) -> bytes:
        self._reading.set()
        time.sleep(0.02)
        return self._fh.read(size)

    def __enter__(self) -> "_SlowFile":
        return self

    def __exit__(self, *_: Any) -> None:
        self._fh.close()


class _SlowCache(WorkerContentCache):
    """A cache whose held copies read slowly, signalling once one is being read."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reading = threading.Event()

    def open_pinned(self, reference: ContentReference) -> BinaryIO:
        return cast(BinaryIO, _SlowFile(super().open_pinned(reference), self.reading))


class _Fabric:
    """A holder behind a content listener, a requester that dials it, and a control
    stand-in that grants each request and routes it as the test says."""

    def __init__(
        self,
        tmp_path: Path,
        ca: _TestCa,
        *,
        requester: MutualTlsMaterial,
        cache: type[WorkerContentCache] = WorkerContentCache,
        transport: Transport = Transport.WORKER_DIRECT,
        transfer_timeout_sec: float = _STALL_SEC,
    ) -> None:
        self.tmp_path = tmp_path
        self.cache = cache(tmp_path / "held")
        self.to_requester = _ToPeer()
        self.to_holder = _ToPeer()
        self.holder = ContentHolder(
            store=self.cache,
            sink=self.to_requester,
            holder_id="wkr-1",
            generation=1,
            grant_arrival_wait_sec=0.2,
            stall_timeout_sec=_STALL_SEC,
        )
        self.listener = MutualTlsFrameListener(
            material=ca.material("node-b", "127.0.0.1"),
            handler=lambda sink, peer: _HolderConnection(self.holder, sink, peer),
            admits=bool,
            purposes=frozenset({PeerPurpose.CONTENT}),
        )
        self.observed: list[str] = []
        self.grants: list[ContentHydrationGrant] = []
        self.transport = transport
        self.endpoint = ""
        # Routes dialed before the listener's own, as an unreachable first candidate.
        self.leading: list[tuple[Transport, str]] = []
        self.client = ContentHydrationClient(
            sink=self.to_holder,
            request_grant=self._grant,
            transfer_timeout_sec=transfer_timeout_sec,
            dialer=ContentPeerDialer(
                ssl_context=client_context(requester),
                connect_budget_sec=2.0,
                observe=lambda _, route, outcome: self.observed.append(outcome.value),
            ),
        )
        self.to_requester.peer = self.client.on_frame
        self.to_holder.peer = self.holder.on_frame

    def routes(self) -> list[tuple[Transport, str]]:
        if self.transport == Transport.CONTROL_RELAY:
            return []
        return [*self.leading, (self.transport, self.endpoint)]

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        await self.listener.start_on_socket(sock)
        self.endpoint = f"127.0.0.1:{self.listener.port}"

    def _grant(
        self,
        reference: ContentReference,
        task_id: str,
        peer_only: bool,
        request_id: int,
    ) -> None:
        grant = ContentHydrationGrant(
            grant_id=new_hydration_grant_id(),
            reference=reference,
            requester_subject="wkr-2",
            requester_origin_id="nde-a",
            holder_id="wkr-1",
            holder_generation=1,
            transfer_session_id=new_relay_session_id(),
            expires_at_epoch=time.time() + 30,
        )
        self.grants.append(grant)
        carriage = ContentCarriagePlan(
            transfer_session_id=grant.transfer_session_id,
            grant_id=grant.grant_id,
            holder_worker_id="wkr-1",
            holder_generation=1,
            listener_generation=1,
            routes=(
                tuple(
                    ContentPeerRoute(transport=transport, endpoint=endpoint)
                    for transport, endpoint in self.routes()
                )
            ),
            fresh_for_sec=30.0,
        )
        self.holder.accept_grant(grant, _REQUESTER_HOST)
        asyncio.get_running_loop().call_soon(
            self.client.deliver_grant, request_id, grant, carriage
        )

    async def read(self, reference: ContentReference, peer_only: bool = False) -> bytes:
        staging = self.tmp_path / "staging"
        staging.mkdir(exist_ok=True)
        path = await self.client.hydrate_to_file(reference, "tsk-1", staging, peer_only)
        try:
            return path.read_bytes()
        finally:
            path.unlink()

    def staged(self) -> list[Path]:
        staging = self.tmp_path / "staging"
        return list(staging.iterdir()) if staging.exists() else []

    def relayed_data(self) -> list[RelayFrame]:
        return [
            frame
            for frame in self.to_holder.frames + self.to_requester.frames
            if frame.kind is RelayFrameKind.DATA
        ]


@contextlib.asynccontextmanager
async def _fabric(tmp_path: Path, ca: _TestCa, **kwargs: Any) -> AsyncIterator[_Fabric]:
    fabric = _Fabric(
        tmp_path,
        ca,
        requester=kwargs.pop("requester", None) or _requester(ca),
        **kwargs,
    )
    await fabric.start()
    try:
        yield fabric
    finally:
        await fabric.listener.stop()
        await fabric.holder.aclose()


def _requester(ca: _TestCa) -> MutualTlsMaterial:
    return ca.material("node-a", _REQUESTER_HOST)


@pytest.fixture(scope="module")
def ca() -> _TestCa:
    return new_ca()


@pytest.mark.asyncio
async def test_a_direct_transfer_carries_every_byte_off_the_relay(
    tmp_path, ca: _TestCa
) -> None:
    async with _fabric(tmp_path, ca) as fabric:
        data = bytes(range(256)) * (3 * CHUNK_BYTES // 256 + 7)
        reference = fabric.cache.write("org", data)

        assert await fabric.read(reference, peer_only=True) == data

    assert fabric.to_holder.frames == [] and fabric.to_requester.frames == []
    assert fabric.observed == ["verified"]
    assert len(fabric.grants) == 1
    assert fabric.staged() == []


@pytest.mark.asyncio
async def test_a_grant_redeemed_from_another_node_releases_nothing(
    tmp_path, ca: _TestCa
) -> None:
    other_node = ca.material("node-c", "10.0.0.9")
    async with _fabric(tmp_path, ca, requester=other_node) as fabric:
        reference = fabric.cache.write("org", b"payload")

        with pytest.raises(ContentHydrationError, match="wrong_requester"):
            await fabric.read(reference, peer_only=True)

    # Each attempt was refused before admission, and each refusal demotes the pair:
    # this node's certificate does not fit the host it advertises.
    assert len(fabric.grants) == 2
    assert fabric.observed == ["verified", "tls_failure", "verified", "tls_failure"]
    assert fabric.staged() == []


@pytest.mark.asyncio
async def test_a_grant_that_never_reaches_the_holder_demotes_nothing(
    tmp_path, ca: _TestCa, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _fabric(tmp_path, ca) as fabric:
        reference = fabric.cache.write("org", b"payload")
        monkeypatch.setattr(fabric.holder, "accept_grant", lambda *_: None)

        with pytest.raises(ContentHydrationError, match="unknown_grant"):
            await fabric.read(reference, peer_only=True)

    assert "tls_failure" not in fabric.observed


@pytest.mark.asyncio
async def test_a_common_name_matching_the_requester_host_is_not_its_binding(
    tmp_path, ca: _TestCa
) -> None:
    impostor = ca.named_material(_REQUESTER_HOST, "node-c.example")
    async with _fabric(tmp_path, ca, requester=impostor) as fabric:
        reference = fabric.cache.write("org", b"payload")

        with pytest.raises(ContentHydrationError, match="wrong_requester"):
            await fabric.read(reference, peer_only=True)


@pytest.mark.asyncio
async def test_a_dial_that_fails_sends_the_same_grant_over_the_relay(
    tmp_path, ca: _TestCa
) -> None:
    async with _fabric(tmp_path, ca) as fabric:
        reference = fabric.cache.write("org", b"payload")
        await fabric.listener.stop()

        assert await fabric.read(reference) == b"payload"

    assert len(fabric.grants) == 1
    assert fabric.observed == ["connect_failure"]
    assert fabric.relayed_data()


@pytest.mark.asyncio
async def test_a_peer_only_read_never_falls_to_the_relay(tmp_path, ca: _TestCa) -> None:
    async with _fabric(tmp_path, ca) as fabric:
        reference = fabric.cache.write("org", b"payload")
        await fabric.listener.stop()

        with pytest.raises(ContentHydrationError, match="no peer transport"):
            await fabric.read(reference, peer_only=True)

    assert fabric.to_holder.frames == []
    assert len(fabric.grants) == 1


@pytest.mark.asyncio
async def test_a_holder_lost_mid_transfer_fails_the_read_promptly(
    tmp_path, ca: _TestCa
) -> None:
    async with _fabric(tmp_path, ca, cache=_SlowCache) as fabric:
        data = b"z" * (64 * CHUNK_BYTES)
        reference = fabric.cache.write("org", data)

        async def lose_holder() -> None:
            # Lose the holder once its transfer is under way.
            await asyncio.to_thread(cast(_SlowCache, fabric.cache).reading.wait, 5.0)
            await fabric.listener.stop()

        loss = asyncio.ensure_future(lose_holder())
        started = time.monotonic()
        with pytest.raises(ContentHydrationError):
            await fabric.read(reference, peer_only=True)
        elapsed = time.monotonic() - started
        await loss

    assert elapsed < _STALL_SEC / 2
    assert "route_failure" in fabric.observed
    assert fabric.staged() == []
    # The retry after the loss took a fresh grant rather than reusing the first.
    assert len(fabric.grants) == 2
    assert fabric.grants[0].grant_id != fabric.grants[1].grant_id


@pytest.mark.asyncio
async def test_a_held_copy_that_is_not_the_object_is_never_delivered(
    tmp_path, ca: _TestCa
) -> None:
    async with _fabric(tmp_path, ca) as fabric:
        data = b"q" * (2 * CHUNK_BYTES)
        reference = fabric.cache.write("org", data)
        held = next(p for p in (tmp_path / "held").rglob("*") if p.is_file())
        held.chmod(0o644)
        held.write_bytes(b"r" * len(data))

        with pytest.raises(ContentHydrationError):
            await fabric.read(reference, peer_only=True)

    assert not fabric.cache.holds(reference)
    assert "route_failure" not in fabric.observed
    assert fabric.staged() == []


@pytest.mark.asyncio
async def test_one_grant_fetched_twice_at_once_is_served_once(
    tmp_path, ca: _TestCa
) -> None:
    async with _fabric(tmp_path, ca, transfer_timeout_sec=1.0) as fabric:
        reference = fabric.cache.write("org", b"payload")
        grant = ContentHydrationGrant(
            grant_id=new_hydration_grant_id(),
            reference=reference,
            requester_subject="wkr-2",
            requester_origin_id="nde-a",
            holder_id="wkr-1",
            holder_generation=1,
            transfer_session_id=new_relay_session_id(),
            expires_at_epoch=time.time() + 30,
        )
        fabric.holder.accept_grant(grant, _REQUESTER_HOST)
        carriage = ContentCarriagePlan(
            transfer_session_id=grant.transfer_session_id,
            grant_id=grant.grant_id,
            holder_worker_id="wkr-1",
            holder_generation=1,
            listener_generation=1,
            routes=(
                ContentPeerRoute(
                    transport=Transport.WORKER_DIRECT, endpoint=fabric.endpoint
                ),
            ),
            fresh_for_sec=30.0,
        )
        dialer = ContentPeerDialer(
            ssl_context=client_context(_requester(ca)),
            connect_budget_sec=2.0,
            observe=lambda *_: None,
        )

        async def fetch() -> bytes:
            client = ContentHydrationClient(
                sink=fabric.to_holder,
                request_grant=lambda *_: None,
                transfer_timeout_sec=1.0,
                dialer=dialer,
            )
            staging = tmp_path / f"staging-{id(client)}"
            staging.mkdir()
            reading = asyncio.ensure_future(
                client.hydrate_to_file(reference, "tsk-1", staging, True)
            )
            await asyncio.sleep(0)
            client.deliver_grant(1, grant, carriage)
            return (await reading).read_bytes()

        outcomes = await asyncio.gather(fetch(), fetch(), return_exceptions=True)

    assert sorted(type(o).__name__ for o in outcomes) == [
        "ContentHydrationError",
        "bytes",
    ]


@pytest.mark.asyncio
async def test_a_dialed_session_ignores_frames_from_any_other_carriage(
    tmp_path,
) -> None:
    cache = WorkerContentCache(tmp_path / "held")
    relay = _ToPeer()
    holder = ContentHolder(store=cache, sink=relay, holder_id="wkr-1", generation=1)
    dialed = _ToPeer()

    def frame(kind: RelayFrameKind, seq: int = 0) -> RelayFrame:
        return RelayFrame(
            kind=kind,
            session_id="rly-1",
            correlation_id="chg-1",
            operation_id="",
            direction=RelayDirection.ORIGIN_TO_TARGET,
            seq=seq,
            payload=b"{}" if kind is RelayFrameKind.DATA else b"",
        )

    await holder.on_frame(frame(RelayFrameKind.DATA, 1), dialed, None)
    await holder.on_frame(frame(RelayFrameKind.CANCEL))
    assert holder._sessions.keys() == {"rly-1"}
    holder.drop_connection(dialed)
    assert holder._sessions == {}
    await holder.aclose()


@pytest.mark.asyncio
async def test_a_route_that_does_not_connect_gives_way_to_the_next_under_one_grant(
    tmp_path, ca: _TestCa
) -> None:
    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.bind(("127.0.0.1", 0))
    dead = f"127.0.0.1:{closed.getsockname()[1]}"
    closed.close()
    async with _fabric(tmp_path, ca, transport=Transport.NODE_RELAY) as fabric:
        fabric.leading = [(Transport.WORKER_DIRECT, dead)]
        reference = fabric.cache.write("org", b"payload")

        assert await fabric.read(reference, peer_only=True) == b"payload"

    assert len(fabric.grants) == 1
    assert fabric.observed == ["connect_failure", "verified"]
    assert fabric.to_holder.frames == []
