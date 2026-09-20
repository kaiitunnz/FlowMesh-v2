"""One content transfer end to end between a requesting worker and its holder."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable

import pytest

from shared.content import (
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
    reference_for,
)
from shared.content.wire import KIND_FETCH, KIND_REJECT
from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession, RelaySessionRole
from shared.utils.ids import new_hydration_grant_id, new_relay_session_id
from worker.content import ContentHolder, ContentHydrationClient, WorkerContentCache

_BODY = b'{"prompts": ["one", "two"]}'


class _ToPeer:
    """One direction's sink, delivering straight into the peer's frame handler."""

    def __init__(self) -> None:
        self.peer: Callable[[RelayFrame], Awaitable[None]] | None = None
        self.frames: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.frames.append(frame)
        if self.peer is not None:
            await self.peer(frame)


def _grant(reference: ContentReference, **overrides) -> ContentHydrationGrant:
    fields = {
        "grant_id": new_hydration_grant_id(),
        "reference": reference,
        "requester_subject": "wkr-2",
        "requester_origin_id": "rog-2",
        "holder_id": "wkr-1",
        "holder_generation": 1,
        "transfer_session_id": new_relay_session_id(),
        "expires_at_epoch": time.time() + 30,
    }
    return ContentHydrationGrant(**{**fields, **overrides})


class _Pair:
    """A holder and a requester wired to each other over in-process sinks."""

    def __init__(self, tmp_path, *, holder_generation: int = 1) -> None:
        self.store_root = tmp_path / "held"
        self.store = WorkerContentCache(self.store_root, retain_sec=60.0)
        self.to_requester = _ToPeer()
        self.to_holder = _ToPeer()
        self.granted: list[ContentReference] = []
        self.holder = ContentHolder(
            store=self.store,
            sink=self.to_requester,
            holder_id="wkr-1",
            generation=holder_generation,
            grant_arrival_wait_sec=0.05,
        )
        self.client = ContentHydrationClient(
            sink=self.to_holder,
            request_grant=lambda reference, task_id: self.granted.append(reference),
            transfer_timeout_sec=5.0,
        )
        self.to_requester.peer = self.client.on_frame
        self.to_holder.peer = self.holder.on_frame


@pytest.mark.asyncio
async def test_a_granted_transfer_delivers_the_verified_object(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)

    assert await hydration == _BODY
    assert pair.granted == [reference]


@pytest.mark.asyncio
async def test_a_transfer_carries_no_object_identity_in_its_frames(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)
    await hydration

    for frame in pair.to_holder.frames + pair.to_requester.frames:
        # What a relaying server may read is routing only: the grant id correlates the
        # transfer, and the object itself is inside the opaque payload.
        assert frame.correlation_id == grant.grant_id
        assert reference.content_digest not in (
            frame.session_id + frame.correlation_id + frame.operation_id
        )
    assert any(frame.payload for frame in pair.to_requester.frames)


@pytest.mark.asyncio
async def test_a_fetch_waits_for_a_grant_still_in_flight(tmp_path) -> None:
    # Control relays the grant to each end on its own path, so the fetch can arrive
    # first; the holder waits for the grant it was sent rather than refusing it.
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)
    await asyncio.sleep(0)
    pair.holder.accept_grant(grant)

    assert await hydration == _BODY


@pytest.mark.asyncio
async def test_a_grant_the_holder_never_received_serves_nothing(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    forged = _grant(reference)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(forged)

    with pytest.raises(ContentHydrationError, match="unknown_grant"):
        await hydration


@pytest.mark.asyncio
async def test_an_expired_grant_serves_nothing(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference, expires_at_epoch=time.time() - 1)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)

    with pytest.raises(ContentHydrationError, match="expired"):
        await hydration


@pytest.mark.asyncio
async def test_a_grant_for_a_superseded_holder_incarnation_serves_nothing(
    tmp_path,
) -> None:
    pair = _Pair(tmp_path, holder_generation=2)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)

    with pytest.raises(ContentHydrationError, match="stale_generation"):
        await hydration


@pytest.mark.asyncio
async def test_an_object_the_holder_lost_is_a_typed_failure(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = reference_for("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)

    with pytest.raises(ContentHydrationError, match="unavailable"):
        await hydration


@pytest.mark.asyncio
async def test_a_denied_request_fails_with_controls_reason(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = reference_for("local", _BODY, media_type="application/json")

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_denial(reference, "no binding authorizes this reference")

    with pytest.raises(ContentHydrationError, match="no binding"):
        await hydration


@pytest.mark.asyncio
async def test_a_grant_is_good_for_one_transfer(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    first = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)
    assert await first == _BODY

    replay = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)
    with pytest.raises(ContentHydrationError, match="consumed"):
        await replay


@pytest.mark.asyncio
async def test_an_object_larger_than_one_chunk_reassembles(tmp_path) -> None:
    pair = _Pair(tmp_path)
    body = bytes(range(256)) * 2048
    reference = pair.store.write("local", body)
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)

    assert await hydration == body
    chunks = [f for f in pair.to_requester.frames if f.payload]
    assert len(chunks) > 3  # head, several chunks, done


@pytest.mark.asyncio
async def test_a_transfer_in_flight_holds_its_object_against_the_sweep(
    tmp_path,
) -> None:
    """A copy being served survives a sweep that would otherwise drop it.

    Eviction and a transfer run on the same worker, so a copy whose retention has
    elapsed can come up for eviction while a peer is still reading it. Dropping it
    mid-transfer would fail a read that was already authorized and underway.
    """
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    reading = threading.Event()
    release = threading.Event()
    served = pair.store.fetch

    def _blocking_fetch(ref: ContentReference) -> bytes:
        reading.set()
        release.wait(5)
        return served(ref)

    pair.store.fetch = _blocking_fetch  # type: ignore[assignment]
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    hydration = asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1"))
    await asyncio.sleep(0)
    pair.client.deliver_grant(grant)
    await asyncio.to_thread(reading.wait, 5)

    # Mid-serve: the object is named as in transfer, and a sweep with nothing retained
    # leaves it alone while every other copy goes.
    assert reference.content_digest in pair.holder.in_transfer
    spare = pair.store.write("local", b"another object", media_type="text/plain")
    aged = WorkerContentCache(pair.store_root, retain_sec=0.0)
    assert aged.evict_aged(in_transfer=pair.holder.in_transfer) == 1
    assert pair.store.holds(reference) and not pair.store.holds(spare)

    release.set()
    assert await hydration == _BODY


@pytest.mark.asyncio
async def test_two_reads_of_one_object_each_take_their_own_grant(tmp_path) -> None:
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    first, second = _grant(reference), _grant(reference)
    pair.holder.accept_grant(first)
    pair.holder.accept_grant(second)

    reads = [
        asyncio.ensure_future(pair.client.hydrate(reference, "tsk-1")),
        asyncio.ensure_future(pair.client.hydrate(reference, "tsk-2")),
    ]
    await asyncio.sleep(0)
    pair.client.deliver_grant(first)
    pair.client.deliver_grant(second)

    assert await asyncio.gather(*reads) == [_BODY, _BODY]
    assert len(pair.granted) == 2  # each read asked for its own authorization


@pytest.mark.asyncio
async def test_a_grant_naming_another_session_serves_nothing(tmp_path) -> None:
    """A grant names the session its bytes flow over, so it is good on that one only.

    Control mints the grant and its transfer session together. A requester presenting
    a real grant over a session control did not pair it with is refused, rather than
    the pairing being a field nothing reads.
    """
    pair = _Pair(tmp_path)
    reference = pair.store.write("local", _BODY, media_type="application/json")
    grant = _grant(reference)
    pair.holder.accept_grant(grant)

    elsewhere = FramedRelaySession(
        session_id=new_relay_session_id(),
        correlation_id=grant.grant_id,
        role=RelaySessionRole.ORIGIN,
        sink=pair.to_holder,
    )
    pair.to_requester.peer = elsewhere.on_frame
    await elsewhere.send_wire(KIND_FETCH, grant=grant.model_dump(mode="json"))
    reply = await elsewhere.recv_wire(timeout=5.0)

    assert reply is not None and reply["kind"] == KIND_REJECT
    assert reply["reason"] == "wrong_session"
    assert pair.holder.in_transfer == frozenset()
