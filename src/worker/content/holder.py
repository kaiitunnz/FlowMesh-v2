"""Serving the objects this worker holds to the workers control authorizes.

The holder answers one transfer at a time per session: it admits the presented grant
against the ones control pre-delivered, opens its copy of the exact object that grant
names, and streams it back over the session's window, hashing as it reads. A copy that
turns out not to be the object ends the transfer with a refusal instead of its last
message and is dropped from the cache. A refusal is a typed rejection the requester
surfaces as a hydration failure — never a completed transfer of bytes the reference does
not name, and never bytes for a grant this holder was not handed.

A transfer arrives over the root relay or over a connection the requester dialed, and
every frame of it answers over the carriage it arrived on: a session belongs to the
carriage that opened it, and frames for it from any other are dropped. A dialed
connection also proves which node dialed it, so its fetch is served only when the
connection's certificate covers the host control named for the grant's requester.
"""

import asyncio
import logging
import time
from typing import Any, BinaryIO

from shared.content import (
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
    ContentStoreError,
    ContentVerifier,
    GrantRejection,
    HolderGrantGate,
)
from shared.content.config import DEFAULT_TRANSFER_WINDOW_BYTES
from shared.content.wire import (
    CHUNK_BYTES,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FETCH,
    KIND_HEAD,
    KIND_REJECT,
)
from shared.network.frame_stream import FrameSink
from shared.network.mtls_listener import PeerContext
from shared.network.relay_frame import RelayFrame, RelayFrameKind
from shared.network.session import FramedRelaySession, RelaySessionRole

from .store import WorkerContentCache

# Control relays a grant to the holder and the requester independently, so a fetch can
# arrive before this holder's copy of the grant does. A bounded wait lets the two meet
# without weakening the fence: what is waited for is the grant control actually sent,
# and a fetch for a grant that never arrives is still refused.
_GRANT_ARRIVAL_WAIT_SEC = 2.0

# A transfer only advances as the requester drains and grants window, so a requester
# that stops — a timeout, a crash, a killed worker — would otherwise leave this serve
# blocked forever, holding its task, the object's bytes, and the digest against
# eviction. A serve that makes no progress for this long is given up, so a peer that
# goes quiet costs one transfer, while one that keeps reading runs as long as it needs.
_STALL_TIMEOUT_SEC = 60.0


class ContentHolder:
    """This worker's end of the transfers it serves."""

    def __init__(
        self,
        *,
        store: WorkerContentCache,
        sink: FrameSink,
        holder_id: str,
        generation: int,
        window_bytes: int = DEFAULT_TRANSFER_WINDOW_BYTES,
        grant_arrival_wait_sec: float = _GRANT_ARRIVAL_WAIT_SEC,
        stall_timeout_sec: float = _STALL_TIMEOUT_SEC,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._sink = sink
        self._gate = HolderGrantGate(holder_id=holder_id, generation=generation)
        self._window_bytes = window_bytes
        self._grant_arrival_wait_sec = grant_arrival_wait_sec
        self._stall_timeout_sec = stall_timeout_sec
        self._logger = logger or logging.getLogger("content-holder")
        self._sessions: dict[str, FramedRelaySession] = {}
        self._carriages: dict[str, FrameSink] = {}
        self._peers: dict[str, PeerContext] = {}
        self._requesters: dict[str, tuple[str, float]] = {}
        self._serves: dict[str, asyncio.Task[None]] = {}
        self._in_transfer: dict[str, str] = {}
        self._serving: frozenset[str] = frozenset()
        self._arrivals: dict[str, asyncio.Event] = {}

    def rebind(self, holder_id: str, generation: int) -> None:
        """Serve only grants minted for this worker's new registration."""
        self._gate = HolderGrantGate(holder_id=holder_id, generation=generation)

    @property
    def in_transfer(self) -> frozenset[str]:
        """The digests a transfer is serving right now, which eviction leaves alone.

        Eviction runs off this loop, so what it reads is a snapshot this loop replaces
        rather than the live mapping it would otherwise iterate mid-mutation.
        """
        return self._serving

    def _track(self, session_id: str, digest: str | None) -> None:
        if digest is None:
            self._in_transfer.pop(session_id, None)
        else:
            self._in_transfer[session_id] = digest
        self._serving = frozenset(self._in_transfer.values())

    def accept_grant(
        self, grant: ContentHydrationGrant, requester_host: str | None = None
    ) -> None:
        """Register a grant control minted against this holder.

        ``requester_host`` is the host control expects the requester's node to dial
        from; a dialed fetch of this grant is served only to a certificate covering it.
        """
        now = time.time()
        for grant_id, (_, expires_at) in list(self._requesters.items()):
            if now > expires_at:
                del self._requesters[grant_id]
        if requester_host:
            self._requesters[grant.grant_id] = (requester_host, grant.expires_at_epoch)
        self._gate.accept(grant)
        if (arrival := self._arrivals.get(grant.grant_id)) is not None:
            arrival.set()

    async def on_frame(
        self,
        frame: RelayFrame,
        sink: FrameSink | None = None,
        peer: PeerContext | None = None,
    ) -> None:
        """Route one inbound frame to its transfer, opening one on a fetch.

        ``sink`` is the dialed connection the frame arrived on, which also answers it,
        and ``peer`` what its listener verified; a frame from the root relay has
        neither.
        """
        carriage = sink or self._sink
        session = self._sessions.get(frame.session_id)
        if session is None:
            if frame.kind is not RelayFrameKind.DATA or frame.seq != 1:
                return
            session = FramedRelaySession(
                session_id=frame.session_id,
                correlation_id=frame.correlation_id,
                operation_id=frame.operation_id,
                role=RelaySessionRole.TARGET,
                sink=carriage,
                window_bytes=self._window_bytes,
            )
            self._sessions[frame.session_id] = session
            self._carriages[frame.session_id] = carriage
            if peer is not None:
                self._peers[frame.session_id] = peer
            self._serves[frame.session_id] = asyncio.ensure_future(
                self._serve(frame.session_id, session)
            )
        elif self._carriages.get(frame.session_id) is not carriage:
            return
        await session.on_frame(frame)
        if frame.kind is RelayFrameKind.CANCEL:
            self._reap(frame.session_id)

    def drop_connection(self, sink: FrameSink) -> None:
        """End every transfer a closed dialed connection was carrying."""
        for session_id, carriage in list(self._carriages.items()):
            if carriage is sink:
                self._reap(session_id)

    async def _serve(self, session_id: str, session: FramedRelaySession) -> None:
        try:
            message = await session.recv_wire(timeout=30.0)
            if message is None or message.get("kind") != KIND_FETCH:
                return
            await self._serve_fetch(session_id, session, message)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("content transfer %s failed", session_id)
        finally:
            self._reap(session_id)

    async def _serve_fetch(
        self, session_id: str, session: FramedRelaySession, message: dict[str, Any]
    ) -> None:
        try:
            grant = ContentHydrationGrant.model_validate(message["grant"])
        except (KeyError, ValueError):
            await session.send_wire(KIND_REJECT, reason="malformed_grant")
            return
        if grant.transfer_session_id != session_id:
            self._logger.warning(
                "refused content transfer %s: grant names another session", session_id
            )
            await session.send_wire(KIND_REJECT, reason="wrong_session")
            return
        if (peer := self._peers.get(session_id)) is not None and peer.identities:
            # The requester's host arrives beside the grant, so a dialed fetch that
            # outruns the grant waits for it before the binding can be checked.
            if grant.grant_id not in self._requesters:
                await self._await_grant(grant.grant_id)
            expected = self._requesters.get(grant.grant_id)
            if expected is None or not peer.names.covers(expected[0]):
                self._logger.warning(
                    "refused content transfer %s: the dialer is not its requester",
                    session_id,
                )
                await session.send_wire(KIND_REJECT, reason="wrong_requester")
                return
        rejection = self._gate.admit(grant)
        if rejection is GrantRejection.UNKNOWN_GRANT and await self._await_grant(
            grant.grant_id
        ):
            rejection = self._gate.admit(grant)
        if rejection is not None:
            self._logger.warning(
                "refused content transfer %s: %s", session_id, rejection.value
            )
            await session.send_wire(KIND_REJECT, reason=rejection.value)
            return
        reference = grant.reference
        self._track(session_id, reference.content_digest)
        try:
            # The open file outlives an eviction that unlinks it mid-transfer, so the
            # copy this transfer serves cannot be taken out from under it.
            fh = await asyncio.to_thread(self._store.open_pinned, reference)
        except (ContentStoreError, OSError) as exc:
            self._logger.warning("holder cannot serve %s: %s", session_id, exc)
            await session.send_wire(KIND_REJECT, reason="unavailable")
            return
        with fh:
            await self._stream(session, reference, fh)

    async def _stream(
        self,
        session: FramedRelaySession,
        reference: ContentReference,
        fh: BinaryIO,
    ) -> None:
        verifier = ContentVerifier(reference)

        def read_chunk() -> bytes:
            # Reading and hashing are both work proportional to the object, and this
            # loop also carries every other transfer's frames and the grants control
            # delivers, so neither runs on it.
            chunk = fh.read(CHUNK_BYTES)
            verifier.update(chunk)
            return chunk

        loop = asyncio.get_running_loop()
        async with asyncio.timeout(self._stall_timeout_sec) as stall:
            await session.send_wire(KIND_HEAD, size_bytes=reference.size_bytes)
            try:
                while chunk := await asyncio.to_thread(read_chunk):
                    stall.reschedule(loop.time() + self._stall_timeout_sec)
                    await session.send_body_wire(KIND_CHUNK, chunk)
                await asyncio.to_thread(verifier.finish)
            except (ContentHydrationError, OSError) as exc:
                self._logger.warning(
                    "held copy of %s is not the object: %s",
                    reference.content_digest,
                    exc,
                )
                self._store.drop(reference)
                await session.send_wire(KIND_REJECT, reason="unavailable")
                return
            stall.reschedule(loop.time() + self._stall_timeout_sec)
            await session.send_wire(KIND_DONE)

    async def _await_grant(self, grant_id: str) -> bool:
        """Wait out the delivery race for one grant; False once the wait is spent."""
        arrival = self._arrivals.setdefault(grant_id, asyncio.Event())
        try:
            async with asyncio.timeout(self._grant_arrival_wait_sec):
                await arrival.wait()
        except TimeoutError:
            return False
        finally:
            self._arrivals.pop(grant_id, None)
        return True

    def _reap(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._carriages.pop(session_id, None)
        self._peers.pop(session_id, None)
        self._track(session_id, None)
        task = self._serves.pop(session_id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        for session_id in list(self._serves):
            self._reap(session_id)
