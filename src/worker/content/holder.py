"""Serving the objects this worker holds to the workers control authorizes.

The holder answers one transfer at a time per session: it admits the presented grant
against the ones control pre-delivered, verifies the stored object against the exact
reference that grant names, and streams the bytes back over the session's window. A
refusal is a typed rejection the requester surfaces as a hydration failure — never a
partial or unverified object, and never bytes for a grant this holder was not handed.
"""

import asyncio
import logging
from typing import Any

from shared.content import (
    ContentHydrationError,
    ContentHydrationGrant,
    ContentStoreError,
    GrantRejection,
    HolderGrantGate,
    verify_content,
)
from shared.content.wire import (
    CHUNK_BYTES,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FETCH,
    KIND_HEAD,
    KIND_REJECT,
)
from shared.network.frame_stream import FrameSink
from shared.network.relay_frame import RelayFrame, RelayFrameKind
from shared.network.session import FramedRelaySession, RelaySessionRole

from .store import WorkerContentCache

# Control relays a grant to the holder and the requester independently, so a fetch can
# arrive before this holder's copy of the grant does. A bounded wait lets the two meet
# without weakening the fence: what is waited for is the grant control actually sent,
# and a fetch for a grant that never arrives is still refused.
_GRANT_ARRIVAL_WAIT_SEC = 2.0


class ContentHolder:
    """This worker's end of the transfers it serves."""

    def __init__(
        self,
        *,
        store: WorkerContentCache,
        sink: FrameSink,
        holder_id: str,
        generation: int,
        window_bytes: int = 65536,
        grant_arrival_wait_sec: float = _GRANT_ARRIVAL_WAIT_SEC,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._sink = sink
        self._gate = HolderGrantGate(holder_id=holder_id, generation=generation)
        self._window_bytes = window_bytes
        self._grant_arrival_wait_sec = grant_arrival_wait_sec
        self._logger = logger or logging.getLogger("content-holder")
        self._sessions: dict[str, FramedRelaySession] = {}
        self._serves: dict[str, asyncio.Task[None]] = {}
        self._in_transfer: dict[str, str] = {}
        self._arrivals: dict[str, asyncio.Event] = {}

    @property
    def in_transfer(self) -> frozenset[str]:
        """The digests a transfer is serving right now, which eviction leaves alone."""
        return frozenset(self._in_transfer.values())

    def accept_grant(self, grant: ContentHydrationGrant) -> None:
        """Register a grant control minted against this holder."""
        self._gate.accept(grant)
        if (arrival := self._arrivals.get(grant.grant_id)) is not None:
            arrival.set()

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound frame to its transfer, opening one on a fetch."""
        session = self._sessions.get(frame.session_id)
        if session is None:
            if frame.kind is not RelayFrameKind.DATA or frame.seq != 1:
                return
            session = FramedRelaySession(
                session_id=frame.session_id,
                correlation_id=frame.correlation_id,
                operation_id=frame.operation_id,
                role=RelaySessionRole.TARGET,
                sink=self._sink,
                window_bytes=self._window_bytes,
            )
            self._sessions[frame.session_id] = session
            self._serves[frame.session_id] = asyncio.ensure_future(
                self._serve(frame.session_id, session)
            )
        await session.on_frame(frame)
        if frame.kind is RelayFrameKind.CANCEL:
            self._reap(frame.session_id)

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
        self._in_transfer[session_id] = reference.content_digest
        try:
            data = verify_content(reference, self._store.fetch(reference))
        except (ContentStoreError, ContentHydrationError) as exc:
            self._logger.warning("holder cannot serve %s: %s", session_id, exc)
            await session.send_wire(KIND_REJECT, reason="unavailable")
            return
        await session.send_wire(KIND_HEAD, size_bytes=len(data))
        for start in range(0, len(data), CHUNK_BYTES):
            await session.send_body_wire(KIND_CHUNK, data[start : start + CHUNK_BYTES])
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
        self._in_transfer.pop(session_id, None)
        task = self._serves.pop(session_id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        for session_id in list(self._serves):
            self._reap(session_id)
