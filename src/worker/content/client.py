"""Hydrating an object this worker does not hold.

The consumer side of a transfer: the worker asks the control plane for a grant on one
exact reference, and control answers with a grant naming the holder it resolved or a
typed denial. With a grant in hand the worker opens the transfer, streams the object
from that holder, and verifies the assembled bytes against the reference before handing
them back. Nothing here decides that the worker may read the object — control does,
against the consumer binding that entitles it — and nothing here recreates an outcome,
re-resolves an input, or releases a credit when a transfer fails: an unavailable holder
is a typed hydration failure and the consumer's own recovery decides what follows.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Sequence

from shared.content import (
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
    verify_content,
)
from shared.content.wire import (
    KIND_CHUNK,
    KIND_DONE,
    KIND_FETCH,
    KIND_HEAD,
    KIND_REJECT,
)
from shared.network.frame_stream import FrameSink
from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession, RelaySessionRole

# Asks the control plane to authorize hydrating one reference for one task.
RequestGrant = Callable[[ContentReference, str], None]

# Reports the objects this worker holds, each by its scope and digest, to the holder
# directory. One call carries a whole report so a periodic re-report is one message.
AnnounceHolding = Callable[[Sequence[tuple[str, str]]], None]

# Keyed by the object a request waits on: control answers a grant or a typed denial
# naming that same reference, so an answer finds its request by what it is about.
_WaitKey = tuple[str, str]
_Answer = asyncio.Future["ContentHydrationGrant | str"]


class GrantDenied(ContentHydrationError):
    """The control plane refused to authorize this hydration."""


class ContentHydrationClient:
    """This worker's end of the transfers it requests."""

    def __init__(
        self,
        *,
        sink: FrameSink,
        request_grant: RequestGrant,
        transfer_timeout_sec: float = 60.0,
        window_bytes: int = 65536,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sink = sink
        self._request_grant = request_grant
        self._timeout = transfer_timeout_sec
        self._window_bytes = window_bytes
        self._logger = logger or logging.getLogger("content-hydration")
        self._waiters: dict[_WaitKey, deque[_Answer]] = {}
        self._sessions: dict[str, FramedRelaySession] = {}

    async def hydrate(self, reference: ContentReference, task_id: str) -> bytes:
        """Fetch and verify one object from whichever holder control resolves."""
        grant = await self._grant_for(reference, task_id)
        session = FramedRelaySession(
            session_id=grant.transfer_session_id,
            correlation_id=grant.grant_id,
            role=RelaySessionRole.ORIGIN,
            sink=self._sink,
            window_bytes=self._window_bytes,
        )
        self._sessions[grant.transfer_session_id] = session
        try:
            return verify_content(reference, await self._transfer(session, grant))
        finally:
            self._sessions.pop(grant.transfer_session_id, None)

    async def _grant_for(
        self, reference: ContentReference, task_id: str
    ) -> ContentHydrationGrant:
        """Ask control to authorize this read, and wait for its answer.

        Each request waits on its own answer because a grant is good for one transfer:
        two tasks reading the same object each ask, and each takes the grant minted for
        it rather than sharing one that the second would find already consumed.
        """
        key = (reference.authorization_scope, reference.content_digest)
        pending: _Answer = asyncio.get_running_loop().create_future()
        waiting = self._waiters.setdefault(key, deque())
        waiting.append(pending)
        # Armed before the request goes up, so an answer that comes straight back finds
        # a waiter rather than arriving at nobody.
        self._request_grant(reference, task_id)
        try:
            answer = await asyncio.wait_for(pending, self._timeout)
        except TimeoutError as exc:
            raise ContentHydrationError(
                f"no hydration grant for {reference.content_digest} in time"
            ) from exc
        finally:
            self._forget(key, pending)
        if isinstance(answer, str):
            raise GrantDenied(
                f"hydration of {reference.content_digest} denied: {answer}"
            )
        return answer

    async def _transfer(
        self, session: FramedRelaySession, grant: ContentHydrationGrant
    ) -> bytes:
        await session.send_wire(KIND_FETCH, grant=grant.model_dump(mode="json"))
        chunks: list[bytes] = []
        while True:
            received = await session.recv_body_wire(timeout=self._timeout)
            if received is None:
                raise ContentHydrationError(
                    f"content transfer {session.session_id} ended without the object"
                )
            message, body = received
            kind = message.get("kind")
            if kind == KIND_CHUNK:
                chunks.append(body)
            elif kind == KIND_DONE:
                return b"".join(chunks)
            elif kind == KIND_REJECT:
                raise ContentHydrationError(
                    f"holder refused the transfer: {message.get('reason')}"
                )
            elif kind != KIND_HEAD:
                raise ContentHydrationError("unexpected content transfer message")

    def deliver_grant(self, grant: ContentHydrationGrant) -> None:
        """Hand a control-minted grant to the request waiting for its object."""
        self._settle(
            (
                grant.reference.authorization_scope,
                grant.reference.content_digest,
            ),
            grant,
        )

    def deliver_denial(self, reference: ContentReference, reason: str) -> None:
        """Fail the request waiting on this object with control's typed reason."""
        self._settle((reference.authorization_scope, reference.content_digest), reason)

    def _settle(self, key: _WaitKey, answer: ContentHydrationGrant | str) -> None:
        """Hand one answer to the longest-waiting request for that object."""
        waiting = self._waiters.get(key)
        while waiting:
            waiter = waiting.popleft()
            if not waiter.done():
                waiter.set_result(answer)
                break
        if not waiting:
            self._waiters.pop(key, None)

    def _forget(self, key: _WaitKey, waiter: "_Answer") -> None:
        waiting = self._waiters.get(key)
        if waiting is None:
            return
        if waiter in waiting:
            waiting.remove(waiter)
        if not waiting:
            self._waiters.pop(key, None)

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound frame into the transfer waiting for it."""
        session = self._sessions.get(frame.session_id)
        if session is not None:
            await session.on_frame(frame)
