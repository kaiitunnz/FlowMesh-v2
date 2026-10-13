"""Hydrating an object this worker does not hold.

The consumer side of a transfer: the worker asks the control plane for a grant on one
exact reference, and control answers with a grant naming the holder it resolved or a
typed denial. With a grant in hand the worker opens the transfer and streams the object
from that holder into a file, hashing as it goes, and hands the file back only once its
bytes verify against the reference; a transfer that fails leaves no file behind.

Control routes the transfer beside its grant: over the root relay, or over a connection
this worker dials to the holder or its node. A failed dial sends nothing, so its grant
carries over to the next route; a dialed transfer that breaks after its fetch may have
spent its grant, so it is retried once under a fresh one. A caller that prefers the
store to the relay asks for a peer transport only.
Nothing here decides that the worker may read the object — control does, against the
consumer binding that entitles it — and nothing here recreates an outcome, re-resolves
an input, or releases a credit when a transfer fails: an unavailable holder is a typed
hydration failure and the consumer's own recovery decides what follows.
"""

import asyncio
import contextlib
import itertools
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from shared.content import (
    ContentCarriagePlan,
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
    ContentVerifier,
)
from shared.content.config import DEFAULT_TRANSFER_WINDOW_BYTES
from shared.content.wire import (
    KIND_CHUNK,
    KIND_DONE,
    KIND_FETCH,
    KIND_HEAD,
    KIND_REJECT,
    REJECT_WRONG_REQUESTER,
)
from shared.network.frame_stream import FrameSink
from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession, RelaySessionRole
from shared.utils.atomic import temp_path

from .peer import ContentPeerDialer, PeerConnection, PeerUnreachable

# Asks the control plane to authorize hydrating one reference for one task; with the
# flag, over a peer transport alone. Control answers with the request's number.
RequestGrant = Callable[[ContentReference, str, bool, int], None]

# Reports the objects this worker holds, each by its scope and digest, to the holder
# directory. One call carries a whole report so a periodic re-report is one message.
AnnounceHolding = Callable[[Sequence[tuple[str, str]]], None]

# How much of a transfer collects in memory before it is written out, so the file is
# written in large blocks off the loop rather than frame by frame on it.
_WRITE_BUFFER_BYTES = 1 << 20

_Granted = tuple[ContentHydrationGrant, ContentCarriagePlan | None]
_Answer = asyncio.Future[_Granted | str]

# A transfer that broke after its fetch is tried again under one fresh grant at most.
_TRANSFER_ATTEMPTS = 2


class GrantDenied(ContentHydrationError):
    """The control plane refused to authorize this hydration."""


class RequesterRefused(ContentHydrationError):
    """The holder refused this worker's node as not the transfer's requester."""


class ContentHydrationClient:
    """This worker's end of the transfers it requests."""

    def __init__(
        self,
        *,
        sink: FrameSink,
        request_grant: RequestGrant,
        transfer_timeout_sec: float = 60.0,
        window_bytes: int = DEFAULT_TRANSFER_WINDOW_BYTES,
        dialer: ContentPeerDialer | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sink = sink
        self._request_grant = request_grant
        self._dialer = dialer
        # Every wait here is bounded by how long a transfer may go without progress,
        # never by how long it takes: a large object over a slow link runs as long as
        # its frames keep arriving.
        self._timeout = transfer_timeout_sec
        self._last_progress = time.monotonic()
        self._window_bytes = window_bytes
        self._logger = logger or logging.getLogger("content-hydration")
        # Each request waits under its own number, which control echoes beside its
        # answer, so two requests for one object never take each other's.
        self._waiters: dict[int, _Answer] = {}
        self._requests = itertools.count(1)
        self._sessions: dict[str, FramedRelaySession] = {}

    @property
    def last_progress(self) -> float:
        """When a transfer here last moved, on the monotonic clock."""
        return self._last_progress

    def _progressed(self) -> None:
        self._last_progress = time.monotonic()

    async def hydrate_to_file(
        self,
        reference: ContentReference,
        task_id: str,
        staging: Path,
        peer_only: bool = False,
    ) -> Path:
        """Fetch one object from whichever holder control resolves into a new file in
        ``staging``, and return the file once its bytes verify.

        With ``peer_only``, a read no peer transport carries fails.
        """
        self._progressed()
        attempts_left = _TRANSFER_ATTEMPTS
        while True:
            attempts_left -= 1
            grant, carriage = await self._grant_for(reference, task_id, peer_only)
            connection = await self._dial(carriage)
            if connection is None and peer_only:
                # Nothing was sent, so the grant is unspent and lapses at the holder.
                raise ContentHydrationError(
                    f"no peer transport reaches a holder of {reference.content_digest}"
                )
            try:
                return await self._transfer_into(grant, staging, connection)
            except ContentHydrationError:
                if connection is None or not attempts_left:
                    raise
                self._logger.info(
                    "content transfer %s over %s failed; asking for a fresh grant",
                    grant.transfer_session_id,
                    connection.route.transport,
                )

    async def _dial(
        self, carriage: ContentCarriagePlan | None
    ) -> PeerConnection | None:
        """Return a connection over the first of the plan's routes that connects, or
        None for the root relay.

        Nothing of the transfer is sent before a route connects, so every route a plan
        names is tried under the one grant.
        """
        if carriage is None or not carriage.fresh() or self._dialer is None:
            return None
        for route in carriage.routes:
            try:
                return await self._dialer.open(carriage, route, self._progressed)
            except PeerUnreachable:
                continue
        return None

    async def _transfer_into(
        self,
        grant: ContentHydrationGrant,
        staging: Path,
        connection: PeerConnection | None,
    ) -> Path:
        session = FramedRelaySession(
            session_id=grant.transfer_session_id,
            correlation_id=grant.grant_id,
            role=RelaySessionRole.ORIGIN,
            sink=connection or self._sink,
            window_bytes=self._window_bytes,
        )
        if connection is None:
            self._sessions[grant.transfer_session_id] = session
        else:
            connection.attach(session)
        target = temp_path(staging)
        served = False
        try:
            await self._transfer(session, grant, target)
            served = True
            self._logger.info(
                "content %s carried over %s",
                grant.reference.content_digest,
                connection.route.transport if connection else "the root relay",
            )
            return target
        except RequesterRefused:
            if connection is not None:
                connection.refused_requester()
                self._logger.warning(
                    "the holder refused this worker's node as not the requester of %s "
                    "over %s: the node's peer certificate does not cover the host it "
                    "advertises",
                    grant.transfer_session_id,
                    connection.route.transport,
                )
            raise
        except OSError as exc:
            raise ContentHydrationError(
                f"content transfer {grant.transfer_session_id} lost its carriage"
            ) from exc
        finally:
            if not served:
                target.unlink(missing_ok=True)
                # A transfer only advances as this end drains and grants window, so
                # abandoning one silently would leave the holder blocked on a send
                # nobody is reading — holding the object against eviction for as long
                # as it lives. Telling it to reap costs one frame.
                with contextlib.suppress(Exception):
                    await session.cancel()
            self._sessions.pop(grant.transfer_session_id, None)
            if connection is not None:
                await connection.aclose()

    async def _grant_for(
        self, reference: ContentReference, task_id: str, peer_only: bool
    ) -> _Granted:
        """Ask control to authorize this read, and wait for its answer.

        Each request waits on its own answer because a grant is good for one transfer:
        two tasks reading the same object each ask, and each takes the grant minted for
        it rather than sharing one that the second would find already consumed.
        """
        request_id = next(self._requests)
        pending: _Answer = asyncio.get_running_loop().create_future()
        # Armed before the request goes up, so an answer that comes straight back finds
        # a waiter rather than arriving at nobody.
        self._waiters[request_id] = pending
        self._request_grant(reference, task_id, peer_only, request_id)
        try:
            answer = await asyncio.wait_for(pending, self._timeout)
        except TimeoutError as exc:
            raise ContentHydrationError(
                f"no hydration grant for {reference.content_digest} in time"
            ) from exc
        finally:
            self._waiters.pop(request_id, None)
        if isinstance(answer, str):
            raise GrantDenied(
                f"hydration of {reference.content_digest} denied: {answer}"
            )
        return answer

    async def _transfer(
        self, session: FramedRelaySession, grant: ContentHydrationGrant, target: Path
    ) -> None:
        await session.send_wire(KIND_FETCH, grant=grant.model_dump(mode="json"))
        verifier = ContentVerifier(grant.reference)
        pending: list[bytes] = []
        buffered = 0
        with target.open("xb") as fh:
            while True:
                frame = await session.recv_body_wire(timeout=self._timeout)
                if frame is None:
                    raise ContentHydrationError(
                        f"content transfer {session.session_id} ended without the "
                        "object"
                    )
                message, body = frame
                kind = message.get("kind")
                if kind == KIND_CHUNK:
                    verifier.update(body)
                    pending.append(body)
                    buffered += len(body)
                    if buffered >= _WRITE_BUFFER_BYTES:
                        await asyncio.to_thread(fh.writelines, pending)
                        pending, buffered = [], 0
                elif kind == KIND_DONE:
                    await asyncio.to_thread(fh.writelines, pending)
                    verifier.finish()
                    return
                elif kind == KIND_REJECT:
                    reason = message.get("reason")
                    refused = (
                        RequesterRefused
                        if reason == REJECT_WRONG_REQUESTER
                        else ContentHydrationError
                    )
                    raise refused(f"holder refused the transfer: {reason}")
                elif kind != KIND_HEAD:
                    raise ContentHydrationError("unexpected content transfer message")

    def deliver_grant(
        self,
        request_id: int,
        grant: ContentHydrationGrant,
        carriage: ContentCarriagePlan | None = None,
    ) -> None:
        """Hand a control-minted grant, and how control routed it, to its request."""
        self._progressed()
        self._settle(request_id, (grant, carriage))

    def deliver_denial(self, request_id: int, reason: str) -> None:
        """Fail a request with control's typed reason."""
        self._progressed()
        self._settle(request_id, reason)

    def _settle(self, request_id: int, answer: _Granted | str) -> None:
        if (waiter := self._waiters.get(request_id)) is not None and not waiter.done():
            waiter.set_result(answer)

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one relayed frame into the transfer waiting for it.

        A transfer carried over a dialed connection reads only that connection, so a
        relayed frame naming its session finds nothing here.
        """
        session = self._sessions.get(frame.session_id)
        if session is not None:
            self._progressed()
            await session.on_frame(frame)
