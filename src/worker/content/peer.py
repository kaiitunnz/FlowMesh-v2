"""Dialing a holder for a transfer control routed over a peer transport.

A requester reaches a holder's content listener directly, or the holder node's listener
that forwards to it, only at the address control resolved beside the transfer's grant.
The connection carries that one transfer in both directions and closes with it. Path
evidence goes back to control as it does for a resident origin: a dial or a mid-transfer
loss is a network failure, a busy or purpose-less target is no evidence at all, and what
the holder answers — a refusal, a copy that does not verify — says nothing of the path.
"""

import asyncio
import contextlib
import logging
import ssl
from collections.abc import Callable

from shared.content.carriage import ContentCarriagePlan
from shared.network.frame_stream import (
    FrameStreamError,
    PeerPurpose,
    read_relay_frame,
    write_relay_frame,
)
from shared.network.peer_dial import (
    PEER_DIAL_ERRORS,
    PeerAcceptError,
    classify_peer_error,
    open_accepted_connection,
)
from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession
from shared.schemas.network import RouteObservationOutcome

# Reports one classified observation of the path a transfer was carried over.
ObservePath = Callable[[ContentCarriagePlan, RouteObservationOutcome], None]


class PeerUnreachable(Exception):
    """The holder's side could not be reached; nothing of the transfer was sent."""


class PeerConnection:
    """One dialed connection carrying one transfer."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        on_loss: Callable[[], None],
        on_frame: Callable[[], None],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._on_loss = on_loss
        self._on_frame = on_frame
        self._lock = asyncio.Lock()
        self._pump: asyncio.Task[None] | None = None
        self._closing = False
        self._lost = False

    async def send(self, frame: RelayFrame) -> None:
        try:
            async with self._lock:
                await write_relay_frame(self._writer, frame)
        except (OSError, FrameStreamError):
            self._lose()
            raise

    def attach(self, session: FramedRelaySession) -> None:
        """Feed the session every frame the connection reads for it."""
        self._pump = asyncio.ensure_future(self._read_into(session))

    async def _read_into(self, session: FramedRelaySession) -> None:
        try:
            while True:
                frame = await read_relay_frame(self._reader)
                if frame.session_id == session.session_id:
                    self._on_frame()
                    await session.on_frame(frame)
        except (asyncio.IncompleteReadError, OSError, FrameStreamError):
            # The transfer is waiting on frames this connection will never carry, so it
            # ends now rather than at its stall deadline.
            if not self._closing:
                self._lose()
            session.abort()

    def _lose(self) -> None:
        if not self._lost and not self._closing:
            self._lost = True
            self._on_loss()

    async def aclose(self) -> None:
        self._closing = True
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
        self._writer.transport.abort()


class ContentPeerDialer:
    """Opens the connection a carriage plan names, reporting what the path did."""

    def __init__(
        self,
        *,
        ssl_context: ssl.SSLContext | None,
        connect_budget_sec: float,
        observe: ObservePath,
        logger: logging.Logger | None = None,
    ) -> None:
        self._ssl_context = ssl_context
        self._budget = connect_budget_sec
        self._observe = observe
        self._logger = logger or logging.getLogger("content-peer-dialer")

    async def open(
        self, carriage: ContentCarriagePlan, on_frame: Callable[[], None]
    ) -> PeerConnection:
        """Dial the plan's endpoint and wait for the target to accept a content
        connection, raising ``PeerUnreachable`` before anything of the transfer goes."""
        try:
            reader, writer = await open_accepted_connection(
                carriage.endpoint, self._ssl_context, self._budget, PeerPurpose.CONTENT
            )
        except PEER_DIAL_ERRORS as exc:
            if not (isinstance(exc, PeerAcceptError) and exc.outcome is None):
                self._observe(carriage, classify_peer_error(exc))
            self._logger.info(
                "content peer %s over %s unreachable: %s",
                carriage.endpoint,
                carriage.transport,
                exc,
            )
            raise PeerUnreachable(str(exc)) from exc
        self._observe(carriage, RouteObservationOutcome.VERIFIED)
        return PeerConnection(
            reader,
            writer,
            on_loss=lambda: self._observe(
                carriage, RouteObservationOutcome.ROUTE_FAILURE
            ),
            on_frame=on_frame,
        )


__all__ = ["ContentPeerDialer", "ObservePath", "PeerConnection", "PeerUnreachable"]
