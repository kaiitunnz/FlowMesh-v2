"""The origin-side resident invocation deputy.

It executes a resolved candidate ladder to deliver a claim-bound bootstrap to a resident
sidecar and then, under the post-``ACCEPTED`` route authorization, carries the response
stream, cancellation, and backpressure over the same data-direct connection. It runs the
two phases the control plane pokes it with — bootstrap, then stream-under-auth — and
holds the sidecar connection between them, keyed per invocation. It executes only the
candidates control resolved and never scans for a peer.

Fallback is not retry: a connect failure proved before the bootstrap was sent takes the
next resolved candidate; once the bootstrap is on the wire, a lost acknowledgement
or any ambiguous delivery is uncertain, never a blind resend to another path.
"""

import asyncio
import contextlib
import logging
import socket
import ssl
from dataclasses import dataclass, field

from ..network import wire as netwire
from ..network.state import (
    ResolvedRoute,
    RouteCandidate,
    RouteObservationOutcome,
    Transport,
)
from . import wire
from .state import AdmissionHandoff, RouteAuthorization


@dataclass
class BootstrapResult:
    """The bootstrap phase outcome, with per-candidate classified route observations."""

    acked: bool
    selected_transport: Transport | None
    rejection: str | None
    uncertain: bool
    observations: list[tuple[Transport, RouteObservationOutcome]] = field(
        default_factory=list
    )


@dataclass
class StreamResult:
    """The stream/cancel phase outcome.

    A non-ok post-acceptance result is uncertain unless ``definite``: the sidecar sets
    that only when the engine refused with a definite status, so no slot is held and the
    caller may release rather than hold the credit.
    """

    ok: bool
    completion: str | None = None
    rejection: str | None = None
    cancelled: bool = False
    definite: bool = False


@dataclass
class _Session:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    transport: Transport
    reaper: asyncio.TimerHandle | None = None
    stream_task: "asyncio.Task[StreamResult] | None" = None


class ResidentInvocationDeputy:
    """Carries a resident invocation from bootstrap to streamed response."""

    def __init__(
        self,
        *,
        connect_budget_sec: float,
        session_ttl_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._budget = connect_budget_sec
        self._session_ttl = session_ttl_sec
        self._logger = logger
        self._sessions: dict[str, _Session] = {}
        self._reaper_tasks: set[asyncio.Task[None]] = set()

    async def bootstrap(
        self,
        session_id: str,
        route: ResolvedRoute,
        handoff: AdmissionHandoff,
        request_payload: str | None,
    ) -> BootstrapResult:
        """Deliver the bootstrap over the candidate ladder and await the enqueue ack.

        A pre-send connect/route failure falls to the next candidate; a failure once
        the bootstrap is sent is an ambiguous delivery, so it stops as uncertain. A
        validated ack holds the connection for the stream phase, bounded by a reaper so
        a session whose stream never arrives cannot strand a connection.
        """
        observations: list[tuple[Transport, RouteObservationOutcome]] = []
        for candidate in route.candidates:
            try:
                reader, writer = await asyncio.wait_for(
                    self._connect(candidate), self._budget
                )
            except (TimeoutError, OSError, ValueError) as exc:
                observations.append((candidate.transport, _classify(exc)))
                continue
            try:
                await wire.write_msg(
                    writer,
                    wire.KIND_BOOTSTRAP,
                    handoff=handoff.model_dump(mode="json"),
                    request=request_payload,
                )
                reply = await asyncio.wait_for(wire.read_msg(reader), self._budget)
            except (TimeoutError, OSError, ValueError, asyncio.IncompleteReadError):
                # The bootstrap is already on the wire: an ambiguous delivery, not a
                # pre-send failure, so it is uncertain and never resent to another path.
                await _close(writer)
                observations.append(
                    (candidate.transport, RouteObservationOutcome.TIMEOUT)
                )
                return BootstrapResult(
                    False, candidate.transport, None, True, observations
                )
            if reply["kind"] == wire.KIND_ACK:
                observations.append(
                    (candidate.transport, RouteObservationOutcome.VERIFIED)
                )
                await self.reap(session_id)
                session = _Session(reader, writer, candidate.transport)
                session.reaper = asyncio.get_running_loop().call_later(
                    self._session_ttl, self._on_reaper, session_id
                )
                self._sessions[session_id] = session
                return BootstrapResult(
                    True, candidate.transport, None, False, observations
                )
            await _close(writer)
            if reply["kind"] == wire.KIND_REJECT:
                # A fence rejection is an authorization failure, not a path failure,
                # so it does not demote and does not fall through to another path to
                # the same replica: the fence would reject there too.
                observations.append(
                    (candidate.transport, RouteObservationOutcome.FENCE_INVALID)
                )
                return BootstrapResult(
                    False, candidate.transport, reply.get("reason"), False, observations
                )
            observations.append(
                (candidate.transport, RouteObservationOutcome.ROUTE_FAILURE)
            )
        return BootstrapResult(False, None, None, False, observations)

    async def stream(self, session_id: str, auth: RouteAuthorization) -> StreamResult:
        """Present the authorization and assemble the streamed response completion."""
        session = self._sessions.get(session_id)
        if session is None:
            return StreamResult(False, rejection="no_session")
        self._stop_reaper(session)
        current = asyncio.current_task()
        if isinstance(current, asyncio.Task):
            session.stream_task = current
        try:
            await wire.write_msg(
                session.writer, wire.KIND_STREAM, auth=auth.model_dump(mode="json")
            )
            parts: list[str] = []
            while True:
                msg = await wire.read_msg(session.reader)
                if msg["kind"] == wire.KIND_CHUNK:
                    parts.append(str(msg.get("data", "")))
                elif msg["kind"] == wire.KIND_DONE:
                    return StreamResult(True, completion="".join(parts))
                elif msg["kind"] == wire.KIND_FAILED:
                    return StreamResult(
                        False,
                        rejection=msg.get("reason"),
                        definite=bool(msg.get("definite")),
                    )
                elif msg["kind"] == wire.KIND_REJECT:
                    return StreamResult(False, rejection=msg.get("reason"))
                else:
                    return StreamResult(False, rejection="protocol")
        except (OSError, ValueError, asyncio.IncompleteReadError):
            return StreamResult(False, rejection="stream_loss")
        finally:
            await self.reap(session_id)

    async def cancel(self, session_id: str) -> StreamResult:
        """Abort a held invocation and reap both ends of its data-direct connection.

        The origin control plane reaps only the session it opened, so there is no fence
        on the seam: it cancels any in-flight stream and closes the connection, which
        the sidecar observes to abort its co-located engine request.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return StreamResult(False, rejection="no_session")
        task = session.stream_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.reap(session_id)
        return StreamResult(True, cancelled=True)

    async def reap(self, session_id: str) -> None:
        """Close and forget a held session and cancel its reaper; safe to call twice."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._stop_reaper(session)
            await _close(session.writer)

    async def aclose(self) -> None:
        """Reap every held session and cancel any pending reaper; for shutdown."""
        for task in list(self._reaper_tasks):
            task.cancel()
        for session_id in list(self._sessions):
            await self.reap(session_id)

    @staticmethod
    def _stop_reaper(session: _Session) -> None:
        if session.reaper is not None:
            session.reaper.cancel()
            session.reaper = None

    def _on_reaper(self, session_id: str) -> None:
        if self._logger is not None:
            self._logger.warning("reaping stranded resident session %s", session_id)
        task = asyncio.ensure_future(self.reap(session_id))
        self._reaper_tasks.add(task)
        task.add_done_callback(self._reaper_tasks.discard)

    async def _connect(
        self, candidate: RouteCandidate
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        host, port = wire.split_host_port(candidate.hops[0].endpoint)
        reader, writer = await asyncio.open_connection(host, port)
        # Each relay hop reads one leading target frame and byte-relays the rest, so
        # writing a frame per hop after the first chains to the terminal sidecar.
        for hop in candidate.hops[1:]:
            await netwire.write_frame(writer, hop.endpoint.encode())
        return reader, writer


def _classify(exc: BaseException) -> RouteObservationOutcome:
    if isinstance(exc, TimeoutError):
        return RouteObservationOutcome.TIMEOUT
    if isinstance(exc, ConnectionRefusedError):
        return RouteObservationOutcome.CONNECT_FAILURE
    if isinstance(exc, ssl.SSLError):
        return RouteObservationOutcome.TLS_FAILURE
    if isinstance(exc, socket.gaierror):
        return RouteObservationOutcome.DNS_FAILURE
    return RouteObservationOutcome.ROUTE_FAILURE


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass
