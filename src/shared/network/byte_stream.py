"""A bidirectional byte stream carried over one framed relay session.

A relayed TCP connection rides one ``rly-*`` session: the origin opens it by naming an
endpoint the target published, then each side sends its bytes as ``data`` messages and
ends its direction with ``eof``. Every message is a data frame, ordered behind the bytes
before it, so a half-close never overtakes data still in flight; a cancel aborts the
whole stream. The servers relay the frames opaquely and never read these messages.
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

from .frame_stream import FrameSink
from .relay_frame import RelayFrame
from .session import FramedRelaySession, RelaySessionRole

OPEN = "open"
DATA = "data"
EOF = "eof"
REFUSED = "refused"

CHUNK_BYTES = 32 * 1024
WINDOW_BYTES = 256 * 1024

# A blocked receive wakes on a cancel, so this only bounds an idle wait between loops.
_RECV_POLL_SEC = 3600.0


class StreamClosed(Exception):
    """The stream ended abnormally: cancelled, refused, or a frame was lost."""


class ByteStreamChannel:
    """One end of a byte stream over a strictly sequenced relay session."""

    def __init__(
        self,
        session_id: str,
        role: RelaySessionRole,
        sink: FrameSink,
        *,
        window_bytes: int = WINDOW_BYTES,
    ) -> None:
        self._session = FramedRelaySession(
            session_id=session_id,
            role=role,
            sink=sink,
            window_bytes=window_bytes,
            strict_sequence=True,
        )
        self._aborted = False

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def closed(self) -> bool:
        return self._session.cancelled

    async def on_frame(self, frame: RelayFrame) -> None:
        await self._session.on_frame(frame)

    async def send_open(self, endpoint_id: str) -> None:
        await self._guard(self._session.send_wire(OPEN, endpoint_id=endpoint_id))

    async def refuse(self, reason: str) -> None:
        """Tell the origin the stream cannot be served, then abort it."""
        try:
            await self._guard(self._session.send_wire(REFUSED, reason=reason))
        finally:
            await self.abort()

    async def send(self, data: bytes) -> None:
        """Send bytes in window-bounded chunks, blocking while the window is full."""
        for start in range(0, len(data), CHUNK_BYTES):
            chunk = data[start : start + CHUNK_BYTES]
            await self._guard(self._session.send_body_wire(DATA, chunk))

    async def send_eof(self) -> None:
        await self._guard(self._session.send_wire(EOF))

    async def recv_message(self) -> tuple[dict[str, Any], bytes]:
        """Return the next message and its body; raise once the stream is aborted."""
        while True:
            try:
                message = await self._session.recv_body_wire(_RECV_POLL_SEC)
            except ValueError as exc:
                raise StreamClosed("malformed relay message") from exc
            if message is not None:
                return message
            if self._session.cancelled:
                raise StreamClosed("relay stream cancelled")

    async def recv(self) -> bytes | None:
        """Return the next bytes, or None once the peer ended its direction."""
        header, body = await self.recv_message()
        match header.get("kind"):
            case "data":
                return body
            case "eof":
                return None
            case "refused":
                raise StreamClosed(f"relay refused: {header.get('reason', '')}")
        raise StreamClosed(f"unexpected relay message {header.get('kind')!r}")

    async def abort(self) -> None:
        """Cancel the stream at both ends; idempotent."""
        if self._aborted:
            return
        self._aborted = True
        # A peer cancel needs no answer; a local gap or abort must reach the peer.
        if self._session.cancelled and not self._session.broken:
            return
        try:
            await self._session.cancel()
        except Exception:
            pass

    async def _guard(self, send: Awaitable[None]) -> None:
        """Run a send that may block on the window, unblocking it on a cancel."""
        sender = asyncio.ensure_future(send)
        waiter = asyncio.ensure_future(self._session.wait_cancelled())
        try:
            await asyncio.wait({sender, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            sender.cancel()
            raise
        finally:
            waiter.cancel()
        if sender.done():
            sender.result()
            return
        sender.cancel()
        raise StreamClosed("relay stream cancelled")


async def splice(
    channel: ByteStreamChannel,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> bool:
    """Relay a socket through ``channel`` until both directions end.

    Each direction half-closes on its own: a socket EOF sends ``eof``, and a received
    ``eof`` shuts the socket's write side. Returns True when both directions ended
    cleanly; anything else aborts the stream at both ends and returns False.
    """

    async def outbound() -> None:
        while data := await reader.read(CHUNK_BYTES):
            await channel.send(data)
        await channel.send_eof()

    async def inbound() -> None:
        while (data := await channel.recv()) is not None:
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    tasks = {asyncio.ensure_future(outbound()), asyncio.ensure_future(inbound())}
    pending = set(tasks)
    clean = False
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        clean = True
    except (StreamClosed, OSError):
        pass
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if not clean:
            await channel.abort()
    return clean


__all__ = [
    "CHUNK_BYTES",
    "DATA",
    "EOF",
    "OPEN",
    "REFUSED",
    "WINDOW_BYTES",
    "ByteStreamChannel",
    "StreamClosed",
    "splice",
]
