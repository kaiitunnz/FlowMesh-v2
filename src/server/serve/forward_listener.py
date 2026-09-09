"""The root-hosted forward serve ingress: one gated HTTP listener per task port.

A forward serve task owns a public port on the root's public host. The root binds a
plain-HTTP listener on that port; a client reaches the task at
``http://<public_host>:<port>/<engine-native-path>`` — the port is the whole address, so
the listener resolves the serve task from the port it arrived on,
never from a client-supplied path, and forwards the engine-native path verbatim.

The listener authenticates nothing itself beyond reading the presented credential: it
freezes the client's transparent request envelope and hands it to the gated edge, which
authenticates the FlowMesh principal, checks task-read access, admits the request over
the same resident claim gate as ``proxy``, and streams the engine's opaque response
frames back. Those frames are written to the client socket here; the credential never
leaves this process for the engine. Binding a port is two-phase: control reserves it and
asks this listener to bind, and only the bound listener's evidence commits the exposure
live.

The deployment's front proxy is expected to terminate TLS and throttle abusive clients
ahead of the root, but the listener does not lean on it alone: it bounds its own request
read and caps concurrent connections so a stalled or flooding client cannot pin the
control plane the root runs on.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import TYPE_CHECKING

from shared.resident.envelope import (
    EnvelopeRejected,
    ServeRequestEnvelope,
    freeze_request_envelope,
)

if TYPE_CHECKING:
    from .service import ServeEvent, ServeResult

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024
_STREAM_IDLE_TIMEOUT_SEC = 300.0
# The request read (head and body) must complete within this bound: a client that opens
# a socket and stalls before finishing the request cannot pin a port and a task open.
_REQUEST_READ_TIMEOUT_SEC = 30.0
# The most requests served concurrently across every forward port. A flood past it is
# refused fast rather than allowed to exhaust the root's sockets and memory.
_MAX_CONCURRENT_CONNECTIONS = 512


class ServeForwardDenied(Exception):
    """A forward request refused before or during admission, with an HTTP status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Authenticate, authorize, and admit one forward request for a resolved task, returning
# its response stream or raising ``ServeForwardDenied``.
AdmitFn = Callable[[str | None, str, ServeRequestEnvelope], Awaitable["ServeResult"]]
# Report a reserved port bound and serving so control commits the exposure live.
BoundFn = Callable[[str, int, int], None]


class RootForwardIngress:
    """Binds one gated HTTP listener per reserved forward port on the root."""

    def __init__(
        self,
        *,
        bind_host: str,
        public_host: str,
        admit: AdmitFn,
        on_bound: BoundFn,
        stream_idle_timeout_sec: float = _STREAM_IDLE_TIMEOUT_SEC,
        request_read_timeout_sec: float = _REQUEST_READ_TIMEOUT_SEC,
        max_connections: int = _MAX_CONCURRENT_CONNECTIONS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._bind_host = bind_host
        self._public_host = public_host
        self._admit = admit
        self._on_bound = on_bound
        self._idle_timeout = stream_idle_timeout_sec
        self._read_timeout = request_read_timeout_sec
        self._max_connections = max_connections
        self._active = 0
        self._log = logger or logging.getLogger("serve-forward-ingress")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._servers: dict[int, asyncio.AbstractServer] = {}
        self._port_to_task: dict[int, str] = {}
        self._listener_generation = 0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the loop the listeners bind and serve on."""
        self._loop = loop

    async def stop(self) -> None:
        servers = list(self._servers.values())
        self._servers.clear()
        self._port_to_task.clear()
        await self._close(servers)

    def schedule_bind(
        self, serve_task_id: str, exposure_generation: int, port: int
    ) -> None:
        """Ask the listener loop to bind a port for a task (called off that loop)."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(
            lambda: loop.create_task(
                self._bind_and_report(serve_task_id, exposure_generation, port)
            )
        )

    def schedule_release(self, port: int) -> None:
        """Ask the listener loop to close a drained port (called off that loop)."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(lambda: loop.create_task(self._release(port)))

    async def _bind_and_report(
        self, serve_task_id: str, exposure_generation: int, port: int
    ) -> None:
        existing = self._servers.get(port)
        if existing is not None:
            await self._close([existing])
            self._servers.pop(port, None)
        try:
            server = await asyncio.start_server(
                self._make_handler(port), host=self._bind_host, port=port
            )
        except OSError as exc:
            # A port that cannot bind reports nothing: the exposure never commits live,
            # so the task fails closed rather than serving on an unbound port.
            self._log.error(
                "forward serve ingress could not bind port %d for %s: %s",
                port,
                serve_task_id,
                exc,
            )
            return
        self._listener_generation += 1
        listener_generation = self._listener_generation
        self._servers[port] = server
        self._port_to_task[port] = serve_task_id
        self._log.info(
            "forward serve ingress bound %s on %s:%d",
            serve_task_id,
            self._public_host,
            port,
        )
        self._on_bound(serve_task_id, exposure_generation, listener_generation)

    async def _release(self, port: int) -> None:
        server = self._servers.pop(port, None)
        self._port_to_task.pop(port, None)
        if server is not None:
            await self._close([server])

    def _make_handler(
        self, port: int
    ) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
        async def _handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await self._handle_client(port, reader, writer)

        return _handler

    async def _handle_client(
        self, port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._active >= self._max_connections:
            # A flood past the cap is refused before it is admitted or streamed, so it
            # cannot exhaust the root; the front proxy is expected to throttle ahead of
            # this, but the cap holds even when a client reaches the bind host directly.
            # Consume the request head (bounded) first so the refusal closes cleanly
            # rather than racing a reset that loses the response.
            try:
                await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), self._read_timeout
                )
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                ConnectionError,
                TimeoutError,
            ):
                pass
            await self._refuse(writer, 503, "forward serve ingress is at capacity")
            await _close_writer(writer)
            return
        self._active += 1
        try:
            await self._serve_client(port, reader, writer)
        finally:
            self._active -= 1
            await _close_writer(writer)

    async def _serve_client(
        self, port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            parsed = await self._read_request(reader)
            if parsed is None:
                return
            envelope, credential = parsed
            serve_task_id = self._port_to_task.get(port)
            if serve_task_id is None:
                await self._refuse(writer, 404, "serve task not found")
                return
            result = await self._admit(credential, serve_task_id, envelope)
            # From here the response is streamed; ``_write_response`` owns every failure
            # after a head reaches the wire, so no head-committed request is ever
            # followed by a status line the catch-all below would append.
            await self._write_response(writer, envelope.method, result)
        except EnvelopeRejected as exc:
            await self._refuse(writer, 400, str(exc))
        except ServeForwardDenied as exc:
            await self._refuse(writer, exc.status, exc.detail)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._log.exception("forward serve request failed on port %d", port)
            await self._refuse(writer, 502, "serve request error")

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> tuple[ServeRequestEnvelope, str | None] | None:
        try:
            head = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), self._read_timeout
            )
        except TimeoutError as exc:
            # A client that stalls before completing the header is timed out rather than
            # left to pin the port; without this bound the read would block forever.
            raise ServeForwardDenied(408, "request header read timed out") from exc
        except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ConnectionError,
        ):
            return None
        if len(head) > _MAX_HEADER_BYTES:
            raise EnvelopeRejected("request header too large")
        lines = head.split(b"\r\n")
        request_line = lines[0].decode("latin-1")
        parts = request_line.split(" ")
        if len(parts) != 3:
            raise EnvelopeRejected("malformed request line")
        method, target, _version = parts
        if not target.startswith("/"):
            # Only origin-form targets are accepted; an absolute-form URL or an
            # authority-form target (CONNECT) would let the request name a host.
            raise EnvelopeRejected("request target must be origin-form")
        header_items: list[tuple[str, str]] = []
        content_length = 0
        for raw in lines[1:]:
            if not raw:
                continue
            name, sep, value = raw.decode("latin-1").partition(":")
            if not sep:
                raise EnvelopeRejected("malformed header field")
            name = name.strip()
            value = value.strip()
            header_items.append((name, value))
            if name.lower() == "content-length" and value.isdigit():
                content_length = int(value)
        path, _, query = target.partition("?")
        # Freeze the head with an empty body first: this rejects ambiguous framing —
        # Expect: 100-continue, any Transfer-Encoding, a duplicate Content-Length —
        # before the body is read, so a client waiting on a ``100 Continue`` this
        # listener never speaks is refused rather than left blocking on ``readexactly``.
        # The validated envelope carries the real body once it is read.
        credential = _bearer(header_items)
        envelope = freeze_request_envelope(
            method=method,
            upstream_path=path.lstrip("/"),
            query=query,
            headers=header_items,
            body=b"",
        )
        if content_length > _MAX_REQUEST_BYTES:
            raise EnvelopeRejected("request body too large")
        if content_length:
            try:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), self._read_timeout
                )
            except TimeoutError as exc:
                raise ServeForwardDenied(408, "request body read timed out") from exc
            except (asyncio.IncompleteReadError, ConnectionError):
                return None
            envelope = envelope.model_copy(update={"body": body})
        return envelope, credential

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        result: "ServeResult",
    ) -> None:
        events = aiter(result.events())
        first = await _next_event(events, self._idle_timeout)
        if first is None or first.kind != "head":
            # The sidecar commits its head first; a stream that opens with no head, a
            # bare terminal, or an error terminal never produced a response, so fail
            # closed rather than synthesize a 200 over a status the client never saw.
            detail = (
                first.detail
                if first is not None and first.detail
                else "resident serve produced no response"
            )
            await self._refuse(writer, 502, detail)
            if first is not None and not first.terminal:
                result.close_client()
            return
        write_body = method.upper() != "HEAD" and _status_allows_body(first.status)
        self._send_head(writer, first.status, first.headers, framed=write_body)
        terminated = False
        try:
            while not terminated:
                event = await _next_event(events, self._idle_timeout)
                if event is None:
                    self._abort(writer)
                    return
                if event.kind == "chunk":
                    if write_body and event.payload:
                        self._write_chunk(writer, event.payload)
                        # Drain each chunk so a slow client backpressures the write
                        # instead of the root buffering the whole response in memory.
                        await self._drain(writer)
                elif event.terminal:
                    terminated = True
                    if event.kind == "error":
                        # A stream that dropped frames or failed after the head was sent
                        # is aborted, so the client sees a truncated body rather than a
                        # well-formed one silently missing bytes.
                        self._abort(writer)
                    else:
                        if write_body:
                            self._end_chunks(writer)
                        await self._drain(writer)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # The client is gone or stopped draining: abort and stop delivering. The
            # claim settles on the sidecar's own fenced terminal, never on this
            # connection ending.
            self._abort(writer)
        except Exception:
            # A head is already on the wire, so a failure here cannot become a status:
            # abort rather than let it surface as a second response line upstream.
            self._log.exception("forward serve response streaming failed")
            self._abort(writer)
        finally:
            if not terminated:
                result.close_client()

    async def _drain(self, writer: asyncio.StreamWriter) -> None:
        """Flush buffered bytes, bounded so a stalled client cannot block forever."""
        await asyncio.wait_for(writer.drain(), self._idle_timeout)

    def _send_head(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        headers: tuple[tuple[str, str], ...],
        *,
        framed: bool,
    ) -> None:
        lines = [f"HTTP/1.1 {status} {_reason(status)}"]
        for name, value in headers:
            if name.lower() not in ("content-length", "transfer-encoding"):
                lines.append(f"{name}: {value}")
        if framed:
            lines.append("Transfer-Encoding: chunked")
        lines.append("Connection: close")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))

    def _write_chunk(self, writer: asyncio.StreamWriter, payload: bytes) -> None:
        writer.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")

    def _end_chunks(self, writer: asyncio.StreamWriter) -> None:
        writer.write(b"0\r\n\r\n")

    def _abort(self, writer: asyncio.StreamWriter) -> None:
        # A chunked body that never receives its terminating zero-length chunk is an
        # incomplete message: closing without it signals the failure rather than
        # implying a clean completion for a response that lost bytes.
        try:
            writer.transport.abort()
        except Exception:
            pass

    async def _refuse(
        self, writer: asyncio.StreamWriter, status: int, detail: str
    ) -> None:
        raw = detail.encode("utf-8")
        head = (
            f"HTTP/1.1 {status} {_reason(status)}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(raw)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        try:
            writer.write(head + raw)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def _close(self, servers: list[asyncio.AbstractServer]) -> None:
        for server in servers:
            server.close()
        for server in servers:
            try:
                await server.wait_closed()
            except Exception:
                pass


def _bearer(headers: list[tuple[str, str]]) -> str | None:
    for name, value in headers:
        if name.lower() == "authorization":
            return value
    return None


async def _next_event(
    events: AsyncIterator["ServeEvent"], timeout: float
) -> "ServeEvent | None":
    try:
        return await asyncio.wait_for(anext(events, None), timeout)
    except (TimeoutError, StopAsyncIteration):
        return None


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass


def _status_allows_body(status: int) -> bool:
    """Whether a response with this status may carry a body at all.

    ``204`` and ``304`` are defined to have none, and a ``1xx`` is informational, so
    framing a body for them is a protocol violation a client mis-parses or hangs on.
    """
    return not (status in (204, 304) or 100 <= status < 200)


def _reason(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Status"
