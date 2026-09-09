"""The worker's forward serve ingress: one gated HTTP(S) listener per task port.

A deployment registers this worker as a forward ingress host — a public authority and a
port range. Control reserves a public port for each forward serve task and tells this
host to bind it; the host binds one listener on that port, mapped to the task's port
exposure, and reports it bound so control can commit the exposure live. A client then
reaches the task at ``https://<authority>:<port>/<engine-native-path>`` — the port is
the whole address, so the listener resolves the serve task from its own exposure, never
from a client-supplied path, and forwards the engine-native path verbatim.

The listener authenticates nothing: it freezes the client's transparent request
envelope, hands the presented credential and the frozen request's descriptor to control
over the worker's authenticated attachment, and serves the request only on control's
admission. The credential never leaves that control message. TLS terminates here from
the operator's profile; a plaintext listener is an explicit local-test mode only.
"""

import logging
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from shared.resident.envelope import (
    EnvelopeRejected,
    ServeRequestEnvelope,
    freeze_request_envelope,
)
from shared.resident.serve_ingress import (
    ServeIngressBound,
    ServeIngressRequest,
    ServeIngressReserve,
)

from .channel import ServeIngressChannel
from .rendezvous import (
    ServeIngressAdmission,
    ServeIngressDenied,
    ServeIngressRendezvous,
)

_MAX_REQUEST_BYTES = 4 * 1024 * 1024


# Sends one admission request up over the worker's authenticated attachment.
ProposeFn = Callable[[ServeIngressRequest], None]
# Starts one admitted request's origin drive on the resident lane loop.
BeginFn = Callable[
    [ServeIngressAdmission, ServeRequestEnvelope, ServeIngressChannel], None
]
# Reports a reserved port bound and serving so control commits the exposure live.
BoundFn = Callable[[ServeIngressBound], None]


@dataclass(frozen=True)
class _Exposure:
    """The port exposure a listener resolves its requests to, assigned by control."""

    serve_task_id: str
    binding_generation: int
    exposure_generation: int


class ForwardIngressHost:
    """Binds one gated HTTP(S) listener per reserved forward port on this worker."""

    def __init__(
        self,
        *,
        bind_host: str,
        authority: str,
        propose: ProposeFn,
        begin: BeginFn,
        new_request_id: Callable[[], str],
        report_bound: BoundFn,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        admission_timeout_sec: float = 30.0,
        stream_idle_timeout_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._bind_host = bind_host
        self._authority = authority
        self._propose = propose
        self._begin = begin
        self._new_request_id = new_request_id
        self._report_bound = report_bound
        self._tls_cert = tls_cert
        self._tls_key = tls_key
        self._admission_timeout = admission_timeout_sec
        self._stream_idle_timeout = stream_idle_timeout_sec
        self._log = logger or logging.getLogger("serve-forward-ingress")
        self.rendezvous = ServeIngressRendezvous()
        self._servers: dict[int, _IngressHTTPServer] = {}
        self._listener_generation = 0
        self._lock = threading.Lock()

    @property
    def stream_idle_timeout(self) -> float:
        return self._stream_idle_timeout

    def reserve(self, reserve: ServeIngressReserve) -> None:
        """Bind a listener on the reserved port and report it bound, or fail closed.

        A TLS exposure without a configured operator profile, or a port that cannot
        bind, reports nothing — the exposure never commits live, so the task fails
        closed rather than serving on an unintended transport.
        """
        with self._lock:
            existing = self._servers.get(reserve.public_port)
            if existing is not None:
                if (
                    existing.exposure.serve_task_id == reserve.serve_task_id
                    and existing.exposure.exposure_generation
                    == reserve.exposure_generation
                ):
                    # A duplicate reserve for the same round: re-report the live
                    # binding.
                    self._report_bound(self._bound(reserve, existing.listener_gen))
                    return
                self._shutdown(reserve.public_port)
            if reserve.tls and not (self._tls_cert and self._tls_key):
                self._log.error(
                    "forward ingress cannot serve %s over TLS without a cert profile",
                    reserve.serve_task_id,
                )
                return
            exposure = _Exposure(
                serve_task_id=reserve.serve_task_id,
                binding_generation=reserve.binding_generation,
                exposure_generation=reserve.exposure_generation,
            )
            server = self._bind(reserve.public_port, exposure, reserve.tls)
            if server is None:
                return
            self._listener_generation += 1
            server.listener_gen = self._listener_generation
            self._servers[reserve.public_port] = server
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._log.info(
                "forward serve ingress bound %s on %s:%d",
                reserve.serve_task_id,
                self._authority,
                reserve.public_port,
            )
            listener_gen = server.listener_gen
        self._report_bound(self._bound(reserve, listener_gen))

    def release(self, public_port: int) -> None:
        """Close a drained exposure's listener and dispose its connections."""
        with self._lock:
            self._shutdown(public_port)

    def stop(self) -> None:
        with self._lock:
            for port in list(self._servers):
                self._shutdown(port)

    def _bind(
        self, port: int, exposure: _Exposure, tls: bool
    ) -> "_IngressHTTPServer | None":
        try:
            server = _IngressHTTPServer(
                (self._bind_host, port), _IngressHandler, self, exposure
            )
        except OSError as exc:
            self._log.error(
                "forward ingress could not bind port %d for %s: %s",
                port,
                exposure.serve_task_id,
                exc,
            )
            return None
        if tls and self._tls_cert and self._tls_key:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self._tls_cert, self._tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        return server

    def _shutdown(self, port: int) -> None:
        server = self._servers.pop(port, None)
        if server is not None:
            server.shutdown()
            server.server_close()

    @staticmethod
    def _bound(
        reserve: ServeIngressReserve, listener_generation: int
    ) -> ServeIngressBound:
        return ServeIngressBound(
            serve_task_id=reserve.serve_task_id,
            binding_generation=reserve.binding_generation,
            exposure_generation=reserve.exposure_generation,
            listener_generation=listener_generation,
            attachment_generation=1,
        )

    def handle(
        self,
        exposure: _Exposure,
        method: str,
        target: str,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> ServeIngressChannel | ServeIngressDenied:
        """Admit one request for this port's exposure through control, or refuse it.

        The request target is the engine-native origin-form path, forwarded verbatim;
        the serve task is this listener's own exposure, never a client-supplied path.
        """
        split = urlsplit(target)
        credential = _bearer(headers)
        try:
            envelope = freeze_request_envelope(
                method=method,
                upstream_path=split.path.lstrip("/"),
                query=split.query,
                headers=headers,
                body=body,
            )
        except EnvelopeRejected as exc:
            return ServeIngressDenied(400, str(exc))

        request_id = self._new_request_id()
        with self.rendezvous.register(request_id) as waiter:
            self._propose(
                ServeIngressRequest(
                    request_id=request_id,
                    serve_task_id=exposure.serve_task_id,
                    binding_generation=exposure.binding_generation,
                    exposure_generation=exposure.exposure_generation,
                    credential=credential,
                    method=envelope.method,
                    path=envelope.path,
                    query=envelope.query,
                    descriptor_digest=envelope.digest(),
                    body_bytes=len(envelope.body),
                )
            )
            decision = waiter.await_decision(self._admission_timeout)
        if decision is None:
            return ServeIngressDenied(504, "admission timed out")
        if isinstance(decision, ServeIngressDenied):
            return decision

        channel = ServeIngressChannel()
        self._begin(decision, envelope, channel)
        return channel


def _bearer(headers: list[tuple[str, str]]) -> str | None:
    for name, value in headers:
        if name.lower() == "authorization":
            return value
    return None


class _IngressHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        host: ForwardIngressHost,
        exposure: _Exposure,
    ) -> None:
        self.host = host
        self.exposure = exposure
        self.listener_gen = 0
        super().__init__(address, handler)


class _IngressHandler(BaseHTTPRequestHandler):
    server: _IngressHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        return None

    def _serve(self) -> None:
        host = self.server.host
        declared = self.headers.get("content-length")
        if declared is not None and declared.isdigit():
            if int(declared) > _MAX_REQUEST_BYTES:
                self._refuse(413, "request body too large")
                return
            body = self.rfile.read(int(declared))
        else:
            body = b""
        outcome = host.handle(
            self.server.exposure,
            self.command,
            self.path,
            list(self.headers.items()),
            body,
        )
        if isinstance(outcome, ServeIngressDenied):
            self._refuse(outcome.status, outcome.detail)
            return
        self._relay(outcome, host.stream_idle_timeout)

    do_GET = _serve
    do_POST = _serve
    do_PUT = _serve
    do_DELETE = _serve
    do_OPTIONS = _serve
    do_HEAD = _serve

    def _refuse(self, status: int, detail: str) -> None:
        raw = detail.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _relay(self, channel: ServeIngressChannel, idle_timeout: float) -> None:
        """Write the engine's own head, then its body frames, as they arrive."""
        started = False
        write_body = self.command != "HEAD"
        try:
            while True:
                frame = channel.drain(idle_timeout)
                if frame is None:
                    if started:
                        self._abort()
                    else:
                        self._refuse(504, "resident serve timed out")
                    return
                if frame.kind == "head":
                    framed = _status_allows_body(frame.status)
                    write_body = write_body and framed
                    self._send_head(frame.status, frame.headers, framed=framed)
                    started = True
                    # The client now holds the head: mark it committed only now, so a
                    # loss before this point may legally re-drive while a loss after it
                    # closes the client rather than re-driving over bytes it already
                    # has.
                    channel.commit_head(frame.status, frame.headers)
                elif frame.kind == "chunk":
                    if not started:
                        self._refuse(502, "resident serve sent a body before a head")
                        return
                    if write_body and frame.payload:
                        self._write_chunk(frame.payload)
                elif frame.terminal:
                    if not started:
                        self._refuse(
                            502, frame.detail or "resident serve produced no response"
                        )
                        return
                    if channel.lost or frame.kind == "error":
                        self._abort()
                    elif write_body:
                        self._end_chunks()
                    return
        except (BrokenPipeError, ConnectionResetError):
            # The client is gone. Stop delivering; the claim settles on the sidecar's
            # own fenced terminal, never on this connection ending.
            channel.close()

    def _abort(self) -> None:
        """Drop a partially delivered response so the client sees a broken transfer.

        A chunked body that never receives its terminating zero-length chunk is an
        incomplete message: closing the connection without it signals the failure rather
        than implying a clean completion for a response that lost bytes.
        """
        self.close_connection = True

    def _send_head(
        self, status: int, headers: tuple[tuple[str, str], ...], *, framed: bool
    ) -> None:
        self.send_response(status)
        for name, value in headers:
            if name.lower() not in ("content-length", "transfer-encoding"):
                self.send_header(name, value)
        if framed:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def _status_allows_body(status: int) -> bool:
    """Whether a response with this status may carry a body at all.

    ``204`` and ``304`` are defined to have none, and a ``1xx`` is informational, so
    framing a body for them is a protocol violation a client mis-parses or hangs on.
    """
    return not (status in (204, 304) or 100 <= status < 200)
