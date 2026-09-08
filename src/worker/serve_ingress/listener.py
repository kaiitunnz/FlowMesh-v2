"""The forward ingress's public HTTP listener.

An external client reaches a public serve task here by its task ID, at the same
task-qualified route shape the root-local proxy serves. The listener authenticates
nothing: it freezes the client's transparent request envelope, hands the presented
credential and the frozen request's descriptor to control over the worker's
authenticated attachment, and serves the request only on control's admission — which
performs authentication, task-read authorization, binding lookup, and admission.

The credential never leaves that control message: it is not logged, not relayed on the
data path, and not forwarded to the engine, which the replica's sidecar reaches with its
own. The listener relays opaque frames and applies no engine semantics of its own.
"""

import logging
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

from .channel import ServeIngressChannel
from .rendezvous import (
    ServeIngressAdmission,
    ServeIngressDenied,
    ServeIngressRendezvous,
)

# The task-qualified route shape, matching the root-local proxy's.
_ROUTE_PREFIX = "/api/v1/serve/tasks/"

_MAX_REQUEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ServeIngressRequest:
    """What the ingress asks control to authenticate, authorize, and admit.

    It carries the frozen request's descriptor, never its body: the raw request stays
    worker-private behind the digest, so control admits the request without the payload
    ever entering control state.
    """

    request_id: str
    serve_task_id: str
    credential: str | None
    method: str
    path: str
    query: str
    descriptor_digest: str
    body_bytes: int


# Sends one admission request up over the worker's authenticated attachment.
ProposeFn = Callable[[ServeIngressRequest], None]
# Starts one admitted request's origin drive on the resident lane loop.
BeginFn = Callable[
    [ServeIngressAdmission, ServeRequestEnvelope, ServeIngressChannel], None
]


class ServeForwardIngress:
    """Serves external task-addressed requests from a worker, gated by control."""

    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        public_url: str,
        propose: ProposeFn,
        begin: BeginFn,
        new_request_id: Callable[[], str],
        admission_timeout_sec: float = 30.0,
        stream_idle_timeout_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._bind_host = bind_host
        self._port = port
        self._public_url = public_url.rstrip("/")
        self._propose = propose
        self._begin = begin
        self._new_request_id = new_request_id
        self._admission_timeout = admission_timeout_sec
        self._stream_idle_timeout = stream_idle_timeout_sec
        self._log = logger or logging.getLogger("serve-forward-ingress")
        self.rendezvous = ServeIngressRendezvous()
        self._server: _IngressHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def public_url(self) -> str:
        """The base url clients reach this ingress at, as registered with control."""
        return self._public_url

    def start(self) -> int:
        """Bind the public listener and begin serving; returns the bound port."""
        server = _IngressHTTPServer(
            (self._bind_host, self._port), _IngressHandler, self
        )
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        self._log.info(
            "gated forward serve ingress listening on %s:%d as %s",
            self._bind_host,
            server.server_address[1],
            self._public_url,
        )
        return int(server.server_address[1])

    def stop(self) -> None:
        if (server := self._server) is not None:
            server.shutdown()
            server.server_close()
        if (thread := self._thread) is not None:
            thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def handle(
        self,
        method: str,
        target: str,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> ServeIngressChannel | ServeIngressDenied:
        """Admit one request through control and begin relaying it, or refuse it."""
        split = urlsplit(target)
        if not split.path.startswith(_ROUTE_PREFIX):
            return ServeIngressDenied(404, "not found")
        remainder = split.path[len(_ROUTE_PREFIX) :]
        serve_task_id, _, upstream_path = remainder.partition("/")
        if not serve_task_id or not upstream_path:
            return ServeIngressDenied(404, "not found")

        credential = _bearer(headers)
        try:
            envelope = freeze_request_envelope(
                method=method,
                upstream_path=upstream_path,
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
                    serve_task_id=serve_task_id,
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

    @property
    def stream_idle_timeout(self) -> float:
        return self._stream_idle_timeout


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
        ingress: ServeForwardIngress,
    ) -> None:
        self.ingress = ingress
        super().__init__(address, handler)


class _IngressHandler(BaseHTTPRequestHandler):
    server: _IngressHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        return None

    def _serve(self) -> None:
        ingress = self.server.ingress
        declared = self.headers.get("content-length")
        if declared is not None and declared.isdigit():
            if int(declared) > _MAX_REQUEST_BYTES:
                self._refuse(413, "request body too large")
                return
            body = self.rfile.read(int(declared))
        else:
            body = b""
        outcome = ingress.handle(
            self.command, self.path, list(self.headers.items()), body
        )
        if isinstance(outcome, ServeIngressDenied):
            self._refuse(outcome.status, outcome.detail)
            return
        self._relay(outcome, ingress.stream_idle_timeout)

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
        # A HEAD response carries the headers its GET would produce and no body at all,
        # so its frames are drained to completion but never written out.
        body_allowed = self.command != "HEAD"
        try:
            while True:
                frame = channel.drain(idle_timeout)
                if frame is None:
                    if not started:
                        self._refuse(504, "resident serve timed out")
                    return
                if frame.kind == "head":
                    self._send_head(frame.status, frame.headers)
                    started = True
                elif frame.kind == "chunk":
                    if not started:
                        self._send_head(200, ())
                        started = True
                    if body_allowed and frame.payload:
                        self._write_chunk(frame.payload)
                elif frame.terminal:
                    if not started:
                        if frame.kind == "error":
                            self._refuse(502, frame.detail or "resident serve error")
                        else:
                            self._send_head(200, ())
                            if body_allowed:
                                self._end_chunks()
                        return
                    if body_allowed:
                        self._end_chunks()
                    return
        except (BrokenPipeError, ConnectionResetError):
            # The client is gone. Stop delivering; the claim settles on the sidecar's
            # own fenced terminal, never on this connection ending.
            channel.close()

    def _send_head(self, status: int, headers: tuple[tuple[str, str], ...]) -> None:
        self.send_response(status)
        for name, value in headers:
            # The body is re-framed for this hop, so the engine's own framing headers
            # are not carried onto it.
            if name.lower() not in ("content-length", "transfer-encoding"):
                self.send_header(name, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
