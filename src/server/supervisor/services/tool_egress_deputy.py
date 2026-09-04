"""Supervisor-side router for worker-hosted external-tool egress.

On a node selected as an egress target this binds a claim-free frame endpoint per target
and forwards each opaque operation frame to the target worker over that worker's own
authenticated supervisor attachment; the worker returns the opaque result over the same
attachment and this deputy writes it back to the origin. The supervisor is a pure opaque
router: it never decodes the operation envelope, request, or outcome, never constructs a
provider or reads a credential, and correlates a reply only by an opaque transport
session id. The provider egress runs in the worker executor.
"""

import asyncio
import base64
import logging
from typing import Any

from shared.tools.wire import FRAME_CANCEL, FRAME_OPERATION, FRAME_REAP, FRAME_REPLY
from shared.utils.ids import new_tool_relay_session_id

from ...network import wire as netwire
from .task_listener import TaskListener


class _Session:
    """One in-flight operation: the origin writer and the future the reply resolves."""

    __slots__ = ("worker_id", "future", "writer", "reap_handle")

    def __init__(
        self,
        worker_id: str,
        future: "asyncio.Future[bytes]",
        writer: asyncio.StreamWriter,
    ) -> None:
        self.worker_id = worker_id
        self.future = future
        self.writer = writer
        self.reap_handle: asyncio.TimerHandle | None = None


class WorkerToolRouteDeputy:
    """Binds per-target frame endpoints and routes opaque tool frames to workers."""

    def __init__(
        self,
        *,
        task_listener: TaskListener,
        recv_timeout_sec: float = 60.0,
        reap_ttl_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._task_listener = task_listener
        self._recv_timeout = recv_timeout_sec
        self._reap_ttl = reap_ttl_sec
        self._log = logger or logging.getLogger("worker-tool-route-deputy")
        self._listeners: dict[str, _RouteListener] = {}
        self._sessions: dict[str, _Session] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def bind_sidecar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind (or rebind) the frame endpoint that routes a target to its worker."""
        self._loop = asyncio.get_running_loop()
        target_id = str(payload["target_id"])
        worker_id = str(payload.get("worker_id") or target_id)
        listener = _RouteListener(self, worker_id, route=str(payload["route"]))
        await self._drop(target_id)
        host, port = await listener.start()
        self._listeners[target_id] = listener
        self._log.info(
            "tool route bound target=%s worker=%s route=%s:%s",
            target_id,
            worker_id,
            host,
            port,
        )
        return {"bound": True, "host": host, "port": port}

    async def unbind_sidecar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop a target's frame endpoint; absent is a no-op."""
        target_id = str(payload["target_id"])
        await self._drop(target_id)
        self._log.info("tool route unbound target=%s", target_id)
        return {"unbound": True}

    async def stop(self) -> None:
        for target_id in list(self._listeners):
            await self._drop(target_id)

    async def _drop(self, target_id: str) -> None:
        listener = self._listeners.pop(target_id, None)
        if listener is not None:
            await listener.stop()

    async def serve_operation(
        self,
        worker_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Forward one opaque operation frame to the worker and write back its reply."""
        raw = await netwire.read_frame(reader)
        session_id = new_tool_relay_session_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._sessions[session_id] = _Session(worker_id, fut, writer)
        if not self._task_listener.enqueue_egress(
            worker_id, session_id, FRAME_OPERATION, raw
        ):
            # The target worker is not attached: leave it ambiguous for the origin's
            # re-drive rather than manufacture a terminal outcome here.
            self._pop(session_id)
            return
        try:
            reply = await asyncio.wait_for(fut, self._recv_timeout)
        except TimeoutError:
            self._reap(worker_id, session_id)
            return
        except asyncio.CancelledError:
            self._reap(worker_id, session_id)
            raise
        try:
            # The worker's reply arrived and the egress is complete; only the write back
            # to the origin can still fail (a disconnected origin). Pop the session
            # regardless so a failed write-back never leaks the record.
            await netwire.write_frame(writer, reply)
        finally:
            self._pop(session_id)

    def deliver_up(self, session_id: str, kind: str, frame_b64: str) -> None:
        """Route a worker's opaque up-leg frame back to its origin, by session id."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._deliver_up, session_id, kind, frame_b64)

    def _deliver_up(self, session_id: str, kind: str, frame_b64: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            # A stale, duplicate, or unknown-session frame is dropped, never routed.
            return
        if kind == FRAME_REPLY:
            if not session.future.done():
                session.future.set_result(base64.b64decode(frame_b64))
        elif kind == FRAME_REAP:
            self._pop(session_id)

    def _reap(self, worker_id: str, session_id: str) -> None:
        # A lost or slow reply: forward a cancel and retain the record until the worker
        # reaps it or the bounded reaper fires, never manufacturing a terminal outcome.
        self._task_listener.enqueue_egress(worker_id, session_id, FRAME_CANCEL, b"")
        session = self._sessions.get(session_id)
        if session is not None and self._loop is not None:
            session.reap_handle = self._loop.call_later(
                self._reap_ttl, self._pop, session_id
            )

    def _pop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None and session.reap_handle is not None:
            session.reap_handle.cancel()


class _RouteListener:
    """A per-target TCP frame endpoint that hands each connection to the deputy."""

    def __init__(
        self, deputy: WorkerToolRouteDeputy, worker_id: str, *, route: str
    ) -> None:
        self._deputy = deputy
        self._worker_id = worker_id
        self._host, self._port = netwire.split_host_port(route)
        self._tcp: asyncio.Server | None = None
        self._conns: set[asyncio.Task[None]] = set()

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conns.add(task)
        try:
            await self._deputy.serve_operation(self._worker_id, reader, writer)
        except (
            ValueError,
            ConnectionError,
            asyncio.IncompleteReadError,
            OSError,
        ):
            pass
        finally:
            await _close(writer)
            if task is not None:
                self._conns.discard(task)

    async def start(self) -> tuple[str, int]:
        self._tcp = await asyncio.start_server(
            self._on_connection, self._host, self._port
        )
        host, port = self._tcp.sockets[0].getsockname()[:2]
        return host, port

    async def stop(self) -> None:
        if self._tcp is None:
            return
        self._tcp.close()
        for task in list(self._conns):
            task.cancel()
        if self._conns:
            await asyncio.gather(*self._conns, return_exceptions=True)
        try:
            await self._tcp.wait_closed()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
        self._tcp = None


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass


__all__ = ["WorkerToolRouteDeputy"]
