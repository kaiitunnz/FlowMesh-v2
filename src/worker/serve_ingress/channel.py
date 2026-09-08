"""One in-flight forward-ingress request's response channel.

The origin drive runs on the worker's resident lane loop while the client's connection
is served on a listener thread, so relayed response frames cross between them here. The
drive offers the engine's head, its opaque body frames, and the terminal; the connection
drains them in order and writes them out.

The backlog is bounded: a client that stops draining cannot pin unbounded memory, so
frames past the bound are dropped while the terminal always lands — evicting the oldest
frame if it must — so the connection always closes. Dropping a frame never touches the
claim: only the sidecar's fenced terminal releases its credit.
"""

import queue
from dataclasses import dataclass, field

# The bound on one client's undrained frame backlog.
_QUEUE_MAX = 2048


@dataclass(frozen=True)
class ServeIngressFrame:
    """One event on a request's response stream: the head, a chunk, or a terminal."""

    kind: str  # "head" | "chunk" | "done" | "error"
    payload: bytes = b""
    status: int = 200
    headers: tuple[tuple[str, str], ...] = ()
    detail: str | None = None

    @property
    def terminal(self) -> bool:
        return self.kind in ("done", "error")


@dataclass
class ServeIngressChannel:
    """The response frames one request's connection drains, in order."""

    frames: queue.Queue[ServeIngressFrame] = field(
        default_factory=lambda: queue.Queue(maxsize=_QUEUE_MAX)
    )
    _closed: bool = False

    def head(self, status: int, headers: tuple[tuple[str, str], ...]) -> None:
        self._offer(ServeIngressFrame(kind="head", status=status, headers=headers))

    def chunk(self, payload: bytes) -> None:
        self._offer(ServeIngressFrame(kind="chunk", payload=payload))

    def complete(self) -> None:
        self._finish(ServeIngressFrame(kind="done"))

    def fail(self, detail: str) -> None:
        self._finish(ServeIngressFrame(kind="error", detail=detail))

    def close(self) -> None:
        """Stop delivering to a gone client; the claim settles on its own terminal."""
        self._closed = True

    def _offer(self, frame: ServeIngressFrame) -> None:
        if self._closed:
            return
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def _finish(self, frame: ServeIngressFrame) -> None:
        if self._closed:
            return
        self._closed = True
        # The terminal must reach the connection so its response closes; if the bounded
        # backlog is full, evict the oldest frame to make room for it.
        while True:
            try:
                self.frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    return

    def drain(self, timeout: float) -> ServeIngressFrame | None:
        """The next frame, or None when none arrives before the deadline."""
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None
