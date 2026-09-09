"""One in-flight forward-ingress request's response channel.

The origin drive runs on the worker's resident lane loop while the client's connection
is served on a listener thread, so relayed response frames cross between them here. The
drive offers the engine's head, its opaque body frames, and the terminal; the connection
drains them in order and writes them out.

The backlog is bounded: a client that stops draining cannot pin unbounded memory. A
frame that would overflow the bound is not silently dropped into a clean completion —
the channel records the loss so the connection is aborted at the terminal rather than
closed as if it kept every byte. The terminal itself always lands, evicting the oldest
frame if it must. Neither a dropped frame nor a gone client touches the claim: only the
sidecar's fenced terminal releases its credit.
"""

import queue
from collections.abc import Callable
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
    _lost: bool = False
    _committed: bool = False
    _committed_cb: Callable[[int, tuple[tuple[str, str], ...]], None] | None = None

    @property
    def lost(self) -> bool:
        """Whether a body frame was dropped, so a clean completion would lie."""
        return self._lost

    def on_committed(
        self, cb: Callable[[int, tuple[tuple[str, str], ...]], None]
    ) -> None:
        """Register the hook fired once the client head is written to the socket."""
        self._committed_cb = cb

    def commit_head(self, status: int, headers: tuple[tuple[str, str], ...]) -> None:
        """Signal that the client head is written, exactly once per request."""
        if self._committed:
            return
        self._committed = True
        if self._committed_cb is not None:
            self._committed_cb(status, headers)

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
            # The client is not draining fast enough. Record the gap rather than drop
            # the frame silently, so the terminal aborts the connection instead of
            # implying a response that kept every byte.
            self._lost = True

    def _finish(self, frame: ServeIngressFrame) -> None:
        if self._closed:
            return
        self._closed = True
        # The terminal must reach the connection so its response closes; if the bounded
        # backlog is full, evict the oldest frame to make room — an evicted frame is a
        # gap the connection turns into an aborted delivery.
        while True:
            try:
                self.frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self.frames.get_nowait()
                    self._lost = True
                except queue.Empty:
                    return

    def drain(self, timeout: float) -> ServeIngressFrame | None:
        """The next frame, or None when none arrives before the deadline."""
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None
