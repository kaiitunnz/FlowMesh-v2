"""Runs the worker's content lane on a dedicated asyncio loop.

The worker runtime is thread-based and the transfer protocol is asyncio, so the lane
owns one loop on its own thread: an executor thread asks it to hydrate a reference and
blocks on the result, while the interrupt-monitor thread marshals every control frame —
a grant for a transfer this worker requested, a grant against content it holds, a data
frame — onto the same loop. Produced frames leave as ``CONTENT_FRAME`` events over the
authenticated attachment, which the supervisor bridges onward opaquely.
"""

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from shared.content import (
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
)
from shared.network.frame_stream import WireFrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame

from .client import ContentHydrationClient, RequestGrant
from .holder import ContentHolder
from .store import WorkerObjectStore


class ContentLaneHost:
    """Hosts this worker's holder and hydration client on one loop."""

    def __init__(
        self,
        *,
        store: WorkerObjectStore,
        push_frame: Callable[[dict[str, Any]], None],
        request_grant: RequestGrant,
        worker_id: str,
        generation: int,
        transfer_timeout_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._push_frame = push_frame
        self._request_grant = request_grant
        self._worker_id = worker_id
        self._generation = generation
        self._transfer_timeout_sec = transfer_timeout_sec
        self._logger = logger or logging.getLogger("content-lane-host")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-content", daemon=True
        )
        self._holder: ContentHolder | None = None
        self._client: ContentHydrationClient | None = None
        self._sweep: asyncio.Task[None] | None = None

    @property
    def store(self) -> WorkerObjectStore:
        return self._store

    def start(self) -> None:
        self._thread.start()
        self._call(self._build).result()

    async def _build(self) -> None:
        sink = WireFrameSink(self._push_frame)
        self._holder = ContentHolder(
            store=self._store,
            sink=sink,
            holder_id=self._worker_id,
            generation=self._generation,
            logger=self._logger,
        )
        self._client = ContentHydrationClient(
            sink=sink,
            request_grant=self._request_grant,
            transfer_timeout_sec=self._transfer_timeout_sec,
            logger=self._logger,
        )
        self._sweep = asyncio.ensure_future(self._sweep_orphans())

    def hydrate(self, reference: ContentReference, task_id: str) -> bytes:
        """Fetch one object, from this worker's own store or from its holder."""
        if self._store.holds(reference):
            return self._store.hydrate(reference)
        if self._client is None:
            raise RuntimeError("content lane is not started")
        client = self._client
        transfer = self._call(lambda: client.hydrate(reference, task_id))
        try:
            # The client bounds its own wait; this is the backstop for a transfer that
            # never returns at all, and it fails the read the same typed way.
            return transfer.result(timeout=self._transfer_timeout_sec * 2)
        except concurrent.futures.TimeoutError as exc:
            transfer.cancel()
            raise ContentHydrationError(
                f"hydrating {reference.content_digest} did not complete in time"
            ) from exc

    def reclaim_orphans(self) -> int:
        """Sweep unbound writes, leaving anything a transfer is serving in place."""
        in_transfer = self._holder.in_transfer if self._holder else frozenset()
        return self._store.reclaim_orphans(in_transfer=in_transfer)

    async def _sweep_orphans(self) -> None:
        """Reclaim unbound writes on a cadence derived from their grace period.

        A write only becomes reclaimable once it is older than the grace, so sweeping a
        few times within one grace period is enough to keep a failed preparation's bytes
        from accumulating without ever racing a slow report.
        """
        interval = max(30.0, self._store.orphan_grace_sec / 4)
        while True:
            await asyncio.sleep(interval)
            try:
                if (reclaimed := self.reclaim_orphans()) > 0:
                    self._logger.info("reclaimed %d unbound content writes", reclaimed)
            except Exception:
                self._logger.exception("content orphan sweep failed")

    def route(self, frame_kind: str, frame: dict[str, Any]) -> bool:
        """Marshal one content control frame onto the lane loop; return handled."""
        if frame_kind == "content_grant":
            self._loop.call_soon_threadsafe(self._on_grant, frame)
        elif frame_kind == "content_grant_denied":
            self._loop.call_soon_threadsafe(self._on_denial, frame)
        elif frame_kind == "content_serve_grant":
            self._loop.call_soon_threadsafe(self._on_serve_grant, frame)
        elif frame_kind == "content_frame":
            asyncio.run_coroutine_threadsafe(self._on_frame(frame), self._loop)
        else:
            return False
        return True

    def _on_grant(self, frame: dict[str, Any]) -> None:
        if self._client is not None:
            self._client.deliver_grant(
                ContentHydrationGrant.model_validate(frame["grant"])
            )

    def _on_denial(self, frame: dict[str, Any]) -> None:
        if self._client is not None:
            self._client.deliver_denial(
                ContentReference.model_validate(frame["reference"]),
                str(frame.get("reason") or "denied"),
            )

    def _on_serve_grant(self, frame: dict[str, Any]) -> None:
        if self._holder is not None:
            self._holder.accept_grant(
                ContentHydrationGrant.model_validate(frame["grant"])
            )

    async def _on_frame(self, frame: dict[str, Any]) -> None:
        relay = RelayFrame.from_wire(frame)
        # A frame's direction names the receiver's role: the requester receives
        # target-to-origin, the holder receives origin-to-target.
        if relay.direction is RelayDirection.TARGET_TO_ORIGIN:
            if self._client is not None:
                await self._client.on_frame(relay)
        elif self._holder is not None:
            await self._holder.on_frame(relay)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(
        self, coro_fn: Callable[[], Coroutine[Any, Any, Any]]
    ) -> "concurrent.futures.Future[Any]":
        return asyncio.run_coroutine_threadsafe(coro_fn(), self._loop)

    def stop(self) -> None:
        if self._sweep is not None:
            self._loop.call_soon_threadsafe(self._sweep.cancel)
        if self._holder is not None:
            with contextlib.suppress(Exception):
                self._call(self._holder.aclose).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
