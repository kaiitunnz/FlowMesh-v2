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
    OCTET_STREAM,
    ContentHydrationError,
    ContentHydrationGrant,
    ContentReference,
)
from shared.network.frame_stream import WireFrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame

from .client import AnnounceHolding, ContentHydrationClient, RequestGrant
from .holder import ContentHolder
from .store import WorkerContentCache


class ContentLaneHost:
    """Hosts this worker's holder and hydration client on one loop."""

    def __init__(
        self,
        *,
        store: WorkerContentCache,
        push_frame: Callable[[dict[str, Any]], None],
        request_grant: RequestGrant,
        worker_id: str,
        generation: int,
        transfer_timeout_sec: float = 60.0,
        announce: AnnounceHolding | None = None,
        holder_report_ttl_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._push_frame = push_frame
        self._request_grant = request_grant
        self._worker_id = worker_id
        self._generation = generation
        self._transfer_timeout_sec = transfer_timeout_sec
        self._announce = announce
        self._holder_report_ttl_sec = holder_report_ttl_sec
        self._logger = logger or logging.getLogger("content-lane-host")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-content", daemon=True
        )
        self._holder: ContentHolder | None = None
        self._client: ContentHydrationClient | None = None
        self._sweep: asyncio.Task[None] | None = None

    @property
    def store(self) -> WorkerContentCache:
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
        self._sweep = asyncio.ensure_future(self._keep_held_reachable())

    def hydrate(self, reference: ContentReference, task_id: str) -> bytes:
        """Fetch one object, from this worker's own store or from its holder."""
        if self._store.holds(reference):
            self._logger.info(
                "content %s read from the local cache", reference.content_digest
            )
            return self._store.hydrate(reference)
        if self._client is None:
            raise RuntimeError("content lane is not started")
        client = self._client
        transfer = self._call(lambda: client.hydrate(reference, task_id))
        try:
            # The client bounds its own wait; this is the backstop for a transfer that
            # never returns at all, and it fails the read the same typed way.
            data = transfer.result(timeout=self._transfer_timeout_sec * 2)
            self._logger.info(
                "content %s read from a peer cache", reference.content_digest
            )
            return data
        except concurrent.futures.TimeoutError as exc:
            transfer.cancel()
            raise ContentHydrationError(
                f"hydrating {reference.content_digest} did not complete in time"
            ) from exc

    def keep(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference | None:
        """Cache a copy and keep the cache in bounds; the reference while it is held.

        None when the copy did not survive the bounds — an object larger than the whole
        disk budget — so nothing is announced for a copy that is not here.
        """
        reference = self._store.write(scope, data, media_type=media_type)
        self.evict()
        return reference if self._store.holds(reference) else None

    def evict(self) -> int:
        """Bring the cache within its bounds, leaving anything in transfer alone."""
        in_transfer = self._holder.in_transfer if self._holder else frozenset()
        return self._store.evict(in_transfer=in_transfer)

    def report_held(self) -> int:
        """Report every copy this worker holds, and return how many.

        A holder record lapses unless its holder keeps reporting, so this runs on a
        cadence inside the record's lifetime. It also runs before the worker takes any
        work, so a restarted worker's copies are reachable again rather than sitting
        unused on its disk: the report lands under its new incarnation, superseding the
        record its previous one left behind.
        """
        if self._announce is None:
            return 0
        held = list(self._store.iter_held())
        if held:
            self._announce(held)
        return len(held)

    @property
    def housekeeping_interval_sec(self) -> float:
        """How often the loop reports and sweeps.

        Several times within one holder-record lifetime, so a record is refreshed well
        before it lapses even if a report is missed.
        """
        return max(5.0, min(self._holder_report_ttl_sec / 3, 60.0))

    async def _keep_held_reachable(self) -> None:
        """Keep the cache reachable and bounded: report what is here, drop what is over.

        Eviction runs first so a copy on its way out is not advertised in the same
        breath; a peer that reads a copy this tick evicts finds it gone and falls
        through to the shared store, which is what the fall-through is for.
        """
        while True:
            await asyncio.sleep(self.housekeeping_interval_sec)
            try:
                # Both walk the cache directory, so they run off the loop that is also
                # carrying transfers and the control frames they depend on.
                if (evicted := await asyncio.to_thread(self.evict)) > 0:
                    self._logger.info("evicted %d cached content objects", evicted)
                await asyncio.to_thread(self.report_held)
            except Exception:
                self._logger.exception("content cache housekeeping failed")

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
