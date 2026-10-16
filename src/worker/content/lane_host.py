"""Runs the worker's content lane on a dedicated asyncio loop.

The worker runtime is thread-based and the transfer protocol is asyncio, so the lane
owns one loop on its own thread: an executor thread asks it to hydrate a reference and
blocks on the result, while the interrupt-monitor thread marshals every control frame —
a grant for a transfer this worker requested, a grant against content it holds, a data
frame — onto the same loop. Produced frames leave as ``CONTENT_FRAME`` events over the
authenticated attachment, which the supervisor bridges onward opaquely.

With peer transports on, the lane also serves the worker's content listener and dials
the holders control routes its own transfers to, both on this loop, so a dialed
transfer never crosses threads.
"""

import asyncio
import concurrent.futures
import contextlib
import logging
import shutil
import socket
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from shared.content import (
    HOLDING_RESYNC_FRAME_KIND,
    OCTET_STREAM,
    ContentCarriagePlan,
    ContentHydrationError,
    ContentHydrationGrant,
    ContentPeerRoute,
    ContentReference,
    ContentRouteObservation,
)
from shared.content.config import DEFAULT_TRANSFER_WINDOW_BYTES
from shared.network.frame_stream import FrameSink, PeerPurpose, WireFrameSink
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.mtls_listener import (
    ConnectionHandler,
    MutualTlsFrameListener,
    PeerContext,
)
from shared.network.relay_frame import CONTENT_FRAME_KIND, RelayDirection, RelayFrame
from shared.schemas.network import RouteObservationOutcome
from shared.utils.atomic import write_replacing
from shared.utils.nofollow import fd_path

from .client import ContentHydrationClient, RequestGrant
from .holder import ContentHolder
from .peer import ContentPeerDialer
from .store import WorkerContentCache

# A lane busy sending waits on the event stream; a re-registration waits this long.
_REBIND_WAIT_SEC = 30.0

# How often the cache is brought back within its bounds, when it has any.
_EVICTION_INTERVAL_SEC = 60.0

# Reports one dialed transfer's route evidence to control.
ObservationReport = Callable[[ContentRouteObservation], None]


class _HolderConnection(ConnectionHandler):
    """Hands one dialed connection's frames to the holder, answering over it."""

    def __init__(
        self, holder: ContentHolder, sink: FrameSink, peer: PeerContext
    ) -> None:
        self._holder = holder
        self._sink = sink
        self._peer = peer

    async def on_frame(self, frame: RelayFrame) -> None:
        await self._holder.on_frame(frame, self._sink, self._peer)

    def close(self) -> None:
        self._holder.drop_connection(self._sink)


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
        window_bytes: int = DEFAULT_TRANSFER_WINDOW_BYTES,
        peer_enabled: bool = False,
        peer_material: MutualTlsMaterial | None = None,
        listener_sock: socket.socket | None = None,
        connect_budget_sec: float = 5.0,
        report_observation: ObservationReport | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._push_frame = push_frame
        self._request_grant = request_grant
        self._worker_id = worker_id
        self._generation = generation
        self._transfer_timeout_sec = transfer_timeout_sec
        self._window_bytes = window_bytes
        self._peer_enabled = peer_enabled
        self._peer_material = peer_material
        self._listener_sock = listener_sock
        self._connect_budget_sec = connect_budget_sec
        self._report_observation = report_observation
        self._listener: MutualTlsFrameListener | None = None
        self._logger = logger or logging.getLogger("content-lane-host")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-content", daemon=True
        )
        self._holder: ContentHolder | None = None
        self._client: ContentHydrationClient | None = None
        self._sweep: asyncio.Task[None] | None = None
        # Whether a whole holding report asked for is queued or being walked.
        self._whole_lock = threading.Lock()
        self._whole_queued = False

    @property
    def store(self) -> WorkerContentCache:
        return self._store

    def start(self) -> None:
        self._thread.start()
        self._call(self._build).result()
        self._store.start_reporting()

    async def _build(self) -> None:
        sink = WireFrameSink(self._push_frame)
        self._holder = ContentHolder(
            store=self._store,
            sink=sink,
            holder_id=self._worker_id,
            generation=self._generation,
            window_bytes=self._window_bytes,
            stall_timeout_sec=self._transfer_timeout_sec,
            logger=self._logger,
        )
        dialer = None
        if self._peer_enabled:
            dialer = ContentPeerDialer(
                ssl_context=(
                    client_context(self._peer_material)
                    if self._peer_material is not None
                    else None
                ),
                connect_budget_sec=self._connect_budget_sec,
                observe=self._observe_route,
                logger=self._logger,
            )
        self._client = ContentHydrationClient(
            sink=sink,
            request_grant=self._request_grant,
            transfer_timeout_sec=self._transfer_timeout_sec,
            window_bytes=self._window_bytes,
            dialer=dialer,
            logger=self._logger,
        )
        if self._listener_sock is not None:
            holder = self._holder
            self._listener = MutualTlsFrameListener(
                material=self._peer_material,
                handler=lambda sink, peer: _HolderConnection(holder, sink, peer),
                purposes=frozenset({PeerPurpose.CONTENT}),
                idle_timeout_sec={PeerPurpose.CONTENT: self._transfer_timeout_sec},
                logger=self._logger,
            )
            await self._listener.start_on_socket(self._listener_sock)
        self._sweep = asyncio.ensure_future(self._keep_within_bounds())

    def _observe_route(
        self,
        carriage: ContentCarriagePlan,
        route: ContentPeerRoute,
        outcome: RouteObservationOutcome,
    ) -> None:
        if self._report_observation is None:
            return
        try:
            self._report_observation(
                ContentRouteObservation(
                    transfer_session_id=carriage.transfer_session_id,
                    transport=route.transport.value,
                    outcome=outcome.value,
                )
            )
        except Exception:
            self._logger.warning("could not report a content path", exc_info=True)

    def read_local(self, reference: ContentReference) -> bytes:
        """This worker's own verified copy of an object."""
        data = self._store.hydrate(reference)
        self._logger.info(
            "content %s read from the local cache", reference.content_digest
        )
        return data

    def copy_local(self, reference: ContentReference, dest: Path) -> None:
        """Copy this worker's own verified copy of an object to ``dest``."""
        self._store.hydrate_to(reference, dest)
        self._logger.info(
            "content %s read from the local cache", reference.content_digest
        )

    def link_local(self, reference: ContentReference, dest: Path) -> None:
        """Link this worker's own verified copy of an object at ``dest``."""
        self._store.link_to(reference, dest)
        self._logger.info(
            "content %s read from the local cache", reference.content_digest
        )

    def read_peer(
        self, reference: ContentReference, task_id: str, peer_only: bool = False
    ) -> bytes:
        """An object's verified bytes from the peer control resolves, kept here too.

        With ``peer_only`` the read skips the root relay.
        """
        with self._from_peer(reference, task_id, peer_only) as verified:
            return verified.read_bytes()

    def copy_peer(
        self,
        reference: ContentReference,
        task_id: str,
        dest: Path,
        peer_only: bool = False,
    ) -> None:
        """An object's verified bytes from a peer copied to ``dest``, kept here too."""
        with self._from_peer(reference, task_id, peer_only) as verified:
            _copy_file(verified, dest)

    @contextlib.contextmanager
    def _from_peer(
        self, reference: ContentReference, task_id: str, peer_only: bool
    ) -> Iterator[Path]:
        """The verified file a peer transfer lands in, taken into the cache once the
        caller has read it.

        The caller reads the transfer's own file rather than the cached copy, which an
        eviction is free to remove the moment it is adopted.
        """
        verified = self._transfer(reference, task_id, peer_only)
        try:
            yield verified
            self._logger.info(
                "content %s read from a peer cache", reference.content_digest
            )
            try:
                self._store.adopt(reference, verified)
            except Exception:
                self._logger.warning(
                    "could not cache %s locally",
                    reference.content_digest,
                    exc_info=True,
                )
        finally:
            verified.unlink(missing_ok=True)

    def _transfer(
        self, reference: ContentReference, task_id: str, peer_only: bool
    ) -> Path:
        if self._client is None:
            raise RuntimeError("content lane is not started")
        if peer_only and not self._peer_enabled:
            # Nothing here dials a peer, so asking control for one costs a round trip
            # for a refusal.
            raise ContentHydrationError("this worker carries no peer transport")
        client = self._client
        staging = self._store.staging_dir(reference.authorization_scope)
        transfer = self._call(
            lambda: client.hydrate_to_file(reference, task_id, staging, peer_only)
        )
        # The client bounds each wait by the stall timeout, so a transfer that keeps
        # making progress runs as long as it needs. This is the backstop for a lane
        # that stops making any progress at all, and it fails the read the same typed
        # way.
        while True:
            try:
                return transfer.result(timeout=self._transfer_timeout_sec)
            except concurrent.futures.TimeoutError as exc:
                idle = time.monotonic() - client.last_progress
                if idle < self._transfer_timeout_sec * 2:
                    continue
                if transfer.cancel():
                    raise ContentHydrationError(
                        f"hydrating {reference.content_digest} stopped making progress"
                    ) from exc
                # The transfer finished as the backstop gave up on it.
                return transfer.result()

    def keep(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference | None:
        """Cache a copy and keep the cache in bounds; the reference while it is held.

        None when the copy did not survive the bounds — an object larger than the whole
        disk budget.
        """
        return self.held(self._store.write(scope, data, media_type=media_type))

    def keep_file(
        self, reference: ContentReference, path: Path
    ) -> ContentReference | None:
        """Cache a copy of a file holding the reference's bytes, as ``keep`` does.

        The copy is hashed as it is taken, and one that turns out not to be the
        reference's bytes — the file changed after it was stored — is not kept.
        """
        copied = self._store.write_file(
            reference.authorization_scope, path, media_type=reference.media_type
        )
        if copied != reference:
            self._store.drop(copied)
            return None
        return self.held(reference)

    def keep_completed(
        self, reference: ContentReference, source: BinaryIO
    ) -> ContentReference | None:
        """Cache an opened completed file holding the reference's bytes as the file
        itself, as ``keep`` does, or a copy of it where the filesystem cannot link it
        into the cache."""
        if not self._store.link_completed(reference, source):
            return self.keep_file(reference, fd_path(source.fileno()))
        return self.held(reference)

    def held(self, reference: ContentReference) -> ContentReference | None:
        """The reference while a copy of it is here, once the cache is in bounds."""
        self.evict()
        return reference if self._store.holds(reference) else None

    def evict(self) -> int:
        """Bring the cache within its bounds, leaving anything in transfer alone, and
        return how many copies went."""
        in_transfer = self._holder.in_transfer if self._holder else frozenset()
        return self._store.evict(in_transfer=in_transfer)

    def rebind(self, holder_id: str, generation: int) -> None:
        """Hold this worker's copies under its new registration, and report them."""
        self._worker_id = holder_id
        self._generation = generation

        async def rebind_holder() -> None:
            if self._holder is not None:
                self._holder.rebind(holder_id, generation)

        try:
            self._call(rebind_holder).result(_REBIND_WAIT_SEC)
        except concurrent.futures.TimeoutError:
            self._logger.warning("The cache lane did not rebind its holder in time")
        self.report_held()

    def report_held(self) -> int:
        """Report every copy this worker holds as the first report of its registration,
        and return how many."""
        return self._store.report_whole(restart=True)

    async def _keep_within_bounds(self) -> None:
        """Bring the cache back within its bounds on a cadence, when it has any."""
        while True:
            await asyncio.sleep(_EVICTION_INTERVAL_SEC)
            try:
                # Eviction walks the cache directory, so it runs off the loop that is
                # also carrying transfers and the control frames they depend on.
                if evicted := await asyncio.to_thread(self.evict):
                    self._logger.info("evicted %d cached content objects", evicted)
            except Exception:
                self._logger.exception("content cache eviction failed")

    def route(self, frame_kind: str, frame: dict[str, Any]) -> bool:
        """Marshal one content control frame onto the lane loop; return handled."""
        if frame_kind == "content_grant":
            self._loop.call_soon_threadsafe(self._on_grant, frame)
        elif frame_kind == "content_grant_denied":
            self._loop.call_soon_threadsafe(self._on_denial, frame)
        elif frame_kind == "content_serve_grant":
            self._loop.call_soon_threadsafe(self._on_serve_grant, frame)
        elif frame_kind == CONTENT_FRAME_KIND:
            asyncio.run_coroutine_threadsafe(self._on_frame(frame), self._loop)
        elif frame_kind == HOLDING_RESYNC_FRAME_KIND:
            with self._whole_lock:
                # A walk already queued or under way reports what this one would.
                if self._whole_queued:
                    return True
                self._whole_queued = True
            # The whole holding walks the cache directory, so it is reported off both
            # the thread delivering control frames and the lane's loop.
            asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(self._report_whole), self._loop
            )
        else:
            return False
        return True

    def _report_whole(self) -> None:
        try:
            held = self._store.report_whole()
        except Exception:
            self._logger.exception("could not report the whole content holding")
            return
        finally:
            with self._whole_lock:
                self._whole_queued = False
        self._logger.info("reported %d held content objects on request", held)

    def _on_grant(self, frame: dict[str, Any]) -> None:
        if self._client is not None:
            carriage = frame.get("carriage")
            self._client.deliver_grant(
                int(frame["request_id"]),
                ContentHydrationGrant.model_validate(frame["grant"]),
                (
                    ContentCarriagePlan.model_validate(carriage)
                    if carriage is not None
                    else None
                ),
            )

    def _on_denial(self, frame: dict[str, Any]) -> None:
        if self._client is not None:
            self._client.deliver_denial(
                int(frame["request_id"]), str(frame.get("reason") or "denied")
            )

    def _on_serve_grant(self, frame: dict[str, Any]) -> None:
        if self._holder is not None:
            self._holder.accept_grant(
                ContentHydrationGrant.model_validate(frame["grant"]),
                frame.get("requester_host") or None,
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

    def stop(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout

        def left() -> float:
            return max(0.0, deadline - time.monotonic())

        if self._sweep is not None:
            self._loop.call_soon_threadsafe(self._sweep.cancel)
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._call(self._listener.stop).result(timeout=min(5.0, left()))
        if self._holder is not None:
            with contextlib.suppress(Exception):
                self._call(self._holder.aclose).result(timeout=min(5.0, left()))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=min(5.0, left()))
        self._store.stop_reporting()


def _copy_file(source: Path, dest: Path) -> None:
    """Copy a file to ``dest``, which appears only whole."""
    with source.open("rb") as src:
        write_replacing(dest, lambda fh: shutil.copyfileobj(src, fh))
