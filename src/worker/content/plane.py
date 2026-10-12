"""This worker's view of fabric content.

Content lives in the shared durable store. A worker writes an object there before it
reports the reference naming it, so every reference that reaches a binding names bytes
that already outlive the worker that produced them, and it keeps a copy of what it wrote
so the next read of it is local.

Reading walks three levels, and each is only ever an optimization over the last. What
this worker already has, it reads locally. What another worker has, it hydrates over an
authorized transfer, which saves a trip to the shared store for an object small enough
that the relay is the cheaper path. What no copy can supply — nothing holds it, the
holder died, the grant expired — it reads from the shared store, which always has it.
An object too large to be worth relaying goes straight to the store, and to a peer only
while the store cannot be reached. So a cache miss costs a read rather than a failure,
and a worker dying costs nothing at all. Whatever this worker reads from a peer or the
store it keeps a copy of, so the next reader here, or on a peer, finds one.

The cache is the optional half. A deployment that runs none has no lane here, and every
read and write goes straight to the shared store, which is where the content is either
way.

Everything is taken per task, because that is how access is given: control grants one
task the right to reach one scope's content in the shared store, and the grant control
mints for a peer's copy rests on that task's own binding.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from shared.content import (
    OCTET_STREAM,
    STREAM_CHUNK_BYTES,
    ContentReference,
    ContentStoreAccess,
    ContentUnavailable,
    FabricObjectStore,
)
from shared.outcome import (
    FabricContentStore,
    FinalizationIndexClient,
    FinalizingContentStore,
)

from .access import ContentAccessRegistry
from .client import AnnounceHolding
from .lane_host import ContentLaneHost


class WorkerContentPlane:
    """The worker's content surface, and the per-task stores taken from it."""

    def __init__(
        self,
        lane: ContentLaneHost | None,
        access: ContentAccessRegistry,
        *,
        announce: AnnounceHolding | None = None,
        finalizations: FinalizationIndexClient | None = None,
        peer_max_bytes: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._lane = lane
        self._access = access
        self._announce = announce
        self._finalizations = finalizations
        self._peer_max_bytes = peer_max_bytes
        self._logger = logger or logging.getLogger("content-plane")

    def for_task(self, task_id: str) -> "TaskContentStore":
        """The object surface one task reads and writes through."""
        return TaskContentStore(self, task_id)

    def outcome_store(self, task_id: str) -> FabricContentStore | None:
        """Where one task's outcomes materialize, when it can finalize at all."""
        if self._finalizations is None:
            return None
        return FinalizingContentStore(
            _TaskScopedStores(self, task_id), self._finalizations
        )

    def start(self) -> None:
        """Bring up the cache lane, if this worker runs one, and report what it holds.

        The report lands before this worker takes any work: an object already on its
        disk from a previous incarnation is unreachable until control hears who holds
        it, and the report has to land under the incarnation registration just assigned.
        """
        if self._lane is None:
            return
        self._lane.start()
        if (held := self._lane.report_held()) > 0:
            self._logger.info("reported %d held content objects at startup", held)

    def rebind(self, holder_id: str, generation: int) -> None:
        """Move to the worker's new registration: drop the store access granted to the
        previous one, and hold the cached copies under the new one."""
        self._access.clear()
        if self._lane is not None:
            self._lane.rebind(holder_id, generation)

    def stop(self, timeout: float = 10.0) -> None:
        """Drain the cache lane, if this worker runs one."""
        if self._lane is not None:
            self._lane.stop(timeout)

    def accept_access(self, access: ContentStoreAccess) -> None:
        """Take the access control granted one of this worker's tasks."""
        self._access.accept(access)

    def route(self, frame_kind: str, frame: dict[str, object]) -> None:
        """Hand one content control frame to whatever consumes it."""
        if frame_kind == "content_access":
            self._access.accept(ContentStoreAccess.model_validate(frame))
            return
        if self._lane is not None:
            self._lane.route(frame_kind, frame)

    def write(
        self, task_id: str, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        """Put the bytes in the shared store, keep a copy, and say where the copy is.

        The shared write comes first and its failure is the write's failure: nothing may
        report a reference the durable store does not already have. Caching it and
        telling control about the copy are what make the next read cheap, so a failure
        there costs a local read rather than the object.
        """
        reference = self._store(task_id, scope).write(
            scope, data, media_type=media_type
        )
        if (lane := self._lane) is not None:
            self._cache(
                reference, lambda: lane.keep(scope, data, media_type=media_type)
            )
        return reference

    def write_file(
        self,
        task_id: str,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Put a file's bytes in the shared store, then keep a copy as ``write`` does.

        The file is streamed and never held whole; the cache takes a copy of it rather
        than the file itself, which stays its producer's to change or remove.
        """
        reference = self._store(task_id, scope).write_file(
            scope, path, media_type=media_type, cancelled=cancelled
        )
        if (lane := self._lane) is not None:
            self._cache(reference, lambda: lane.keep_file(reference, path))
        return reference

    def hydrate(self, task_id: str, reference: ContentReference) -> bytes:
        """The object's verified bytes, from the nearest place that has them."""
        lane = self._lane
        if lane is not None:
            local = self._attempt(reference, lambda: lane.read_local(reference))
            if not isinstance(local, _Missed):
                return local
        if lane is not None and self._relayable(reference):
            peer = self._attempt(reference, lambda: lane.read_peer(reference, task_id))
            if not isinstance(peer, _Missed):
                return peer
        try:
            data = self._store(task_id, reference.authorization_scope).hydrate(
                reference
            )
        except ContentUnavailable as outage:
            if lane is None or self._relayable(reference):
                raise
            return self._relay_during(
                outage, lambda: lane.read_peer(reference, task_id)
            )
        self._logger.info(
            "content %s read from the shared store", reference.content_digest
        )
        if lane is not None:
            self._cache(
                reference,
                lambda: lane.keep(
                    reference.authorization_scope,
                    data,
                    media_type=reference.media_type,
                ),
            )
        return data

    def hydrate_to(self, task_id: str, reference: ContentReference, dest: Path) -> None:
        """Write the object's verified bytes to ``dest``, from the nearest place that
        has them, never holding the object whole."""
        lane = self._lane
        if lane is not None:
            local = self._attempt(reference, lambda: lane.copy_local(reference, dest))
            if not isinstance(local, _Missed):
                return
        if lane is not None and self._relayable(reference):
            peer = self._attempt(
                reference, lambda: lane.copy_peer(reference, task_id, dest)
            )
            if not isinstance(peer, _Missed):
                return
        store = self._store(task_id, reference.authorization_scope)
        try:
            store.hydrate_to(reference, dest)
        except ContentUnavailable as outage:
            if lane is None or self._relayable(reference):
                raise
            self._relay_during(outage, lambda: lane.copy_peer(reference, task_id, dest))
            return
        self._logger.info(
            "content %s read from the shared store", reference.content_digest
        )
        if lane is not None:
            self._cache(reference, lambda: lane.keep_file(reference, dest))

    def open(
        self,
        task_id: str,
        reference: ContentReference,
        *,
        chunk_bytes: int = STREAM_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """The object's bytes in chunks, from this worker's copy or the shared store,
        raising before the last chunk unless they verify."""
        if (lane := self._lane) is not None:
            local = self._attempt(
                reference, lambda: lane.store.open(reference, chunk_bytes=chunk_bytes)
            )
            if not isinstance(local, _Missed):
                return local
        return self._store(task_id, reference.authorization_scope).open(
            reference, chunk_bytes=chunk_bytes
        )

    def _relayable(self, reference: ContentReference) -> bool:
        """Whether a peer is worth trying before the store for this object.

        A relayed transfer costs more per byte than a store read, so past a configured
        size the store is read first. This weighs cost only: an object of any size is
        relayed while the store cannot be reached.
        """
        return self._peer_max_bytes <= 0 or reference.size_bytes < self._peer_max_bytes

    def _relay_during[T](self, outage: ContentUnavailable, read: Callable[[], T]) -> T:
        """A peer's copy while the store is down.

        When no peer can serve it, the read fails as the outage it is, so the task
        waits for the store rather than failing on a peer's miss.
        """
        try:
            return read()
        except Exception as miss:
            raise outage from miss

    def _store(self, task_id: str, scope: str) -> FabricObjectStore:
        return self._access.store_for(task_id, scope)

    def _attempt[T](
        self, reference: ContentReference, read: Callable[[], T]
    ) -> T | "_Missed":
        """A read from a copy, or a miss.

        Every cache path is optional, and that includes a cache that cannot represent
        the object, a content directory that has gone unreadable, and a lane that is
        not running. The object is in the shared store whatever happened to a copy of
        it, so anything the cache raises costs this read and nothing else.
        """
        try:
            result = read()
        except Exception as miss:
            self._logger.debug(
                "content %s not read from a copy: %s", reference.content_digest, miss
            )
            return _MISSED
        return result

    def _cache(
        self,
        reference: ContentReference,
        keep: Callable[[], ContentReference | None],
    ) -> None:
        """Keep a copy and announce it; a failure costs only the next read."""
        try:
            if keep() is not None and self._announce is not None:
                self._announce(
                    [(reference.authorization_scope, reference.content_digest)]
                )
        except Exception:
            self._logger.warning(
                "could not cache %s locally", reference.content_digest, exc_info=True
            )


class _Missed:
    """A read from a copy that found none."""


_MISSED = _Missed()


class _TaskScopedStores:
    """One task's content plane, presented as the store each scope is reached through.

    An outcome is content like any other: it goes through the plane, so it lands in the
    shared store, stays in this worker's cache, and is announced there — a later reader
    can then be served the copy instead of paying for the store.
    """

    def __init__(self, plane: "WorkerContentPlane", task_id: str) -> None:
        self._plane = plane
        self._task_id = task_id

    def for_scope(self, scope: str) -> FabricObjectStore:
        return TaskContentStore(self._plane, self._task_id)


class TaskContentStore(FabricObjectStore):
    """One task's object surface over the worker's content plane."""

    def __init__(self, plane: WorkerContentPlane, task_id: str) -> None:
        self._plane = plane
        self._task_id = task_id

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._plane.write(self._task_id, scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._plane.hydrate(self._task_id, reference)

    def hydrate(self, reference: ContentReference) -> bytes:
        # Every path a read can take here verifies the bytes against this exact
        # reference before returning them, so there is nothing left to check.
        return self.fetch(reference)

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        return self._plane.write_file(
            self._task_id, scope, path, media_type=media_type, cancelled=cancelled
        )

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        return self._plane.open(self._task_id, reference, chunk_bytes=chunk_bytes)

    def hydrate_to(self, reference: ContentReference, dest: Path) -> None:
        self._plane.hydrate_to(self._task_id, reference, dest)
