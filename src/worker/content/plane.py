"""This worker's view of fabric content.

Content lives in the shared durable store. A worker writes an object there before it
reports the reference naming it, so every reference that reaches a binding names bytes
that already outlive the worker that produced them, and it keeps a copy of what it wrote
so the next read of it is local.

A read tries this worker's own copy, then another worker's over an authorized transfer,
then the shared store, which always has the object; each level only ever saves a trip
to the next. An object of the configured relay size or larger is read from the store
before the root relay, and over the relay only while the store cannot be reached; a
holder a peer transport reaches is read first at any size. So a cache miss
costs a read rather than a failure, and a worker dying costs nothing at all. Whatever
this worker reads from a peer or the store it keeps a copy of, so the next reader here,
or on a peer, finds one.

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
    ContentStoreError,
    ContentUnavailable,
    FabricObjectStore,
)
from shared.content.config import DEFAULT_RELAY_TRANSFER_MAX_BYTES
from shared.outcome import (
    FabricContentStore,
    FinalizationIndexClient,
    FinalizingContentStore,
)

from .access import ContentAccessDenied, ContentAccessRegistry
from .lane_host import ContentLaneHost


class _Missed:
    """A read from a copy that found none."""


_MISSED = _Missed()


class WorkerContentPlane:
    """The worker's content surface, and the per-task stores taken from it."""

    def __init__(
        self,
        lane: ContentLaneHost | None,
        access: ContentAccessRegistry,
        *,
        finalizations: FinalizationIndexClient | None = None,
        relay_max_bytes: int = DEFAULT_RELAY_TRANSFER_MAX_BYTES,
        logger: logging.Logger | None = None,
    ) -> None:
        self._lane = lane
        self._access = access
        self._finalizations = finalizations
        self._relay_max_bytes = relay_max_bytes
        self._logger = logger or logging.getLogger("content-plane")

    def for_task(self, task_id: str, scope: str) -> "TaskContentStore":
        """The object surface one task reads and writes through, reading only in the
        scope its dispatch carries."""
        return TaskContentStore(self, task_id, scope)

    def outcome_store(self, task_id: str) -> FabricContentStore | None:
        """Where one task's outcomes materialize, when it can finalize at all."""
        if self._finalizations is None:
            return None
        return FinalizingContentStore(
            _TaskScopedStores(self, task_id), self._finalizations
        )

    @property
    def holding_seq(self) -> int:
        """The number of the last holding report this worker sent, 0 without a cache."""
        return self._lane.store.holding_seq if self._lane is not None else 0

    def flush_holding(self) -> None:
        """Send the holding changes this worker's cache is collecting."""
        if self._lane is not None:
            self._lane.store.flush_holding()

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
        """The object's verified bytes, from the nearest place that has them.

        The caller holds the task to the scope it was dispatched in; only a read from
        the shared store needs the task's live access.
        """
        scope = reference.authorization_scope
        return self._read(
            reference,
            local=lambda lane: lane.read_local(reference),
            peer=lambda lane, peer_only: lane.read_peer(reference, task_id, peer_only),
            stored=lambda: self._store(task_id, scope).hydrate(reference),
            keep=lambda lane, data: lane.keep(
                reference.authorization_scope, data, media_type=reference.media_type
            ),
        )

    def hydrate_to(self, task_id: str, reference: ContentReference, dest: Path) -> None:
        """Write the object's verified bytes to ``dest``, from the nearest place that
        has them, never holding the object whole."""
        scope = reference.authorization_scope
        self._read(
            reference,
            local=lambda lane: lane.copy_local(reference, dest),
            peer=lambda lane, peer_only: lane.copy_peer(
                reference, task_id, dest, peer_only
            ),
            stored=lambda: self._store(task_id, scope).hydrate_to(reference, dest),
            keep=lambda lane, _: lane.keep_file(reference, dest),
        )

    def materialize_to(
        self, task_id: str, reference: ContentReference, dest: Path
    ) -> None:
        """Put the object's verified bytes at ``dest`` for a reader to load.

        Where this worker keeps a copy, ``dest`` links to it, so what a task reads
        counts against the cache's own bounds.
        """
        lane = self._lane
        if lane is None:
            self.hydrate_to(task_id, reference, dest)
            return

        def link() -> None:
            lane.link_local(reference, dest)

        if not isinstance(self._attempt(reference, link), _Missed):
            return
        staged = dest.with_name(f"{dest.name}.fetching")
        self.hydrate_to(task_id, reference, staged)
        if not isinstance(self._attempt(reference, link), _Missed):
            staged.unlink(missing_ok=True)
            return
        staged.rename(dest)

    def open(
        self,
        task_id: str,
        reference: ContentReference,
        *,
        chunk_bytes: int = STREAM_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """The object's bytes in chunks, from this worker's copy or else the shared
        store, raising before the last chunk unless they verify.

        A copy that fails verification part way through is dropped as it raises, so
        the next open reads the store.
        """
        if (lane := self._lane) is not None:
            local = self._attempt(
                reference, lambda: lane.store.open(reference, chunk_bytes=chunk_bytes)
            )
            if not isinstance(local, _Missed):
                return local
        return self._store(task_id, reference.authorization_scope).open(
            reference, chunk_bytes=chunk_bytes
        )

    def _read[T](
        self,
        reference: ContentReference,
        *,
        local: Callable[[ContentLaneHost], T],
        peer: Callable[[ContentLaneHost, bool], T],
        stored: Callable[[], T],
        keep: Callable[[ContentLaneHost, T], ContentReference | None],
    ) -> T:
        """Read from this worker's copy, a peer's, or the shared store, in that order;
        keep a copy of what the store supplied.

        A holder reached over a peer transport is read before the store at any size;
        the root relay comes before the store only for an object worth relaying, and at
        any size while the store cannot be reached.
        """
        lane = self._lane
        if lane is not None:
            if not isinstance(
                got := self._attempt(reference, lambda: local(lane)), _Missed
            ):
                return got
            relayable = self._relayable(reference)
            if not isinstance(
                got := self._attempt(reference, lambda: peer(lane, not relayable)),
                _Missed,
            ):
                return self._kept(reference, lane, got)
        try:
            value = stored()
        except (ContentUnavailable, ContentAccessDenied) as outage:
            if lane is None or self._relayable(reference):
                raise
            return self._kept(
                reference, lane, self._relay_during(outage, lambda: peer(lane, False))
            )
        self._logger.info(
            "content %s read from the shared store", reference.content_digest
        )
        if lane is not None:
            self._cache(reference, lambda: keep(lane, value))
        return value

    def _kept[T](
        self, reference: ContentReference, lane: ContentLaneHost, value: T
    ) -> T:
        """Keep the copy a peer read left here within the cache's bounds."""
        self._cache(reference, lambda: lane.held(reference))
        return value

    def _relayable(self, reference: ContentReference) -> bool:
        """Whether the root relay is worth trying before the store for this object.

        A relayed transfer costs more per byte than a store read, so past a configured
        size the store is read first. This weighs cost only: an object of any size is
        relayed while the store cannot be reached.
        """
        return (
            self._relay_max_bytes <= 0 or reference.size_bytes < self._relay_max_bytes
        )

    def _relay_during[T](self, outage: ContentStoreError, read: Callable[[], T]) -> T:
        """A peer's copy while the store is down or out of the task's reach.

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
    ) -> T | _Missed:
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
        """Keep a copy, which the cache reports; a failure costs only the next read."""
        try:
            keep()
        except Exception:
            self._logger.warning(
                "could not cache %s locally", reference.content_digest, exc_info=True
            )


class _TaskScopedStores:
    """One task's content plane, presented as the store each scope is reached through.

    An outcome is content like any other: it goes through the plane, so it lands in the
    shared store, stays in this worker's cache, and is reported there — a later reader
    can then be served the copy instead of paying for the store.
    """

    def __init__(self, plane: "WorkerContentPlane", task_id: str) -> None:
        self._plane = plane
        self._task_id = task_id

    def for_scope(self, scope: str) -> FabricObjectStore:
        return TaskContentStore(self._plane, self._task_id, scope)


class TaskContentStore(FabricObjectStore):
    """One task's object surface over the worker's content plane.

    It reads only in the scope the task was dispatched in, whatever copy can serve the
    read: a copy held here or by a peer is no more the task's to read than the store
    itself is.
    """

    def __init__(self, plane: WorkerContentPlane, task_id: str, scope: str) -> None:
        self._plane = plane
        self._task_id = task_id
        self._scope = scope

    def _authorized(self, reference: ContentReference) -> ContentReference:
        if reference.authorization_scope != self._scope:
            raise ContentAccessDenied(
                f"task {self._task_id} reads in scope {self._scope}, not "
                f"{reference.authorization_scope}"
            )
        return reference

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._plane.write(self._task_id, scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._plane.hydrate(self._task_id, self._authorized(reference))

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
        return self._plane.open(
            self._task_id, self._authorized(reference), chunk_bytes=chunk_bytes
        )

    def hydrate_to(self, reference: ContentReference, dest: Path) -> None:
        self._plane.hydrate_to(self._task_id, self._authorized(reference), dest)
