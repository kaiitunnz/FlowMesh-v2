"""This worker's view of fabric content.

Content lives in the shared durable store. A worker writes an object there before it
reports the reference naming it, so every reference that reaches a binding names bytes
that already outlive the worker that produced them, and it keeps a copy of what it wrote
so the next read of it is local.

Reading walks three levels, and each is only ever an optimization over the last. What
this worker already has, it reads locally. What another worker has, it hydrates over an
authorized transfer, which saves a trip to the shared store on the path that matters.
What no copy can supply — nothing holds it, the holder died, the grant expired — it
reads from the shared store, which always has it. So a cache miss costs a read rather
than a failure, and a worker dying costs nothing at all.

The cache is the optional half. A deployment that runs none has no lane here, and every
read and write goes straight to the shared store, which is where the content is either
way.

Everything is taken per task, because that is how access is given: control grants one
task the right to reach one scope's content in the shared store, and the grant control
mints for a peer's copy rests on that task's own binding.
"""

import logging

from shared.content import (
    OCTET_STREAM,
    ContentReference,
    ContentStoreAccess,
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
        logger: logging.Logger | None = None,
    ) -> None:
        self._lane = lane
        self._access = access
        self._announce = announce
        self._finalizations = finalizations
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

    def stop(self) -> None:
        """Drain the cache lane, if this worker runs one."""
        if self._lane is not None:
            self._lane.stop()

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
        reference = self._access.store_for(task_id, scope).write(
            scope, data, media_type=media_type
        )
        if self._lane is None or self._announce is None:
            return reference
        try:
            self._lane.store.write(scope, data, media_type=media_type)
            self._announce([(scope, reference.content_digest)])
        except Exception:  # noqa: BLE001 - the object is safe; only the copy is not
            self._logger.warning(
                "could not cache %s locally", reference.content_digest, exc_info=True
            )
        return reference

    def hydrate(self, task_id: str, reference: ContentReference) -> bytes:
        """The object's verified bytes, from the nearest place that has them."""
        if self._lane is not None:
            try:
                return self._lane.hydrate(reference, task_id)
            except Exception as miss:  # noqa: BLE001 - see below: every cache path is
                # optional, and that includes a cache that cannot represent the object,
                # a content directory that has gone unreadable, and a lane that is not
                # running. The object is in the shared store whatever happened to a copy
                # of it, so anything the cache raises costs this read and nothing else.
                self._logger.debug(
                    "reading %s from the shared store: %s",
                    reference.content_digest,
                    miss,
                )
        data = self._access.store_for(task_id, reference.authorization_scope).hydrate(
            reference
        )
        self._logger.info(
            "content %s read from the shared store", reference.content_digest
        )
        return data


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
