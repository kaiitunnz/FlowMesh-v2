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

Everything is taken per task, because that is how access is given: control grants one
task the right to reach one scope's content in the shared store, and the grant control
mints for a peer's copy rests on that task's own binding.
"""

import logging

from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
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
from .client import AnnounceHolding, GrantDenied
from .lane_host import ContentLaneHost


class WorkerContentPlane:
    """The worker's content surface, and the per-task stores taken from it."""

    def __init__(
        self,
        lane: ContentLaneHost,
        access: ContentAccessRegistry,
        *,
        announce: AnnounceHolding,
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
            _TaskScopedStores(self._access, task_id), self._finalizations
        )

    def accept_access(self, access: ContentStoreAccess) -> None:
        """Take the access control granted one of this worker's tasks."""
        self._access.accept(access)

    def release(self, task_id: str) -> None:
        """Drop what a finished task held; its copies stay in the cache."""
        self._access.release(task_id)

    def route(self, frame_kind: str, frame: dict[str, object]) -> None:
        """Hand one content control frame to the lane that consumes it."""
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
        try:
            return self._lane.hydrate(reference, task_id)
        except (GrantDenied, ContentHydrationError) as miss:
            # Every cache path is optional: the object is in the shared store whatever
            # happened to a copy of it, so a miss costs this read and nothing else.
            self._logger.debug(
                "reading %s from the shared store: %s", reference.content_digest, miss
            )
        data = self._access.store_for(task_id, reference.authorization_scope).hydrate(
            reference
        )
        self._logger.info(
            "content %s read from the shared store", reference.content_digest
        )
        return data


class _TaskScopedStores:
    """One task's access to the shared store, opened per scope on demand."""

    def __init__(self, access: ContentAccessRegistry, task_id: str) -> None:
        self._access = access
        self._task_id = task_id

    def for_scope(self, scope: str) -> FabricObjectStore:
        return self._access.store_for(self._task_id, scope)


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
