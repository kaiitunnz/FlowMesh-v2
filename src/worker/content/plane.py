"""This worker's view of fabric content.

Three things sit behind one object surface. What the worker wrote, it holds and reads
locally. What another worker wrote, it hydrates over an authorized transfer once control
grants one. What predates the transfer protocol — an object no worker ever reported
holding, which control answers with exactly that — it reads from the compatibility store
the deployment still runs, so migrating consumers do not lose objects written before.

Every read is scoped to the task asking for it: the grant control mints rests on that
task's own binding, so the surface is taken per task rather than shared across them.
"""

import logging
from collections.abc import Callable

from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    FabricObjectStore,
)

from .client import GrantDenied
from .lane_host import ContentLaneHost

# Control's word that nothing ever reported holding the object, which is what makes the
# compatibility store the right place to look rather than a failure to report.
_NOT_TRACKED = "not_tracked"

# Reports one held object to the control plane's holder directory.
AnnounceHolding = Callable[[ContentReference], None]


class WorkerContentPlane:
    """The worker's content surface, and the per-task stores taken from it."""

    def __init__(
        self,
        lane: ContentLaneHost,
        *,
        announce: "AnnounceHolding",
        compat: FabricObjectStore | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._lane = lane
        self._announce = announce
        self._compat = compat
        self._logger = logger or logging.getLogger("content-plane")

    def for_task(self, task_id: str) -> "TaskContentStore":
        """The object surface one task reads and writes through."""
        return TaskContentStore(self, task_id)

    def route(self, frame_kind: str, frame: dict[str, object]) -> None:
        """Hand one content control frame to the lane that consumes it."""
        self._lane.route(frame_kind, frame)

    def bind(self, reference: ContentReference) -> None:
        """Mark an object as named by a binding this worker has now reported."""
        self._lane.store.bind(reference)

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        """Hold the bytes here and tell control where they are.

        The report is location evidence, not a binding: it lets control resolve a
        holder for a later grant, and keeps nothing alive by itself.
        """
        reference = self._lane.store.write(scope, data, media_type=media_type)
        self._announce(reference)
        return reference

    def hydrate(self, reference: ContentReference, task_id: str) -> bytes:
        try:
            return self._lane.hydrate(reference, task_id)
        except GrantDenied as denial:
            if _NOT_TRACKED not in str(denial) or self._compat is None:
                raise
        return self._compat.hydrate(reference)


class TaskContentStore(FabricObjectStore):
    """One task's object surface over the worker's content plane."""

    def __init__(self, plane: WorkerContentPlane, task_id: str) -> None:
        self._plane = plane
        self._task_id = task_id

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._plane.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        raise ContentHydrationError(
            "content is fetched by hydrating a reference, not by an unverified read"
        )

    def hydrate(self, reference: ContentReference) -> bytes:
        return self._plane.hydrate(reference, self._task_id)
