"""The content this worker keeps a copy of.

A cache over the shared durable store, not a holder of record: every object here is also
there, so a copy may be dropped whenever it stops paying for itself and nothing is lost
when a worker dies with copies on its disk. What the cache buys is the read — a task
that needs an object this worker already has never leaves the node, and a task on
another worker can be served from here instead of the shared store.

Copies are kept on disk rather than in memory so they survive the worker process, and
the worker tells control what it holds so a peer's read can be pointed at it. A copy
ages out once it is older than the retention window; nothing else removes one.
"""

import time
from collections.abc import Iterator
from pathlib import Path

from shared.content import (
    OCTET_STREAM,
    ContentReference,
    FabricObjectStore,
    FilesystemObjectBacking,
)


class WorkerContentCache(FabricObjectStore):
    """The copies this worker holds, under a local directory."""

    def __init__(self, root: Path, *, retain_sec: float) -> None:
        self._objects = FilesystemObjectBacking(root)
        self._retain_sec = retain_sec

    @property
    def retain_sec(self) -> float:
        return self._retain_sec

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._objects.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._objects.read(
            reference.authorization_scope, reference.content_digest
        )

    def holds(self, reference: ContentReference) -> bool:
        return self._objects.holds(
            reference.authorization_scope, reference.content_digest
        )

    def iter_held(self) -> Iterator[tuple[str, str]]:
        """Every copy this worker holds, as the scope and digest naming it."""
        return self._objects.iter_objects()

    def evict_aged(self, *, in_transfer: frozenset[str] = frozenset()) -> int:
        """Drop copies past the retention window, and return how many went.

        A copy being served right now stays until its transfer is done; everything else
        is free to go, because the object itself is in the shared store either way.
        """
        deadline = time.time() - self._retain_sec
        evicted = 0
        for scope, digest in list(self._objects.iter_objects()):
            if digest in in_transfer:
                continue
            cached_at = self._objects.written_at(scope, digest)
            if cached_at is None or cached_at > deadline:
                continue
            self._objects.remove(scope, digest)
            evicted += 1
        return evicted
