"""The content this worker keeps a copy of.

A cache over the shared durable store, not a holder of record: every object here is also
there, so a copy may be dropped whenever it stops paying for itself and nothing is lost
when a worker dies with copies on its disk. What the cache buys is the read — a task
that needs an object this worker already has never leaves the node, and a task on
another worker can be served from here instead of the shared store.

Copies are kept on disk rather than in memory so they survive the worker process, and
the worker tells control what it holds so a peer's read can be pointed at it. Two bounds
keep the cache in check, each off unless configured: a copy unused for longer than the
retention window ages out, and past the disk budget the least recently used copies go
first. Reading a copy — for a task here or to serve a peer — counts as using it.
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

    def __init__(
        self, root: Path, *, retain_sec: float = 0.0, max_bytes: int = 0
    ) -> None:
        self._objects = FilesystemObjectBacking(root)
        self._retain_sec = retain_sec
        self._max_bytes = max_bytes

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = self._objects.write(scope, data, media_type=media_type)
        self._objects.touch(scope, reference.content_digest)
        return reference

    def fetch(self, reference: ContentReference) -> bytes:
        data = self._objects.read(
            reference.authorization_scope, reference.content_digest
        )
        self._objects.touch(reference.authorization_scope, reference.content_digest)
        return data

    def holds(self, reference: ContentReference) -> bool:
        return self._objects.holds(
            reference.authorization_scope, reference.content_digest
        )

    def iter_held(self) -> Iterator[tuple[str, str]]:
        """Every copy this worker holds, as the scope and digest naming it."""
        return self._objects.iter_objects()

    def evict(self, *, in_transfer: frozenset[str] = frozenset()) -> int:
        """Bring the cache within its bounds, and return how many copies went.

        Copies unused past the retention window go first, then the least recently used
        until what is left fits the disk budget. A copy being served right now stays
        until its transfer is done; everything else is free to go, because the object
        itself is in the shared store either way.
        """
        if self._retain_sec <= 0 and self._max_bytes <= 0:
            return 0
        held: list[tuple[float, int, str, str]] = []
        for scope, digest in list(self._objects.iter_objects()):
            if (stat := self._objects.stat(scope, digest)) is not None:
                held.append((stat.st_mtime, stat.st_size, scope, digest))
        held.sort()
        evicted = 0
        kept: list[tuple[float, int, str, str]] = []
        deadline = time.time() - self._retain_sec
        for entry in held:
            used_at, _, scope, digest = entry
            if (
                self._retain_sec > 0
                and used_at <= deadline
                and digest not in in_transfer
            ):
                self._objects.remove(scope, digest)
                evicted += 1
            else:
                kept.append(entry)
        if self._max_bytes > 0:
            total = sum(size for _, size, _, _ in kept)
            for _, size, scope, digest in kept:
                if total <= self._max_bytes:
                    break
                if digest in in_transfer:
                    continue
                self._objects.remove(scope, digest)
                total -= size
                evicted += 1
        return evicted
