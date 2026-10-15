"""The content this worker keeps a copy of.

A cache over the shared durable store, not a holder of record: every object here is also
there, so a copy may be dropped whenever it stops paying for itself and nothing is lost
when a worker dies with copies on its disk. What the cache buys is the read — a task
that needs an object this worker already has never leaves the node, and a task on
another worker can be served from here instead of the shared store.

Copies are kept on disk rather than in memory so they survive the worker process, and
the cache reports each copy it keeps or drops so a peer's read can be pointed at it.
Two bounds keep the cache in check, each off unless configured: a copy unused for longer
than the retention window ages out, and past the disk budget the least recently used
copies go first. Reading a copy — for a task here or to serve a peer — counts as using
it. A copy found not to be the object it is named for is dropped, since the store still
has the object.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

from shared.content import (
    OCTET_STREAM,
    STREAM_CHUNK_BYTES,
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
    FilesystemObjectBacking,
    verify_content,
)

from .holding import AnnounceHolding, HoldingReporter

_STALE_STAGING_SEC = 3600.0


class WorkerContentCache(FabricObjectStore):
    """The copies this worker holds, under a local directory."""

    def __init__(
        self,
        root: Path,
        *,
        retain_sec: float = 0.0,
        max_bytes: int = 0,
        announce: AnnounceHolding | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._objects = FilesystemObjectBacking(root, repair=True)
        self._objects.clear_staging(_STALE_STAGING_SEC)
        self._retain_sec = retain_sec
        self._max_bytes = max_bytes
        # Held while a copy is removed, or a kept one is noted, so the notes follow
        # the disk.
        self._lock = threading.RLock()
        self._holding = (
            HoldingReporter(announce, self._lock, logger=logger)
            if announce is not None
            else None
        )

    @property
    def holding_seq(self) -> int | None:
        """The number of the last holding report, None without a reporter."""
        return self._holding.seq if self._holding is not None else None

    def start_reporting(self) -> None:
        """Send the holding changes on a cadence from here on."""
        if self._holding is not None:
            self._holding.start()

    def stop_reporting(self) -> None:
        """Send the holding changes still collecting, and stop sending."""
        if self._holding is not None:
            self._holding.stop()

    def report_whole(self, *, restart: bool = False) -> int:
        """Report every copy held, and return how many; with ``restart``, as the first
        report of a new registration."""
        with self._lock:
            held = list(self._objects.iter_objects())
            if self._holding is not None:
                self._holding.whole(held, restart=restart)
        return len(held)

    def flush_holding(self) -> None:
        """Send the holding changes collecting now."""
        if self._holding is not None:
            self._holding.flush()

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = self._objects.write(scope, data, media_type=media_type)
        self._objects.touch(scope, reference.content_digest)
        self._kept(reference)
        return reference

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        reference = self._objects.write_file(
            scope, path, media_type=media_type, cancelled=cancelled
        )
        self._objects.touch(scope, reference.content_digest)
        self._kept(reference)
        return reference

    def staging_dir(self, scope: str) -> Path:
        """Where a copy on its way in is written before it is adopted."""
        return self._objects.staging_dir(scope)

    def adopt(self, reference: ContentReference, verified: Path) -> None:
        """Take a file already verified as the reference's bytes in as its copy."""
        self._objects.adopt(reference, verified)
        self._objects.touch(reference.authorization_scope, reference.content_digest)
        self._kept(reference)

    def fetch(self, reference: ContentReference) -> bytes:
        data = self._objects.read(
            reference.authorization_scope, reference.content_digest
        )
        self._objects.touch(reference.authorization_scope, reference.content_digest)
        return data

    def hydrate(self, reference: ContentReference) -> bytes:
        data = self.fetch(reference)
        try:
            return verify_content(reference, data)
        except ContentHydrationError:
            self.drop(reference)
            raise

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        stream = self._objects.stream(reference, chunk_bytes)
        self._objects.touch(reference.authorization_scope, reference.content_digest)
        return self._dropping_corrupt(reference, stream)

    def _dropping_corrupt(
        self, reference: ContentReference, stream: Iterator[bytes]
    ) -> Iterator[bytes]:
        try:
            yield from stream
        except ContentUnavailable:
            raise
        except ContentHydrationError:
            self.drop(reference)
            raise

    def open_pinned(self, reference: ContentReference) -> BinaryIO:
        """The copy opened for reading, readable to its end even if it is evicted."""
        fh = self._objects.open_object(
            reference.authorization_scope, reference.content_digest
        )
        self._objects.touch(reference.authorization_scope, reference.content_digest)
        return fh

    def link_to(self, reference: ContentReference, dest: Path) -> None:
        """Link this worker's copy at ``dest`` once it verifies as the reference's
        bytes, so a reader gets the copy without a second one on disk."""
        for _ in self.open(reference):
            pass
        source = self._objects.object_path(
            reference.authorization_scope, reference.content_digest
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, dest)

    def drop(self, reference: ContentReference) -> None:
        """Remove a copy."""
        with self._lock:
            self._remove(reference.authorization_scope, reference.content_digest)

    def holds(self, reference: ContentReference) -> bool:
        return self._objects.holds(
            reference.authorization_scope, reference.content_digest
        )

    def evict(self, *, in_transfer: frozenset[str] = frozenset()) -> int:
        """Bring the cache within its bounds, and return how many copies went.

        Copies unused past the retention window go first, then the least recently used
        until what is left fits the disk budget. A copy being served right now stays
        until its transfer is done; everything else is free to go, because the object
        itself is in the shared store either way.
        """
        if self._retain_sec <= 0 and self._max_bytes <= 0:
            return 0
        # The walk runs outside the lock, so a keep waits only on each removal; a copy
        # used since the walk read it is left for the next pass.
        held: list[tuple[float, int, str, str]] = []
        for scope, digest in list(self._objects.iter_objects()):
            if (stat := self._objects.stat(scope, digest)) is not None:
                held.append((stat.st_mtime, stat.st_size, scope, digest))
        held.sort()
        evicted = 0
        kept: list[tuple[float, int, str, str]] = []
        deadline = time.time() - self._retain_sec
        for entry in held:
            used_at, _, _, digest = entry
            if (
                self._retain_sec > 0
                and used_at <= deadline
                and digest not in in_transfer
            ):
                if self._remove_unused(entry):
                    evicted += 1
                    continue
                if self._gone(entry):
                    continue
            kept.append(entry)
        if self._max_bytes > 0:
            total = sum(size for _, size, _, _ in kept)
            for entry in kept:
                if total <= self._max_bytes:
                    break
                if entry[3] in in_transfer:
                    continue
                if self._remove_unused(entry):
                    evicted += 1
                    total -= entry[1]
                elif self._gone(entry):
                    # Another eviction, or a drop, freed it since the walk.
                    total -= entry[1]
        return evicted

    def _gone(self, entry: tuple[float, int, str, str]) -> bool:
        return self._objects.stat(entry[2], entry[3]) is None

    def _remove_unused(self, entry: tuple[float, int, str, str]) -> bool:
        """Remove a copy the walk found, unless it was used or replaced since."""
        used_at, size, scope, digest = entry
        with self._lock:
            stat = self._objects.stat(scope, digest)
            if stat is None or (stat.st_mtime, stat.st_size) != (used_at, size):
                return False
            self._remove(scope, digest)
        return True

    def _kept(self, reference: ContentReference) -> None:
        if self._holding is None:
            return
        with self._lock:
            # An eviction may have taken the copy since it was written; it noted that.
            if self.holds(reference):
                self._holding.kept(
                    (reference.authorization_scope, reference.content_digest)
                )

    def _remove(self, scope: str, digest: str) -> None:
        self._objects.remove(scope, digest)
        if self._holding is not None:
            self._holding.dropped((scope, digest))
