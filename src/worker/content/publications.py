"""Keeping what each publication names until control commits or rejects it.

A worker reports a success, a checkpoint or a prepared request by naming the objects it
stored. Control may refuse such a report before binding anything, when a named object
is being removed from the store or is no longer in it; the worker waits out a removal
and reports again, and stores an object that is gone again from the copy it kept, so no
task, provider call or input resolution runs again to repair it. Every report is
repeated on a pace until control answers it committed or rejected, since an event the
worker sends may be lost with the control process that received it. An object control
asks for that the worker holds no intact copy of cannot be repaired here, so the
dispatch that reported it is failed instead.

Each object a dispatch stores is kept under the dispatch, as another link to the file it
was written from or as the bytes written. A copy goes once no unanswered publication
names it and none may name it yet: once the publications naming it are answered for
good and the dispatch has not stored it again since, or once the dispatch has ended. A
publication names the objects its report references and the members of each tree whose
manifest it references. The outbox belongs to one registration: what an earlier one
reported is lost with its dispatches.
"""

import itertools
import logging
import os
import random
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from shared.content import ContentReference, StoreRefusal
from shared.content.reference import OCTET_STREAM
from shared.content.store import STREAM_CHUNK_BYTES, FabricObjectStore
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifestError, iter_members
from shared.tasks.publication import (
    Publication,
    PublicationAnswer,
    PublicationDisposition,
)
from shared.utils.nofollow import (
    UNLINKABLE_ERRNOS,
    link_opened,
    open_dir,
    remove_below,
)

# The directory, under a worker's results, holding what its publications name.
PUBLICATIONS_DIR = ".publications"
# How often an unanswered publication is reported again.
RESEND_SEC = 30.0
# How soon a publication the lane had no room for is offered again, before control's
# reservation for it lapses.
BUSY_SEC = 2.0
# The pace of reports control refused or could not decide.
_RETRY_MIN_SEC = 1.0
_RETRY_MAX_SEC = 30.0
_COPY_CHUNK = 1 << 20

type Send = Callable[[str, str, str, dict[str, Any]], None]
type Upload = Callable[[str, str, Path, str], ContentReference]
type Fail = Callable[[str, str, str], None]
"""Fail one task's dispatch, by task, dispatch and error."""


@dataclass
class _Entry:
    publication: Publication
    event_type: str
    task_id: str
    payload: dict[str, Any]
    due: float
    names: set[str]
    backoff: float = _RETRY_MIN_SEC
    # Objects control asked to be stored again before the next report.
    restore: tuple[str, ...] = ()


@dataclass
class _Kept:
    scope: str
    media_type: str
    size_bytes: int
    path: Path
    # When the dispatch last stored it, and when a report last named it, on one count.
    kept_at: int
    named_at: int = -1


@dataclass
class _Dispatch:
    entries: dict[tuple[str, str], _Entry] = field(default_factory=dict)
    kept: dict[str, _Kept] = field(default_factory=dict)
    ended: bool = False


class _Unrestorable(Exception):
    """No intact copy of an object control asked to be stored again is kept."""


class PublicationOutbox:
    """Owes control every publication this registration reported, until answered."""

    def __init__(
        self,
        root: Path,
        registration: Callable[[], str],
        send: Send,
        upload: Upload,
        fail: Fail,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._registration = registration
        self._send = send
        self._upload = upload
        self._fail = fail
        self._logger = logger or logging.getLogger("publications")
        self._clock = clock
        self._cv = threading.Condition()
        self._dispatches: dict[str, _Dispatch] = {}
        self._ticks = itertools.count()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Drop what an earlier registration kept, and start reporting."""
        remove_below(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="publications", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(max(0.0, timeout))

    def clear(self) -> None:
        """Forget every publication, as when the registration that reported them
        ends."""
        with self._cv:
            dispatches, self._dispatches = self._dispatches, {}
        for dispatch_id in dispatches:
            remove_below(self._root / dispatch_id)

    def owed(self) -> int:
        """Return how many publications are still unanswered."""
        with self._cv:
            return sum(len(d.entries) for d in self._dispatches.values())

    def dispatch_of(self, task_id: str) -> str | None:
        """Return the dispatch owing a publication of ``task_id``, whose store access
        a re-store is written under."""
        with self._cv:
            for dispatch_id, dispatch in self._dispatches.items():
                if any(e.task_id == task_id for e in dispatch.entries.values()):
                    return dispatch_id
        return None

    def keeping(self, dispatch_id: str, store: FabricObjectStore) -> FabricObjectStore:
        """Return ``store`` keeping a copy of every object written through it for the
        dispatch's publications, until the dispatch ends."""
        with self._cv:
            self._dispatches.setdefault(dispatch_id, _Dispatch())
        return _KeepingStore(
            store, lambda reference, source: self._keep(dispatch_id, reference, source)
        )

    def _keep(
        self,
        dispatch_id: str,
        reference: ContentReference,
        source: BinaryIO | bytes | Path,
    ) -> None:
        try:
            self.keep(dispatch_id, reference, source)
        except OSError as exc:
            self._logger.warning(
                "Could not keep a copy of %s for %s; it cannot be stored again: %s",
                reference.content_digest,
                dispatch_id,
                exc,
            )

    def keep(
        self,
        dispatch_id: str,
        reference: ContentReference,
        source: BinaryIO | bytes | Path,
    ) -> None:
        """Keep a copy of an object a running dispatch stored, from the file it was
        written from or its bytes, for as long as a publication may need it stored
        again; a dispatch that has ended keeps nothing more."""
        digest = reference.content_digest
        with self._cv:
            dispatch = self._dispatches.get(dispatch_id)
            if dispatch is None or dispatch.ended:
                return
            tick = next(self._ticks)
            if (kept := dispatch.kept.get(digest)) is not None:
                kept.kept_at = tick
                return
            path = self._root / dispatch_id / f"{digest}.{tick}"
            kept = dispatch.kept[digest] = _Kept(
                reference.authorization_scope,
                reference.media_type,
                reference.size_bytes,
                path,
                tick,
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, bytes):
                path.write_bytes(source)
            elif isinstance(source, Path):
                self._keep_path(path, source)
            else:
                self._keep_opened(path, source)
        except OSError:
            with self._cv:
                if dispatch.kept.get(digest) is kept:
                    del dispatch.kept[digest]
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _keep_path(path: Path, source: Path) -> None:
        try:
            os.link(source, path)
        except OSError as exc:
            if exc.errno not in UNLINKABLE_ERRNOS:
                raise
            shutil.copyfile(source, path)

    @staticmethod
    def _keep_opened(path: Path, source: BinaryIO) -> None:
        with open_dir(path.parent) as dir_fd:
            if link_opened(source.fileno(), dir_fd, path.name):
                return
        position = source.tell()
        source.seek(0)
        try:
            with open(path, "xb") as copy:
                shutil.copyfileobj(source, copy, _COPY_CHUNK)
        finally:
            source.seek(position)

    def reported(
        self, event_type: str, task_id: str, dispatch_id: str | None, payload: Any
    ) -> None:
        """Owe a report just sent until control answers it for good."""
        if dispatch_id is None or not isinstance(payload, dict):
            return
        publication = Publication.of(
            self._registration(), dispatch_id, task_id, event_type, payload
        )
        if publication is None:
            return
        named = set(_referenced(payload))
        key = (publication.kind.value, publication.key)
        with self._cv:
            dispatch = self._dispatches.setdefault(dispatch_id, _Dispatch())
            entry = _Entry(
                publication,
                event_type,
                task_id,
                payload,
                self._clock() + RESEND_SEC,
                named,
            )
            replaced = dispatch.entries.get(key)
            dispatch.entries[key] = entry
            self._name_locked(dispatch, named)
            trees = [
                kept.path
                for digest in named
                if (kept := dispatch.kept.get(digest)) is not None
                and kept.media_type == TREE_MEDIA_TYPE
            ]
            doomed = (
                self._release_locked(dispatch_id, dispatch, replaced.names)
                if replaced is not None
                else []
            )
            self._cv.notify_all()
        _remove(doomed)
        if not trees:
            return
        members = self._members(trees)
        with self._cv:
            if self._dispatches.get(dispatch_id) is not dispatch:
                return
            if dispatch.entries.get(key) is entry:
                entry.names |= members
            self._name_locked(dispatch, members)
            doomed = self._release_locked(dispatch_id, dispatch, members)
        _remove(doomed)

    def _members(self, manifests: list[Path]) -> set[str]:
        """Return the digests of every member the kept manifests name."""
        members: set[str] = set()
        for path in manifests:
            try:
                with path.open("rb") as stream:
                    chunks = iter(lambda: stream.read(STREAM_CHUNK_BYTES), b"")
                    members.update(member.digest for member in iter_members(chunks))
            except (OSError, TreeManifestError) as exc:
                self._logger.warning(
                    "Could not read the kept manifest %s; its members stay kept "
                    "until its dispatch ends: %s",
                    path.name,
                    exc,
                )
        return members

    def _name_locked(self, dispatch: _Dispatch, digests: Iterable[str]) -> None:
        tick = next(self._ticks)
        for digest in digests:
            if (kept := dispatch.kept.get(digest)) is not None:
                kept.named_at = tick

    def end(self, dispatch_id: str | None) -> None:
        """Note that a dispatch stored and reported everything it will; the copies no
        unanswered publication names go."""
        if dispatch_id is None:
            return
        with self._cv:
            if (dispatch := self._dispatches.get(dispatch_id)) is None:
                return
            dispatch.ended = True
            doomed = self._release_locked(dispatch_id, dispatch, list(dispatch.kept))
        _remove(doomed)

    def answer(self, answer: PublicationAnswer) -> None:
        """Take control's answer about one publication."""
        publication = answer.publication
        doomed: list[Path] = []
        with self._cv:
            dispatch = self._dispatches.get(publication.dispatch_id)
            entry = (
                dispatch.entries.get((publication.kind.value, publication.key))
                if dispatch is not None
                else None
            )
            if dispatch is None or entry is None or entry.publication != publication:
                return
            now = self._clock()
            match answer.disposition:
                case PublicationDisposition.COMMITTED | PublicationDisposition.REJECTED:
                    if answer.disposition is PublicationDisposition.REJECTED:
                        self._logger.info(
                            "Publication %s of %s was rejected: %s",
                            publication.key,
                            publication.dispatch_id,
                            answer.reason,
                        )
                    del dispatch.entries[(publication.kind.value, publication.key)]
                    doomed = self._release_locked(
                        publication.dispatch_id, dispatch, entry.names
                    )
                case PublicationDisposition.REFUSED:
                    match answer.refusal:
                        case StoreRefusal.BUSY:
                            entry.due = now + BUSY_SEC * (
                                1 + random.random()  # nosec B311 - jitter only
                            )
                        case StoreRefusal.REMATERIALIZE:
                            entry.restore = tuple(answer.digests)
                            entry.due = now + self._backoff(entry)
                        case _:
                            entry.due = now + self._backoff(entry)
            self._cv.notify_all()
        _remove(doomed)

    @staticmethod
    def _backoff(entry: _Entry) -> float:
        wait = entry.backoff
        entry.backoff = min(_RETRY_MAX_SEC, wait * 2)
        return wait

    def _release_locked(
        self, dispatch_id: str, dispatch: _Dispatch, digests: Iterable[str]
    ) -> list[Path]:
        """Stop keeping each of ``digests`` no unanswered publication names and none
        may name yet; returns what to remove once the lock is released."""
        if dispatch.ended and not dispatch.entries:
            del self._dispatches[dispatch_id]
            return [self._root / dispatch_id]
        doomed: list[Path] = []
        for digest in digests:
            kept = dispatch.kept.get(digest)
            if kept is None or (not dispatch.ended and kept.kept_at > kept.named_at):
                continue
            if any(digest in entry.names for entry in dispatch.entries.values()):
                continue
            del dispatch.kept[digest]
            doomed.append(kept.path)
        return doomed

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stopping:
                    return
                due = self._due_locked()
                if not due:
                    self._cv.wait(self._wait_locked())
                    continue
            for dispatch_id, entry in due:
                self._report(dispatch_id, entry)

    def _due_locked(self) -> list[tuple[str, _Entry]]:
        now = self._clock()
        due: list[tuple[str, _Entry]] = []
        for dispatch_id, dispatch in self._dispatches.items():
            for entry in dispatch.entries.values():
                if entry.due <= now:
                    entry.due = now + RESEND_SEC
                    due.append((dispatch_id, entry))
        return due

    def _wait_locked(self) -> float:
        now = self._clock()
        dues = [
            entry.due - now
            for dispatch in self._dispatches.values()
            for entry in dispatch.entries.values()
        ]
        return max(0.05, min(dues, default=RESEND_SEC))

    def _report(self, dispatch_id: str, entry: _Entry) -> None:
        if entry.restore and not self._restore(dispatch_id, entry):
            return
        try:
            self._send(entry.event_type, entry.task_id, dispatch_id, entry.payload)
        except Exception:
            self._logger.debug("could not report a publication again", exc_info=True)

    def _restore(self, dispatch_id: str, entry: _Entry) -> bool:
        """Store again each object control asked for, from the copy kept of it; a
        store that cannot take it now is retried, and an object no intact copy is kept
        of fails the dispatch."""
        with self._cv:
            dispatch = self._dispatches.get(dispatch_id)
            kept = dict(dispatch.kept) if dispatch is not None else {}
        try:
            for digest in entry.restore:
                self._restore_one(entry, kept.get(digest), digest)
        except _Unrestorable as exc:
            self._logger.error(
                "Publication %s of %s cannot be repaired, so its dispatch fails: %s",
                entry.publication.key,
                dispatch_id,
                exc,
            )
            try:
                self._fail(
                    entry.task_id, dispatch_id, f"publication_unrepairable: {exc}"
                )
            except Exception:
                self._logger.debug(
                    "could not report the dispatch failed", exc_info=True
                )
        except Exception as exc:
            self._logger.warning(
                "Storing an object of %s again failed; retrying: %s", dispatch_id, exc
            )
            with self._cv:
                entry.due = self._clock() + self._backoff(entry)
            return False
        else:
            self._logger.info(
                "Stored %d object(s) of publication %s again",
                len(entry.restore),
                entry.publication.key,
            )
        with self._cv:
            entry.restore = ()
        return True

    def _restore_one(self, entry: _Entry, kept: _Kept | None, digest: str) -> None:
        if kept is None:
            raise _Unrestorable(f"{digest} was not kept")
        try:
            size = kept.path.stat().st_size
        except FileNotFoundError:
            raise _Unrestorable(f"the copy kept of {digest} is gone") from None
        if size != kept.size_bytes:
            raise _Unrestorable(f"the copy kept of {digest} has {size} bytes")
        stored = self._upload(entry.task_id, kept.scope, kept.path, kept.media_type)
        if stored.content_digest != digest:
            raise _Unrestorable(f"the copy kept of {digest} no longer holds its bytes")


def _referenced(payload: Any) -> Iterator[str]:
    """Yield the digest of every content reference in a report's payload."""
    match payload:
        case {"content_digest": str() as digest}:
            yield digest
            for value in payload.values():
                yield from _referenced(value)
        case dict():
            for value in payload.values():
                yield from _referenced(value)
        case list():
            for value in payload:
                yield from _referenced(value)


def _remove(paths: list[Path]) -> None:
    for path in paths:
        remove_below(path)


class _KeepingStore(FabricObjectStore):
    """A store that hands each object written through it to ``keep``."""

    def __init__(
        self,
        inner: FabricObjectStore,
        keep: Callable[[ContentReference, BinaryIO | bytes | Path], None],
    ) -> None:
        self._inner = inner
        self._keep = keep

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = self._inner.write(scope, data, media_type=media_type)
        self._keep(reference, data)
        return reference

    def fetch(self, reference: ContentReference) -> bytes:
        return self._inner.fetch(reference)

    def hydrate(self, reference: ContentReference) -> bytes:
        return self._inner.hydrate(reference)

    def stored_size(self, reference: ContentReference) -> int:
        return self._inner.stored_size(reference)

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        reference = self._inner.write_file(
            scope, path, media_type=media_type, cancelled=cancelled
        )
        self._keep(reference, path)
        return reference

    def write_completed(
        self,
        scope: str,
        source: BinaryIO,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        reference = self._inner.write_completed(
            scope, source, media_type=media_type, cancelled=cancelled
        )
        self._keep(reference, source)
        return reference

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        return self._inner.open(reference, chunk_bytes=chunk_bytes)

    def hydrate_to(self, reference: ContentReference, dest: Path) -> None:
        self._inner.hydrate_to(reference, dest)

    def open_range(
        self,
        reference: ContentReference,
        start: int,
        length: int,
        *,
        chunk_bytes: int = STREAM_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        return self._inner.open_range(reference, start, length, chunk_bytes=chunk_bytes)


__all__ = ["PUBLICATIONS_DIR", "PublicationOutbox"]
