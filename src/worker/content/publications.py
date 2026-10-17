"""Keeping what each publication names until control commits or rejects it.

A worker reports a success, a checkpoint or a prepared request by naming the objects it
stored. Control may refuse such a report before binding anything, when a named object
is being removed from the store or is no longer in it; the worker then stores the
object again from the copy it kept and reports again, so no task, provider call or
input resolution runs again to repair it. Every report is repeated on a pace until
control answers it committed or rejected, since an event the worker sends may be lost
with the control process that received it.

Each object a dispatch stores is kept under the dispatch, as another link to the file it
was written from or as the bytes written, and the copies go once every publication of
the dispatch is answered for good and the dispatch has ended. The outbox belongs to one
registration: what an earlier one reported is lost with its dispatches.
"""

import logging
import os
import random
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from shared.content import ContentReference
from shared.content.reference import OCTET_STREAM
from shared.content.store import STREAM_CHUNK_BYTES, FabricObjectStore
from shared.tasks.publication import (
    BUSY_REASON,
    CHECKPOINT_FIELD,
    PREPARED_FIELD,
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
# The pace of reports control could not decide, and of waiting out a removal.
_RETRY_MIN_SEC = 1.0
_RETRY_MAX_SEC = 30.0
_COPY_CHUNK = 1 << 20

type Send = Callable[[str, str, str, dict[str, Any]], None]
type Upload = Callable[[str, str, Path, str], ContentReference]


@dataclass
class _Entry:
    publication: Publication
    event_type: str
    task_id: str
    payload: dict[str, Any]
    named: set[str]
    due: float
    backoff: float = _RETRY_MIN_SEC
    # Objects control asked to be stored again before the next report.
    restore: tuple[str, ...] = ()


@dataclass
class _Dispatch:
    entries: dict[tuple[str, str], _Entry] = field(default_factory=dict)
    kept: dict[str, dict[str, Any]] = field(default_factory=dict)
    ended: bool = False


def named_digests(event_type: str, payload: dict[str, Any]) -> set[str]:
    """Return the digests of the objects a report names."""
    named: set[str] = set()

    def add(raw: Any) -> None:
        if isinstance(raw, dict) and isinstance(raw.get("content_digest"), str):
            named.add(raw["content_digest"])

    def add_bindings(raw: Any) -> None:
        for item in raw if isinstance(raw, list) else ():
            if isinstance(item, dict):
                add(item.get("reference"))

    if event_type == "TASK_UPDATE":
        add_bindings(payload.get(CHECKPOINT_FIELD))
        return named
    if isinstance(prepared := payload.get(PREPARED_FIELD), dict):
        add(prepared.get("reference"))
        return named
    add(payload.get("result_reference"))
    children = payload.get("child_result_references")
    for raw in children.values() if isinstance(children, dict) else ():
        add(raw)
    add_bindings(payload.get("artifacts"))
    child_artifacts = payload.get("child_artifacts")
    for raw in child_artifacts.values() if isinstance(child_artifacts, dict) else ():
        add_bindings(raw)
    return named


class PublicationOutbox:
    """Owes control every publication this registration reported, until answered."""

    def __init__(
        self,
        root: Path,
        registration: Callable[[], str],
        send: Send,
        upload: Upload,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._registration = registration
        self._send = send
        self._upload = upload
        self._logger = logger or logging.getLogger("publications")
        self._clock = clock
        self._cv = threading.Condition()
        self._dispatches: dict[str, _Dispatch] = {}
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
        dispatch's publications."""
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
        """Keep a copy of an object a dispatch stored, from the file it was written
        from or its bytes, for as long as a publication may need it stored again."""
        digest = reference.content_digest
        with self._cv:
            dispatch = self._dispatches.setdefault(dispatch_id, _Dispatch())
            if digest in dispatch.kept:
                return
            dispatch.kept[digest] = {
                "scope": reference.authorization_scope,
                "media_type": reference.media_type,
                "size_bytes": reference.size_bytes,
            }
        directory = self._root / dispatch_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if isinstance(source, bytes):
                (directory / digest).write_bytes(source)
            elif isinstance(source, Path):
                self._keep_path(directory, digest, source)
            else:
                self._keep_opened(directory, digest, source)
        except FileExistsError:
            return
        except OSError:
            with self._cv:
                dispatch.kept.pop(digest, None)
            raise

    @staticmethod
    def _keep_path(directory: Path, digest: str, source: Path) -> None:
        try:
            os.link(source, directory / digest)
        except OSError as exc:
            if exc.errno not in UNLINKABLE_ERRNOS:
                raise
            shutil.copyfile(source, directory / digest)

    @staticmethod
    def _keep_opened(directory: Path, digest: str, source: BinaryIO) -> None:
        with open_dir(directory) as dir_fd:
            if link_opened(source.fileno(), dir_fd, digest):
                return
        position = source.tell()
        source.seek(0)
        try:
            with open(directory / digest, "xb") as copy:
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
        with self._cv:
            dispatch = self._dispatches.setdefault(dispatch_id, _Dispatch())
            dispatch.entries[(publication.kind.value, publication.key)] = _Entry(
                publication,
                event_type,
                task_id,
                payload,
                named_digests(event_type, payload),
                self._clock() + RESEND_SEC,
            )
            self._cv.notify_all()

    def end(self, dispatch_id: str | None) -> None:
        """Note that a dispatch stored and reported everything it will; its copies go
        once its publications are answered."""
        if dispatch_id is None:
            return
        with self._cv:
            if (dispatch := self._dispatches.get(dispatch_id)) is None:
                return
            dispatch.ended = True
            self._prune_locked(dispatch_id, dispatch)

    def answer(self, answer: PublicationAnswer) -> None:
        """Take control's answer about one publication."""
        publication = answer.publication
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
                    self._prune_locked(publication.dispatch_id, dispatch)
                case PublicationDisposition.RETRY:
                    entry.due = now + (
                        BUSY_SEC * (1 + random.random())  # nosec B311 - jitter only
                        if answer.reason == BUSY_REASON
                        else self._backoff(entry)
                    )
                case PublicationDisposition.REMATERIALIZE:
                    entry.restore = tuple(answer.digests)
                    entry.due = now
                case PublicationDisposition.RETRY_RETIREMENT:
                    entry.restore = tuple(answer.digests)
                    entry.due = now + self._backoff(entry)
            self._cv.notify_all()

    @staticmethod
    def _backoff(entry: _Entry) -> float:
        wait = entry.backoff
        entry.backoff = min(_RETRY_MAX_SEC, wait * 2)
        return wait

    def _prune_locked(self, dispatch_id: str, dispatch: _Dispatch) -> None:
        if dispatch.entries or not dispatch.ended:
            return
        del self._dispatches[dispatch_id]
        remove_below(self._root / dispatch_id)

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
        """Store again each object control asked for, from the copy kept of it."""
        with self._cv:
            dispatch = self._dispatches.get(dispatch_id)
            kept = dict(dispatch.kept) if dispatch is not None else {}
        for digest in entry.restore:
            facts = kept.get(digest)
            if facts is None or digest not in entry.named:
                self._logger.error(
                    "Publication %s of %s names %s, which this worker did not keep",
                    entry.publication.key,
                    dispatch_id,
                    digest,
                )
                continue
            try:
                stored = self._upload(
                    entry.task_id,
                    facts["scope"],
                    self._root / dispatch_id / digest,
                    facts["media_type"],
                )
            except Exception as exc:
                self._logger.warning(
                    "Storing %s again for %s failed; retrying: %s",
                    digest,
                    dispatch_id,
                    exc,
                )
                with self._cv:
                    entry.due = self._clock() + self._backoff(entry)
                return False
            if stored.content_digest != digest:
                self._logger.error(
                    "The copy kept of %s for %s no longer holds its bytes",
                    digest,
                    dispatch_id,
                )
        self._logger.info(
            "Stored %d object(s) of publication %s again",
            len(entry.restore),
            entry.publication.key,
        )
        with self._cv:
            entry.restore = ()
        return True


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


__all__ = ["PUBLICATIONS_DIR", "PublicationOutbox", "named_digests"]
