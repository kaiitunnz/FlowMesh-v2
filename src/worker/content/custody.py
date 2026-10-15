"""Custody of artifact bytes a success announced before the shared store had them.

A task reports success once its artifacts are copied and hashed here; their upload to
the shared store follows. Until control settles each upload, the bytes are held in a
directory per dispatch beside the scope's objects, where no eviction, staging cleanup or
output cleanup reaches: each staged object is the file its bytes were hashed into, so
custody holds exactly what the success named. Where the worker runs a cache, the cache's
name for the object is a second link to that file. The kernel's link count is what every
process sharing the content directory sees, so a cached copy also held in custody is
kept from eviction by any of them, and the custody file outlives any removal of the
cache's name.

A dispatch's directory also holds its journal: what the worker staged, under which
registration, and the success it reported, so the success and its uploads can be
reported again until control answers. The journal is written before any staged file,
so a directory without one holds nothing staged. Custody ends only when control settles
every task it covers; nothing here ages it out.
"""

import errno
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shared.content import (
    OCTET_STREAM,
    STREAM_CHUNK_BYTES,
    ContentReference,
    ContentStoreError,
    WriteCancelled,
    new_hasher,
)
from shared.content.filesystem import safe_segment
from shared.utils.atomic import temp_path, write_replacing

CUSTODY_DIR = "custody"
_JOURNAL = "journal.json"

type KeepCopy = Callable[[ContentReference, Path], None]


class StagedTask(BaseModel):
    """One task's staged artifacts, and how far control has taken them."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    members: list[ContentReference] = Field(default_factory=list)
    # Control recorded the success that announced these artifacts.
    accepted: bool = False
    # Every member is in the shared store and the receipt saying so was sent.
    uploaded: bool = False
    # Why the upload cannot finish, once a member is found unuploadable.
    failed: str | None = None
    # Control settled the upload: durable, lost or rejected.
    settled: bool = False


class UploadJournal(BaseModel):
    """What one dispatch staged, for the registration that staged it."""

    model_config = ConfigDict(extra="ignore")

    dispatch_id: str
    scope: str
    # The worker's address on its node, which outlives a registration; the content
    # directory is shared by every worker on the node.
    alias: str
    worker_id: str
    incarnation: int
    # The task the dispatch ran and the success it reported, re-sent until accepted.
    task_id: str
    success: dict[str, Any] = Field(default_factory=dict)
    tasks: dict[str, StagedTask] = Field(default_factory=dict)

    def owes(self) -> list[StagedTask]:
        """Return the tasks whose upload control has not settled."""
        return [task for task in self.tasks.values() if not task.settled]


class ArtifactCustody:
    """The staged bytes and journals under one content directory.

    ``keep`` names a staged file as the cache's copy of its object, where the worker
    runs a cache, so a peer can be served from it.
    """

    def __init__(
        self,
        root: Path,
        keep: KeepCopy | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._root = root
        self._keep = keep
        self._logger = logger or logging.getLogger("artifact-custody")

    def publication_dir(self, scope: str, dispatch_id: str) -> Path:
        return (
            self._root / safe_segment(scope) / CUSTODY_DIR / safe_segment(dispatch_id)
        )

    def source(self, journal: UploadJournal, reference: ContentReference) -> Path:
        """Return the custody file holding a staged object's bytes."""
        return self.publication_dir(journal.scope, journal.dispatch_id) / safe_segment(
            reference.content_digest
        )

    def stage(
        self,
        scope: str,
        dispatch_id: str,
        source: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Copy a file into a dispatch's custody, named by the digest of exactly the
        bytes copied, and return its reference.

        The bytes are copied rather than linked, so nothing the producer still holds
        open can change them after they are hashed.
        """
        directory = self.publication_dir(scope, dispatch_id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(directory)
        hasher = new_hasher()
        size = 0
        try:
            with source.open("rb") as src, tmp.open("xb") as dst:
                while chunk := src.read(STREAM_CHUNK_BYTES):
                    if cancelled is not None and cancelled():
                        raise WriteCancelled(f"staging of {source.name} was cancelled")
                    hasher.update(chunk)
                    dst.write(chunk)
                    size += len(chunk)
            reference = ContentReference(
                authorization_scope=scope,
                content_digest=hasher.hexdigest(),
                size_bytes=size,
                media_type=media_type,
            )
            held = directory / safe_segment(reference.content_digest)
            try:
                os.link(tmp, held)
            except FileExistsError:
                # The dispatch staged the same bytes under another name already.
                pass
            except OSError as exc:
                if exc.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP):
                    raise ContentStoreError(
                        "the content filesystem must support hard links to hold "
                        f"staged artifacts: {exc}"
                    ) from exc
                raise
        finally:
            tmp.unlink(missing_ok=True)
        if self._keep is not None:
            self._keep(reference, held)
        return reference

    def record(self, journal: UploadJournal) -> None:
        """Write a dispatch's journal durably, replacing the one there."""
        path = self.publication_dir(journal.scope, journal.dispatch_id) / _JOURNAL
        data = journal.model_dump_json().encode("utf-8")
        write_replacing(path, lambda fh: fh.write(data), durable=True)

    def journals(self) -> Iterator[UploadJournal]:
        """Yield every journal under this content directory."""
        if not self._root.exists():
            return
        for scope_dir in self._root.iterdir():
            custody = scope_dir / CUSTODY_DIR
            if not custody.is_dir():
                continue
            for publication in custody.iterdir():
                path = publication / _JOURNAL
                try:
                    yield UploadJournal.model_validate_json(path.read_bytes())
                except FileNotFoundError:
                    # Nothing is staged before the journal is written.
                    continue
                except (OSError, ValidationError):
                    self._logger.warning(
                        "Keeping unreadable artifact custody %s", publication
                    )

    def release(self, scope: str, dispatch_id: str) -> None:
        """End a dispatch's custody; a staged object the cache also names stays as
        its cached copy."""
        shutil.rmtree(self.publication_dir(scope, dispatch_id), ignore_errors=True)
