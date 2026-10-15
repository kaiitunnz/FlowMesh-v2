"""Custody of artifact bytes a success announced before the shared store had them.

A task reports success once its artifacts are copied and hashed here; their upload to
the shared store follows. Until control settles each upload, this worker is the only
place the bytes exist, so they are held apart from the cache's copies: each staged
object is a second link to the cache's file, in a directory per dispatch beside the
scope's objects, where no eviction, staging cleanup or output cleanup reaches. The
kernel's link count is what every process sharing the content directory sees, so a
copy in custody is kept from eviction by any of them, and its bytes outlive any removal
of the cache's name for it.

A dispatch's directory also holds its journal: what the worker staged, under which
registration, and the success it reported, so the success and its uploads can be
reported again until control answers. Custody ends only when control settles every
task it covers; nothing here ages it out.
"""

import contextlib
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
    ContentReference,
    ContentStoreError,
    FilesystemObjectBacking,
)
from shared.content.filesystem import safe_segment
from shared.utils.atomic import write_replacing

CUSTODY_DIR = "custody"
_JOURNAL = "journal.json"

type StageWriter = Callable[..., ContentReference]


class StagedTask(BaseModel):
    """One task's staged artifacts, and how far control has taken them."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    members: list[ContentReference] = Field(default_factory=list)
    # Control recorded the success that announced these artifacts.
    accepted: bool = False
    # Every member is in the shared store and the receipt saying so was sent.
    uploaded: bool = False
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
        """The tasks whose upload control has not settled."""
        return [task for task in self.tasks.values() if not task.settled]

    def members(self) -> dict[str, ContentReference]:
        """Every staged object, once each, by digest."""
        return {
            member.content_digest: member
            for task in self.tasks.values()
            for member in task.members
        }


class ArtifactCustody:
    """The staged bytes and journals under one content directory.

    ``write_file`` copies a file into the content backing, hashing exactly the bytes
    copied; the cache's own writer is passed where the worker runs one, so the copy is
    also a cached copy it reports holding.
    """

    def __init__(
        self,
        root: Path,
        write_file: StageWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._root = root
        self._backing = FilesystemObjectBacking(root, repair=True)
        self._write_file = write_file or self._backing.write_file
        self._logger = logger or logging.getLogger("artifact-custody")

    def publication_dir(self, scope: str, dispatch_id: str) -> Path:
        return (
            self._root / safe_segment(scope) / CUSTODY_DIR / safe_segment(dispatch_id)
        )

    def source(self, journal: UploadJournal, reference: ContentReference) -> Path:
        """The custody file holding a staged object's bytes."""
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
        """Copy a file into custody for a dispatch, named by the digest of the bytes
        copied, and return its reference.

        The bytes are copied rather than linked, so nothing the producer still holds
        open can change them after they are hashed.
        """
        directory = self.publication_dir(scope, dispatch_id)
        directory.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            reference = self._write_file(
                scope, source, media_type=media_type, cancelled=cancelled
            )
            if self._retain(reference, directory):
                return reference
        raise ContentStoreError(
            f"could not take {source.name} into custody: its copy was removed as it "
            "was written"
        )

    def _retain(self, reference: ContentReference, directory: Path) -> bool:
        """Link the backing's copy into custody; False when it was removed first."""
        digest = reference.content_digest
        held = directory / safe_segment(digest)
        copy = self._backing.object_path(reference.authorization_scope, digest)
        try:
            os.link(copy, held)
        except FileNotFoundError:
            return False
        except FileExistsError:
            pass
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise ContentStoreError(
                    "the content filesystem must support hard links to hold staged "
                    f"artifacts: {exc}"
                ) from exc
            raise
        if held.stat().st_size != reference.size_bytes:
            raise ContentStoreError(
                f"the custody copy of {digest} is not {reference.size_bytes} bytes"
            )
        return True

    def restore_copy(self, journal: UploadJournal, reference: ContentReference) -> None:
        """Give the cache back its name for a staged object it no longer has, so a peer
        is served from it again."""
        copy = self._backing.object_path(
            reference.authorization_scope, reference.content_digest
        )
        copy.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError, FileNotFoundError):
            os.link(self.source(journal, reference), copy)

    def record(self, journal: UploadJournal) -> None:
        """Write a dispatch's journal, replacing the one there.

        A dispatch records its journal before staging anything, so custody without a
        journal is only ever a dispatch still being set up.
        """
        path = self.publication_dir(journal.scope, journal.dispatch_id) / _JOURNAL
        data = journal.model_dump_json().encode("utf-8")
        write_replacing(path, lambda fh: fh.write(data))

    def journals(self) -> Iterator[UploadJournal]:
        """Every journal under this content directory."""
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
                    # Another worker sharing this directory is staging it right now.
                    continue
                except (OSError, ValidationError):
                    self._logger.warning(
                        "Keeping unreadable artifact custody %s", publication
                    )

    def release(self, scope: str, dispatch_id: str) -> None:
        """End a dispatch's custody: its staged bytes stay only as cached copies."""
        shutil.rmtree(self.publication_dir(scope, dispatch_id), ignore_errors=True)
