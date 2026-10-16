"""The published artifacts a task's spec names, put on this worker's disk.

A dispatch names each artifact its task loads by a token in the spec, or carries the
publications of the producers whose artifact refs the spec's data reads; the worker
hydrates every such artifact under the task's output and puts its local path in the
spec before anything reads it. A tree lands as a directory of its files, each verified
against its manifest's digest, under a name keyed by the tree's own digest, so two
tokens naming one tree share it. The directory appears only once whole and goes when
the task ends: where the worker keeps a cache, each file links to the cached copy, so
what stays on disk is the cache's, under its bounds.
"""

import os
import shutil
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Self

from shared.content import (
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    TreeManifest,
    TreeManifestError,
)
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.utils.nofollow import private_dir

from ..executors.base_executor import TaskCancelledError
from .access import ContentAccessDenied
from .plane import WorkerContentPlane

ARTIFACT_INPUTS_DIR = ".artifact-inputs"
SSH_INPUTS_DIR = ".ssh-inputs"
# A read that cannot reach the store retries briefly before the task reports it.
_READ_ATTEMPTS = 3
_READ_BACKOFF_SEC = 0.2


class ArtifactInputUnreadable(RuntimeError):
    """An artifact input's tree manifest is not one."""


class ArtifactInputUnavailable(RuntimeError):
    """An object an artifact input names could not be read from the store."""

    def __init__(self, reference: ContentReference, cause: Exception) -> None:
        super().__init__(f"{reference.content_digest}: {cause}")
        self.reference = reference


def artifact_inputs_dir(out_dir: Path) -> Path:
    """Return where a task's artifact inputs land, beside its own output."""
    return out_dir / ARTIFACT_INPUTS_DIR


def ssh_inputs_dir(out_dir: Path) -> Path:
    """Return where an SSH task's upstream artifacts are staged, beside its own
    output."""
    return out_dir / SSH_INPUTS_DIR


class ArtifactMaterializer:
    """Puts one task's artifact inputs on disk through the content plane.

    Used as a context: ``root`` lies in the task's world-writable output, so it is
    held open as a directory only this worker may write while the inputs land, and
    nothing below it is reached through a link put in its place.
    """

    def __init__(
        self,
        plane: WorkerContentPlane,
        task_id: str,
        root: Path,
        cancelled: Callable[[], bool] = lambda: False,
        backoff_sec: float = _READ_BACKOFF_SEC,
        pending: frozenset[str] = frozenset(),
    ) -> None:
        self._plane = plane
        self._pending = pending
        self._task_id = task_id
        self._root = root
        self._cancelled = cancelled
        self._backoff_sec = backoff_sec
        self._held = ExitStack()
        self._dir = root

    def __enter__(self) -> Self:
        self._dir = self._held.enter_context(private_dir(self._root))
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._held.close()
        self._dir = self._root

    def materialize(self, binding: ArtifactBinding) -> Path:
        """Hydrate an artifact under the root once, returning its local path."""
        digest = binding.reference.content_digest
        pending = digest in self._pending
        if binding.kind is not ArtifactKind.TREE:
            rel = PurePosixPath(digest, PurePosixPath(binding.name).name)
            if not (self._dir / rel).exists():
                self._fetch(binding.reference, self._dir / rel, pending)
            return self._root / rel
        if (self._dir / digest).is_dir():
            return self._root / digest
        members = self._members(binding)
        staging = self._dir / f".{digest}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for reference, path in members:
            self._fetch(reference, staging / path, pending)
        staging.rename(self._dir / digest)
        return self._root / digest

    def copy_at(self, binding: ArtifactBinding, rel: PurePosixPath) -> Path:
        """Put a writable copy of the artifact at ``rel`` under the root, laid out as
        it was published: a file as ``rel``, a tree as a directory of its files."""
        pending = binding.reference.content_digest in self._pending
        if binding.kind is not ArtifactKind.TREE:
            self._copy(binding.reference, self._dir / rel, pending)
            return self._root / rel
        members = self._members(binding)
        (self._dir / rel).mkdir(parents=True, exist_ok=True)
        for reference, path in members:
            self._copy(reference, self._dir / rel / path, pending)
        return self._root / rel

    def _members(self, binding: ArtifactBinding) -> list[tuple[ContentReference, str]]:
        """Return each member of the tree ``binding`` names, by reference and path."""
        try:
            manifest = TreeManifest.decode(self._manifest_bytes(binding.reference))
        except TreeManifestError as exc:
            raise ArtifactInputUnreadable(
                f"artifact {binding.name} names no tree: {exc}"
            ) from exc
        scope = binding.reference.authorization_scope
        return [(member.reference(scope), member.path) for member in manifest.members]

    def _copy(self, reference: ContentReference, dest: Path, pending: bool) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read(
            reference,
            lambda: self._plane.hydrate_to(
                self._task_id, reference, dest, prefer_peer=pending
            ),
            pending,
        )

    def _manifest_bytes(self, reference: ContentReference) -> bytes:
        return self._read(
            reference, lambda: self._plane.hydrate(self._task_id, reference)
        )

    def _read[T](
        self, reference: ContentReference, read: Callable[[], T], pending: bool = False
    ) -> T:
        """Read an object, retrying briefly while the store cannot be reached.

        An object of an artifact whose upload is still owed that no holder served and
        the store does not have yet is unavailable, not missing.
        """
        if self._cancelled():
            raise TaskCancelledError(
                f"Task {self._task_id} ended while reading its artifact inputs"
            )
        for attempt in range(1, _READ_ATTEMPTS + 1):
            try:
                return read()
            except (ContentUnavailable, ContentAccessDenied) as exc:
                if attempt == _READ_ATTEMPTS:
                    raise ArtifactInputUnavailable(reference, exc) from exc
            except ContentHydrationError as exc:
                if pending:
                    raise ArtifactInputUnavailable(reference, exc) from exc
                raise
            time.sleep(self._backoff_sec)
        raise AssertionError("unreachable")

    def _fetch(self, reference: ContentReference, dest: Path, pending: bool) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read(
            reference,
            lambda: self._plane.materialize_to(
                self._task_id, reference, dest, prefer_peer=pending
            ),
            pending,
        )
        os.chmod(dest, 0o444)


__all__ = [
    "ARTIFACT_INPUTS_DIR",
    "SSH_INPUTS_DIR",
    "ssh_inputs_dir",
    "ArtifactInputUnavailable",
    "ArtifactInputUnreadable",
    "ArtifactMaterializer",
    "artifact_inputs_dir",
]
