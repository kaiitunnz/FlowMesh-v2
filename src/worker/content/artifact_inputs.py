"""The published artifacts a task's spec names, put on this worker's disk.

A dispatch names each artifact its task loads by a token in the spec, or carries the
publications of the producers whose artifact refs the spec's data reads; the worker
hydrates every such artifact under the task's output and puts its local path in the
spec before anything reads it. A tree lands as a directory of its
files, each verified against the digest its manifest names, under a name keyed by the
tree's own digest, so two tokens naming one tree share it. The directory appears only
once whole, and it goes when the task ends: where the worker keeps a cache, each file is
a link to the cached copy, so what stays on disk is the cache's, under its bounds.
"""

import os
import shutil
from pathlib import Path, PurePosixPath

from shared.content import ContentReference, TreeManifest, TreeManifestError
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind

from .plane import WorkerContentPlane

ARTIFACT_INPUTS_DIR = ".artifact-inputs"


class ArtifactInputUnreadable(RuntimeError):
    """An artifact input's tree manifest is not one."""


def artifact_inputs_dir(out_dir: Path) -> Path:
    """Where a task's artifact inputs land, beside its own output."""
    return out_dir / ARTIFACT_INPUTS_DIR


class ArtifactMaterializer:
    """Puts one task's artifact inputs on disk through the content plane."""

    def __init__(self, plane: WorkerContentPlane, task_id: str, root: Path) -> None:
        self._plane = plane
        self._task_id = task_id
        self._root = root

    def materialize(self, binding: ArtifactBinding) -> Path:
        digest = binding.reference.content_digest
        dest = self._root / digest
        if binding.kind is not ArtifactKind.TREE:
            path = dest / PurePosixPath(binding.name).name
            if not path.exists():
                self._fetch(binding.reference, path)
            return path
        if dest.is_dir():
            return dest
        try:
            manifest = TreeManifest.decode(
                self._plane.hydrate(self._task_id, binding.reference)
            )
        except TreeManifestError as exc:
            raise ArtifactInputUnreadable(
                f"artifact {binding.name} names no tree: {exc}"
            ) from exc
        scope = binding.reference.authorization_scope
        staging = self._root / f".{digest}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for member in manifest.members:
            self._fetch(member.reference(scope), staging / member.path)
        staging.rename(dest)
        return dest

    def copy_at(self, binding: ArtifactBinding, dest: Path) -> None:
        """Put a writable copy of the artifact at ``dest`` under its own layout: a
        file as ``dest``, a tree as a directory of its files."""
        if binding.kind is not ArtifactKind.TREE:
            self._copy(binding.reference, dest)
            return
        try:
            manifest = TreeManifest.decode(
                self._plane.hydrate(self._task_id, binding.reference)
            )
        except TreeManifestError as exc:
            raise ArtifactInputUnreadable(
                f"artifact {binding.name} names no tree: {exc}"
            ) from exc
        scope = binding.reference.authorization_scope
        dest.mkdir(parents=True, exist_ok=True)
        for member in manifest.members:
            self._copy(member.reference(scope), dest / member.path)

    def _copy(self, reference: ContentReference, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._plane.hydrate_to(self._task_id, reference, dest)

    def _fetch(self, reference: ContentReference, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._plane.materialize_to(self._task_id, reference, dest)
        os.chmod(dest, 0o444)


__all__ = [
    "ARTIFACT_INPUTS_DIR",
    "ArtifactInputUnreadable",
    "ArtifactMaterializer",
    "artifact_inputs_dir",
]
