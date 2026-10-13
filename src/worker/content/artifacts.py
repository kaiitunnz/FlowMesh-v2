"""Publishing a task's artifacts into the shared store.

A task's artifacts are what its executor left under its output's ``artifacts``
directory. Each top-level entry publishes as one artifact — a file as itself, a
directory as a tree of its files — except the trainer's ``checkpoint-<n>`` saves under
``checkpoints``, which publish as their own revisions while training runs. Every file is
in the store before the binding naming it is reported, so a binding control accepts
names bytes that outlive this worker.
"""

import hashlib
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from shared.content import (
    TREE_MEDIA_TYPE,
    FabricObjectStore,
    TreeManifest,
    TreeMember,
)
from shared.content.archive import TreeArchive, digest_archive
from shared.schemas.artifact import ArtifactRef
from shared.tasks.artifact_binding import (
    FINAL_REVISION,
    ArchiveFacts,
    ArtifactBinding,
    ArtifactKind,
)
from shared.utils.atomic import is_atomic_temp
from shared.utils.nofollow import regular_files

CHECKPOINTS_DIR = "checkpoints"
_CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)")
_READ_BYTES = 1 << 20
_ARCHIVE_SUFFIX = ".tar.gz"


class ArtifactChanged(RuntimeError):
    """A file changed while it was being published."""


def checkpoint_step(name: str) -> int | None:
    """The training step a trainer's ``checkpoint-<n>`` directory saved, or None."""
    return int(match.group(1)) if (match := _CHECKPOINT.fullmatch(name)) else None


class ArtifactPublisher:
    """Writes a task's artifacts under its scope and names each by a binding."""

    def __init__(
        self,
        store: FabricObjectStore,
        scope: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._store = store
        self._scope = scope
        self._cancelled = cancelled

    def publish_final(
        self, artifacts_dir: Path, archives: Mapping[str, str]
    ) -> list[ArtifactBinding]:
        """Publish the ``final`` set: every top-level artifact, and an archive view
        for each ``archives`` name over the tree it maps to."""
        bindings: list[ArtifactBinding] = []
        trees: dict[str, TreeManifest] = {}
        for name, path, is_dir in _final_entries(artifacts_dir):
            if is_dir:
                binding, manifest = self.publish_tree(name, path, FINAL_REVISION)
                trees[name] = manifest
            else:
                binding = self.publish_file(name, path, FINAL_REVISION)
            bindings.append(binding)
        tree_bindings = {binding.name: binding for binding in bindings}
        for archive_name, tree_name in archives.items():
            if (archived := trees.get(tree_name)) is None:
                continue
            bindings.append(
                self.publish_archive(
                    archive_name,
                    tree_bindings[tree_name],
                    archived,
                    artifacts_dir / tree_name,
                )
            )
        return bindings

    def publish_file(self, name: str, path: Path, revision: str) -> ArtifactBinding:
        reference = self._store.write_file(self._scope, path, cancelled=self._cancelled)
        return ArtifactBinding(
            name=name,
            revision=revision,
            kind=ArtifactKind.FILE,
            reference=reference,
            size_bytes=reference.size_bytes,
            largest=reference,
        )

    def publish_tree(
        self, name: str, top: Path, revision: str
    ) -> tuple[ArtifactBinding, TreeManifest]:
        """Publish every regular file under ``top`` and the manifest naming them."""
        members: list[TreeMember] = []
        for rel, opened in regular_files(top):
            if isinstance(opened, OSError):
                raise opened
            opened.close()
            reference = self._store.write_file(
                self._scope, top / rel, cancelled=self._cancelled
            )
            members.append(
                TreeMember(rel, reference.content_digest, reference.size_bytes)
            )
        manifest = TreeManifest.of(members)
        reference = self._store.write(
            self._scope, manifest.encode(), media_type=TREE_MEDIA_TYPE
        )
        largest = manifest.largest()
        binding = ArtifactBinding(
            name=name,
            revision=revision,
            kind=ArtifactKind.TREE,
            reference=reference,
            size_bytes=manifest.size_bytes,
            files=len(manifest.members),
            largest=largest.reference(self._scope) if largest is not None else None,
        )
        return binding, manifest

    def publish_archive(
        self,
        name: str,
        tree: ArtifactBinding,
        manifest: TreeManifest,
        top: Path,
    ) -> ArtifactBinding:
        """Name the archive view of a published tree, hashing the archive in one pass
        over the tree's files, each checked against the digest it was published as."""
        archive = TreeArchive(manifest, archive_top(tree.name))
        digest = digest_archive(archive, _verified_reader(top))
        return ArtifactBinding(
            name=name,
            revision=tree.revision,
            kind=ArtifactKind.ARCHIVE,
            reference=tree.reference,
            size_bytes=digest.size_bytes,
            files=tree.files,
            largest=tree.largest,
            of=tree.name,
            archive=ArchiveFacts(
                sha256=digest.sha256,
                crc32=digest.crc32,
                size_bytes=digest.size_bytes,
                top=archive_top(tree.name),
            ),
        )


def declared_archives(result: BaseModel) -> dict[str, str]:
    """The archive views a result names, each mapped to the tree it archives.

    An executor names a model's archive in a ``*_archive`` field beside the model, as
    the model's path with ``.tar.gz`` appended.
    """
    archives: dict[str, str] = {}
    for field in type(result).model_fields:
        value = getattr(result, field)
        if (
            field.endswith("_archive")
            and isinstance(value, ArtifactRef)
            and value.path.endswith(_ARCHIVE_SUFFIX)
        ):
            archives[value.path] = value.path.removesuffix(_ARCHIVE_SUFFIX)
    return archives


def archive_top(tree_name: str) -> str:
    """The directory an archive of a tree puts its members under."""
    return PurePosixPath(tree_name).name


def _verified_reader(top: Path) -> Callable[[TreeMember, int, int], Iterator[bytes]]:
    def read(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        hasher = hashlib.sha256()
        whole = start == 0 and length == member.size_bytes
        with (top / member.path).open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining:
                chunk = fh.read(min(_READ_BYTES, remaining))
                if not chunk:
                    raise ArtifactChanged(f"{member.path} shrank while it published")
                remaining -= len(chunk)
                if whole:
                    hasher.update(chunk)
                yield chunk
        if whole and hasher.hexdigest() != member.digest:
            raise ArtifactChanged(f"{member.path} changed while it published")

    return read


def _final_entries(artifacts_dir: Path) -> Iterator[tuple[str, Path, bool]]:
    """Each artifact of the ``final`` set: its name, path, and whether it is a tree."""
    for name, path, is_dir in _entries(artifacts_dir, ""):
        if name != CHECKPOINTS_DIR or not is_dir:
            yield name, path, is_dir
            continue
        for child, child_path, child_is_dir in _entries(path, f"{CHECKPOINTS_DIR}/"):
            if not (child_is_dir and checkpoint_step(child_path.name) is not None):
                yield child, child_path, child_is_dir


def _entries(directory: Path, prefix: str) -> Iterator[tuple[str, Path, bool]]:
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except FileNotFoundError:
        return
    for entry in entries:
        if is_atomic_temp(entry.name):
            continue
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
            yield f"{prefix}{entry.name}", Path(entry.path), stat.S_ISDIR(mode)


__all__ = [
    "CHECKPOINTS_DIR",
    "ArtifactChanged",
    "ArtifactPublisher",
    "archive_top",
    "checkpoint_step",
    "declared_archives",
]
