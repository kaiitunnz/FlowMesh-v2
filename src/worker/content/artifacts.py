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
import shutil
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pydantic import BaseModel

from shared.content import (
    TREE_MEDIA_TYPE,
    FabricObjectStore,
    TreeManifest,
    TreeMember,
)
from shared.content.archive import TreeArchive, digest_archive
from shared.schemas.artifact import CHECKPOINTS_ARTIFACT, ArtifactRef
from shared.tasks.artifact_binding import (
    FINAL_REVISION,
    ArchiveFacts,
    ArtifactBinding,
    ArtifactKind,
    revision_step,
)
from shared.utils.atomic import is_atomic_temp
from shared.utils.manifest import RESULTS_NAME
from shared.utils.nofollow import (
    PathRefused,
    fd_path,
    open_dir,
    open_dir_at,
    open_regular,
    regular_files,
    regular_files_at,
)

CHECKPOINTS_DIR = CHECKPOINTS_ARTIFACT
_CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)")
_READ_BYTES = 1 << 20
_ARCHIVE_SUFFIX = ".tar.gz"
CHECKPOINT_SPOOL_DIR = ".checkpoint-spool"


class ArtifactChanged(RuntimeError):
    """A file changed while it was being published."""


class ReservedArtifactName(ValueError):
    """A task left an artifact under a name its result owns."""


def checkpoint_spool_dir(out_dir: Path) -> Path:
    """Return where a task's closed checkpoints wait to publish, beside its
    artifacts."""
    return out_dir / CHECKPOINT_SPOOL_DIR


def spool_checkpoint(checkpoint_dir: Path, spool_dir: Path, step: int) -> Path | None:
    """Snapshot a closed ``checkpoint-<n>`` directory into the spool by hard link.

    The snapshot appears whole, under ``step-<n>``, once every file is linked, so the
    uploader never sees half of it. Each file is linked from the descriptor that
    opened it without following a link. A trainer rotating its checkpoints only
    unlinks them, which leaves the snapshot's links in place. Returns the snapshot, or
    None when the step already has one.
    """
    ready = spool_dir / f"step-{step}"
    if ready.exists():
        return None
    staging = spool_dir / f".step-{step}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    target = staging / checkpoint_dir.name
    for rel, opened in regular_files(checkpoint_dir):
        if isinstance(opened, OSError):
            raise opened
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with opened:
            try:
                os.link(fd_path(opened.fileno()), dest, follow_symlinks=True)
            except OSError:
                with dest.open("xb") as copy:
                    shutil.copyfileobj(opened, copy)
    target.mkdir(parents=True, exist_ok=True)
    staging.rename(ready)
    return ready


_SPOOL_OPEN = ".uploading"
_SPOOL_FAILED = ".failed"
# How long a spool stays open after its uploader last refreshed it, which bounds how
# long a save waits on an uploader that stopped without closing the spool.
SPOOL_STALE_SEC = 60.0


def open_spool(spool_dir: Path) -> None:
    """Mark a spool as having an uploader, which a trainer's save waits on."""
    (spool_dir / _SPOOL_OPEN).touch()


def refresh_spool(spool_dir: Path) -> None:
    """Refresh an open spool's mark; raises ``FileNotFoundError`` once it is closed."""
    os.utime(spool_dir / _SPOOL_OPEN, follow_symlinks=False)


def close_spool(spool_dir: Path) -> None:
    """Mark a spool as having no uploader, so nothing waits on it."""
    (spool_dir / _SPOOL_OPEN).unlink(missing_ok=True)


def spool_is_open(spool_dir: Path) -> bool:
    """Return whether an uploader publishes what the spool holds."""
    try:
        opened = os.stat(spool_dir / _SPOOL_OPEN, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return time.time() - opened.st_mtime < SPOOL_STALE_SEC


def fail_spool(spool_dir: Path, error: str) -> None:
    """Record that a spooled checkpoint could not be published."""
    try:
        (spool_dir / _SPOOL_FAILED).write_text(error, encoding="utf-8")
    except OSError:
        pass


def spool_failed(spool_dir: Path) -> bool:
    """Return whether a spooled checkpoint could not be published."""
    return (spool_dir / _SPOOL_FAILED).exists()


def spooled_steps(spool_dir: Path) -> list[tuple[int, Path]]:
    """Return the snapshots waiting in the spool, by step."""
    try:
        entries = list(os.scandir(spool_dir))
    except FileNotFoundError:
        return []
    steps = [
        (step, Path(entry.path))
        for entry in entries
        if (step := revision_step(entry.name)) is not None and entry.is_dir()
    ]
    return sorted(steps)


def checkpoint_step(name: str) -> int | None:
    """Return the training step a trainer's ``checkpoint-<n>`` directory saved, or
    None."""
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
        self,
        artifacts_dir: Path,
        archives: Mapping[str, str],
        step_revisions: bool = False,
    ) -> list[ArtifactBinding]:
        """Publish the ``final`` set: every top-level artifact, and an archive view
        for each ``archives`` name over the tree it maps to.

        A task whose checkpoints publish as ``step-<n>`` revisions (``step_revisions``)
        leaves its ``checkpoints/checkpoint-<n>`` directories out. Raises
        ``ReservedArtifactName`` for a top-level ``results.json``, which names the
        task's result.
        """
        bindings: list[ArtifactBinding] = []
        trees: dict[str, TreeManifest] = {}
        with open_dir(artifacts_dir) as top_fd:
            root = fd_path(top_fd)
            for name, is_dir in _final_entries(top_fd, step_revisions):
                if name == RESULTS_NAME:
                    raise ReservedArtifactName(
                        f"an artifact may not be named {RESULTS_NAME}, which names "
                        "the task's result"
                    )
                if is_dir:
                    binding, manifest = self._publish_members(
                        name,
                        regular_files_at(top_fd, *name.split("/")),
                        FINAL_REVISION,
                    )
                    trees[name] = manifest
                elif (published := self._publish_file_at(top_fd, name)) is not None:
                    binding = published
                else:
                    continue
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
                        root / tree_name,
                    )
                )
        return bindings

    def _publish_file_at(self, top_fd: int, name: str) -> ArtifactBinding | None:
        """Publish the regular file ``name`` below ``top_fd`` from the descriptor
        that opened it, or None when it is no longer one."""
        *parents, last = name.split("/")
        try:
            with open_dir_at(top_fd, *parents) as dir_fd:
                opened = open_regular(dir_fd, last)
        except (FileNotFoundError, PathRefused):
            return None
        if opened is None:
            return None
        with opened:
            reference = self._store.write_file(
                self._scope, fd_path(opened.fileno()), cancelled=self._cancelled
            )
        return ArtifactBinding(
            name=name,
            revision=FINAL_REVISION,
            kind=ArtifactKind.FILE,
            reference=reference,
            size_bytes=reference.size_bytes,
            largest=reference,
        )

    def publish_tree(
        self, name: str, top: Path, revision: str
    ) -> tuple[ArtifactBinding, TreeManifest]:
        """Publish every regular file under ``top`` and the manifest naming them."""
        return self._publish_members(name, regular_files(top), revision)

    def _publish_members(
        self,
        name: str,
        files: Iterable[tuple[str, BinaryIO | OSError]],
        revision: str,
    ) -> tuple[ArtifactBinding, TreeManifest]:
        members: list[TreeMember] = []
        for rel, opened in files:
            if isinstance(opened, OSError):
                raise opened
            with opened:
                reference = self._store.write_file(
                    self._scope, fd_path(opened.fileno()), cancelled=self._cancelled
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
    """Return the archive views a result names, each mapped to the tree it archives.

    An executor names a model's archive in a ``*_archive`` field beside the model, as
    the model's path with ``.tar.gz`` appended.
    """
    archives: dict[str, str] = {}
    for field, value in result:
        if (
            field.endswith("_archive")
            and isinstance(value, ArtifactRef)
            and value.path.endswith(_ARCHIVE_SUFFIX)
        ):
            archives[value.path] = value.path.removesuffix(_ARCHIVE_SUFFIX)
    return archives


def archive_top(tree_name: str) -> str:
    """Return the directory an archive of a tree puts its members under."""
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


def _final_entries(top_fd: int, step_revisions: bool) -> Iterator[tuple[str, bool]]:
    """Yield each artifact of the ``final`` set: its name and whether it is a tree."""
    for name, is_dir in _entries(top_fd):
        if not step_revisions or name != CHECKPOINTS_DIR or not is_dir:
            yield name, is_dir
            continue
        try:
            with open_dir_at(top_fd, CHECKPOINTS_DIR) as fd:
                children = _entries(fd)
        except (FileNotFoundError, PathRefused):
            continue
        for child, child_is_dir in children:
            if not (child_is_dir and checkpoint_step(child) is not None):
                yield f"{CHECKPOINTS_DIR}/{child}", child_is_dir


def _entries(dir_fd: int) -> list[tuple[str, bool]]:
    found: list[tuple[str, bool]] = []
    with os.scandir(dir_fd) as entries:
        for entry in entries:
            if is_atomic_temp(entry.name):
                continue
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
                found.append((entry.name, stat.S_ISDIR(mode)))
    return sorted(found)


__all__ = [
    "CHECKPOINTS_DIR",
    "CHECKPOINT_SPOOL_DIR",
    "SPOOL_STALE_SEC",
    "checkpoint_spool_dir",
    "close_spool",
    "fail_spool",
    "open_spool",
    "refresh_spool",
    "spool_checkpoint",
    "spool_failed",
    "spool_is_open",
    "spooled_steps",
    "ArtifactChanged",
    "ReservedArtifactName",
    "ArtifactPublisher",
    "archive_top",
    "checkpoint_step",
    "declared_archives",
]
