"""Digest-indexed object files under a per-scope directory partition.

The backing a worker's cache and a shared-filesystem content store both keep their
objects in: one file per object, named by its digest, under the scope that isolates it.
An object is written to a temp file and linked into place only where nothing is, so
re-writing identical bytes is free and an object is never rewritten in place. It holds
bytes only — what an object means, and which binding keeps it alive, live above it.
"""

import contextlib
import errno
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

from shared.utils.atomic import is_atomic_temp, link_no_replace, temp_path

from .reference import OCTET_STREAM, ContentReference
from .store import (
    STREAM_CHUNK_BYTES,
    ContentHydrationError,
    ContentStoreError,
    ContentUnavailable,
    FabricObjectStore,
    WriteCancelled,
    check_existing_size,
    new_hasher,
    reference_for,
    verified_chunks,
    write_verified,
)

_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def safe_segment(value: str | None) -> str:
    """A single path segment safe from traversal, or ``_`` for an empty scope."""
    if not value:
        return "_"
    if value in {".", ".."} or any(c not in _SAFE for c in value):
        raise ContentStoreError(f"unsafe content-store segment {value!r}")
    return value


class FilesystemObjectBacking:
    """Content-addressed object files under a local directory.

    With ``durable`` every committed object and its directory entry reach the disk
    before the write returns, as a store of record needs; a cache that can lose a copy
    without losing the object skips that cost. With ``repair`` an object found the wrong
    size under its digest is replaced, which only a copy of an object held elsewhere
    can afford: a store of record keeps what it has and refuses the write.
    """

    def __init__(
        self, root: Path, *, durable: bool = False, repair: bool = False
    ) -> None:
        self._root = root
        self._durable = durable
        self._repair = repair

    def object_path(self, scope: str | None, digest: str) -> Path:
        digest = safe_segment(digest)
        return (
            self._root
            / safe_segment(scope)
            / "objects"
            / safe_segment(digest[:2])
            / digest
        )

    def staging_dir(self, scope: str | None) -> Path:
        """Where in-flight writes to a scope land before they are committed.

        On the same filesystem as the objects, so a commit is a link, and outside the
        object tree, so nothing that walks objects ever sees a partial one.
        """
        staging = self._root / safe_segment(scope) / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        return staging

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        tmp = temp_path(self.staging_dir(scope))
        try:
            with tmp.open("xb") as fh:
                fh.write(data)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self.adopt(reference, tmp)
        return reference

    def write_file(
        self,
        scope: str,
        source: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Copy a file in, naming it by the digest of exactly the bytes copied."""
        tmp = temp_path(self.staging_dir(scope))
        hasher = new_hasher()
        size = 0
        try:
            with source.open("rb") as src, tmp.open("xb") as dst:
                while chunk := src.read(STREAM_CHUNK_BYTES):
                    if cancelled is not None and cancelled():
                        raise WriteCancelled(f"write of {source.name} was cancelled")
                    hasher.update(chunk)
                    dst.write(chunk)
                    size += len(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        reference = ContentReference(
            authorization_scope=scope,
            content_digest=hasher.hexdigest(),
            size_bytes=size,
            media_type=media_type,
        )
        self.adopt(reference, tmp)
        return reference

    def adopt(self, reference: ContentReference, tmp: Path) -> None:
        """Commit a finished file holding exactly the reference's bytes.

        An object already under the digest is kept rather than replaced, once its size
        shows it is not a truncated copy.
        """
        target = self.object_path(
            reference.authorization_scope, reference.content_digest
        )
        if self._repair and self._wrong_size(reference, target):
            target.unlink(missing_ok=True)
        try:
            if link_no_replace(tmp, target, durable=self._durable):
                return
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise ContentStoreError(
                    f"content write of {reference.content_digest} failed: the "
                    "content filesystem must support hard links, which commit an "
                    f"object without replacing one already there: {exc}"
                ) from exc
            raise ContentStoreError(
                f"content write of {reference.content_digest} failed: {exc}"
            ) from exc
        self._check_existing(reference, target)

    @staticmethod
    def _wrong_size(reference: ContentReference, target: Path) -> bool:
        try:
            return target.stat().st_size != reference.size_bytes
        except FileNotFoundError:
            return False

    def _check_existing(self, reference: ContentReference, target: Path) -> None:
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ContentStoreError(
                f"content {reference.content_digest} is neither written nor readable: "
                f"{exc}"
            ) from exc
        check_existing_size(reference, size)

    def open_object(self, scope: str, digest: str) -> BinaryIO:
        """The object file opened for reading, which stays readable once open even if
        the object is removed."""
        try:
            return self.object_path(scope, digest).open("rb")
        except FileNotFoundError as exc:
            raise ContentHydrationError(
                f"no content for {digest} in scope {scope}"
            ) from exc
        except OSError as exc:
            raise ContentUnavailable(
                f"could not read {digest} in scope {scope}: {exc}"
            ) from exc

    def read(self, scope: str, digest: str) -> bytes:
        with self.open_object(scope, digest) as fh:
            try:
                return fh.read()
            except OSError as exc:
                raise ContentUnavailable(
                    f"could not read {digest} in scope {scope}: {exc}"
                ) from exc

    def stream(
        self, reference: ContentReference, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        """The object's verified bytes in chunks."""
        fh = self.open_object(reference.authorization_scope, reference.content_digest)

        def chunks() -> Iterator[bytes]:
            with fh:
                while True:
                    try:
                        chunk = fh.read(chunk_bytes)
                    except OSError as exc:
                        raise ContentUnavailable(
                            f"could not read {reference.content_digest}: {exc}"
                        ) from exc
                    if not chunk:
                        return
                    yield chunk

        return verified_chunks(reference, chunks())

    def copy_out(self, reference: ContentReference, dest: Path) -> None:
        """Copy a held object to ``dest``, verified on the way."""
        write_verified(self.stream(reference), dest)

    def holds(self, scope: str, digest: str) -> bool:
        return self.object_path(scope, digest).exists()

    def stat(self, scope: str, digest: str) -> os.stat_result | None:
        """The object file's size and last-touched time, or None if it is not here."""
        try:
            return self.object_path(scope, digest).stat()
        except FileNotFoundError:
            return None

    def touch(self, scope: str, digest: str) -> None:
        """Mark the object as touched now."""
        with contextlib.suppress(FileNotFoundError):
            os.utime(self.object_path(scope, digest))

    def remove(self, scope: str, digest: str) -> None:
        self.object_path(scope, digest).unlink(missing_ok=True)

    def clear_staging(self, idle_sec: float) -> int:
        """Drop in-flight writes untouched for ``idle_sec``, and return how many.

        A live write keeps its file moving, so one idle that long belongs to a process
        that stopped mid-write; the directory may be shared with other live writers,
        whose files are left alone.
        """
        if not self._root.exists():
            return 0
        cutoff = time.time() - idle_sec
        removed = 0
        for scope_dir in self._root.iterdir():
            if not (staging := scope_dir / "staging").is_dir():
                continue
            for entry in staging.iterdir():
                with contextlib.suppress(FileNotFoundError):
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        removed += 1
        return removed

    def iter_objects(self) -> Iterator[tuple[str, str]]:
        """Every ``(scope, digest)`` this backing holds."""
        if not self._root.exists():
            return
        for scope_dir in self._root.iterdir():
            objects = scope_dir / "objects"
            if not objects.is_dir():
                continue
            for shard in objects.iterdir():
                if not shard.is_dir():
                    continue
                for entry in shard.iterdir():
                    if entry.is_file() and not is_atomic_temp(entry.name):
                        yield scope_dir.name, entry.name


class SharedFilesystemObjectStore(FabricObjectStore):
    """The shared durable store on a filesystem every node mounts at the same path.

    Isolation here is the filesystem's: a deployment gets per-scope separation only
    where projected identities, mounts, or ACLs enforce it, because a directory name
    under one shared identity does not.
    """

    def __init__(self, root: Path) -> None:
        self._objects = FilesystemObjectBacking(root, durable=True)

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._objects.write(scope, data, media_type=media_type)

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        return self._objects.write_file(
            scope, path, media_type=media_type, cancelled=cancelled
        )

    def fetch(self, reference: ContentReference) -> bytes:
        return self._objects.read(
            reference.authorization_scope, reference.content_digest
        )

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        return self._objects.stream(reference, chunk_bytes)
