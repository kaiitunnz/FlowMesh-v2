"""Digest-indexed object files under a per-scope directory partition.

The backing a worker's cache and a shared-filesystem content store both keep their
objects in: one file per object, named by its digest, under the scope that isolates it.
A write is put-if-absent, so re-writing identical bytes is free and an object is never
rewritten in place. It holds bytes only — what an object means, and which binding keeps
it alive, live above it.
"""

from collections.abc import Iterator
from pathlib import Path

from shared.utils.atomic import atomic_write_bytes

from .reference import OCTET_STREAM, ContentReference
from .store import ContentHydrationError, ContentStoreError, reference_for

_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def safe_segment(value: str | None) -> str:
    """A single path segment safe from traversal, or ``_`` for an empty scope."""
    if not value:
        return "_"
    if value in {".", ".."} or any(c not in _SAFE for c in value):
        raise ContentStoreError(f"unsafe content-store segment {value!r}")
    return value


class FilesystemObjectBacking:
    """Content-addressed object files under a local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def object_path(self, scope: str | None, digest: str) -> Path:
        digest = safe_segment(digest)
        return (
            self._root
            / safe_segment(scope)
            / "objects"
            / safe_segment(digest[:2])
            / digest
        )

    def scope_path(self, scope: str | None, *segments: str) -> Path:
        path = self._root / safe_segment(scope)
        for segment in segments:
            path = path / safe_segment(segment)
        return path

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        atomic_write_bytes(
            self.object_path(scope, reference.content_digest), data, if_absent=True
        )
        return reference

    def read(self, scope: str, digest: str) -> bytes:
        path = self.object_path(scope, digest)
        if not path.exists():
            raise ContentHydrationError(f"no content for {digest} in scope {scope}")
        return path.read_bytes()

    def holds(self, scope: str, digest: str) -> bool:
        return self.object_path(scope, digest).exists()

    def written_at(self, scope: str, digest: str) -> float | None:
        """When the object landed, or None if it is not here."""
        path = self.object_path(scope, digest)
        return path.stat().st_mtime if path.exists() else None

    def remove(self, scope: str, digest: str) -> None:
        self.object_path(scope, digest).unlink(missing_ok=True)

    def iter_objects(self) -> Iterator[tuple[str, str]]:
        """Every ``(scope, digest)`` this backing holds."""
        if not self._root.exists():
            return
        for scope_dir in self._root.iterdir():
            objects = scope_dir / "objects"
            if not objects.is_dir():
                continue
            for shard in objects.iterdir():
                for entry in shard.iterdir():
                    if entry.is_file():
                        yield scope_dir.name, entry.name
