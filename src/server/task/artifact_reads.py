"""How the root serves a task's published artifacts from the shared content store.

A download names an artifact by its published name and, inside a tree, a member's
path. The bytes stream from the store and are never held whole: a whole read verifies
each object before its last bytes leave, so a corrupt object ends the response short; a
range reads only the slice it names, which cannot be checked against a digest of the
whole. An archive view frames the tree's members as they stream, never storing the
archive. A bundle tars and compresses a task's result, artifacts and logs on a thread
of its own, handing the response a bounded window of its output.
"""

import base64
import io
import queue
import tarfile
import threading
from collections.abc import Buffer, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
from functools import partial
from pathlib import PurePosixPath

from shared.content import ContentReference, TreeManifest, TreeMember
from shared.content.archive import ARCHIVE_MEDIA_TYPE, TreeArchive
from shared.tasks.artifact_binding import (
    ArtifactBinding,
    ArtifactBindings,
    ArtifactKind,
    resolve_artifact_path,
)

from .results import ResultReader

# What a bundle's writer may run ahead of the response sending it.
_BUNDLE_WINDOW_CHUNKS = 4
_BUNDLE_CHUNK_BYTES = 1 << 20
_BUNDLE_PUT_TIMEOUT_SEC = 1.0


class ArtifactNotFound(LookupError):
    """No published artifact, or member of one, has the name."""


@dataclass(frozen=True, slots=True)
class ArtifactDownload:
    """One downloadable object: its size, the digest of its whole bytes, when it was
    published, and how to read it whole or in part."""

    name: str
    size_bytes: int
    sha256: str
    published_at: str | None
    media_type: str | None
    whole: Callable[[], Iterator[bytes]]
    part: Callable[[int, int], Iterator[bytes]]

    @property
    def etag(self) -> str:
        return f'"sha256:{self.sha256}"'

    @property
    def repr_digest(self) -> str:
        encoded = base64.b64encode(bytes.fromhex(self.sha256)).decode("ascii")
        return f"sha-256=:{encoded}:"

    @property
    def last_modified(self) -> str | None:
        if not self.published_at:
            return None
        try:
            published = datetime.fromisoformat(self.published_at)
        except ValueError:
            return None
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        return format_datetime(published.astimezone(UTC), usegmt=True)


def resolve_download(
    reader: ResultReader, bindings: ArtifactBindings, revision: str, path: str
) -> ArtifactDownload:
    """The object ``path`` names among a task's artifacts at ``revision``."""
    if (found := resolve_artifact_path(bindings, revision, path)) is None:
        raise ArtifactNotFound(path)
    binding, inner = found
    match binding.kind:
        case ArtifactKind.FILE:
            if inner:
                raise ArtifactNotFound(path)
            return _object_download(reader, binding, binding.reference, path)
        case ArtifactKind.TREE:
            member = (
                reader.read_manifest(binding.reference).member(inner) if inner else None
            )
            if member is None:
                raise ArtifactNotFound(path)
            return _object_download(
                reader,
                binding,
                member.reference(binding.reference.authorization_scope),
                path,
            )
        case ArtifactKind.ARCHIVE:
            if inner:
                raise ArtifactNotFound(path)
            return _archive_download(reader, binding, path)


def _object_download(
    reader: ResultReader,
    binding: ArtifactBinding,
    reference: ContentReference,
    path: str,
) -> ArtifactDownload:
    return ArtifactDownload(
        name=path,
        size_bytes=reference.size_bytes,
        sha256=reference.content_digest,
        published_at=binding.published_at,
        media_type=None,
        whole=lambda: reader.open(reference),
        part=lambda start, length: reader.open_range(reference, start, length),
    )


def _archive_download(
    reader: ResultReader, binding: ArtifactBinding, path: str
) -> ArtifactDownload:
    facts = binding.archive
    assert facts is not None
    archive = TreeArchive(reader.read_manifest(binding.reference), facts.top)
    scope = binding.reference.authorization_scope
    if archive.size_bytes != facts.size_bytes:
        raise ArtifactNotFound(path)

    def whole_member(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        return reader.open(member.reference(scope))

    def part_member(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        return reader.open_range(member.reference(scope), start, length)

    return ArtifactDownload(
        name=path,
        size_bytes=facts.size_bytes,
        sha256=facts.sha256,
        published_at=binding.published_at,
        media_type=ARCHIVE_MEDIA_TYPE,
        whole=lambda: archive.chunks(whole_member, crc32=facts.crc32),
        part=lambda start, length: archive.chunks(
            part_member, crc32=facts.crc32, start=start, end=start + length
        ),
    )


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One regular file a bundle carries, under ``arcname``."""

    arcname: str
    size_bytes: int
    read: Callable[[], Iterator[bytes]]
    mtime: int = 0


def artifact_bundle_files(
    reader: ResultReader, bindings: ArtifactBindings, revision: str, top: str
) -> Iterator[BundleFile]:
    """Every published file of a revision under ``top``, a tree as its members.

    An archive view is left out: its bytes are its tree's, already in the bundle.
    """
    for (rev, name), binding in sorted(bindings.items()):
        if rev != revision or binding.kind is ArtifactKind.ARCHIVE:
            continue
        mtime = _mtime(binding.published_at)
        if binding.kind is ArtifactKind.FILE:
            reference = binding.reference
            yield BundleFile(
                f"{top}/{name}",
                reference.size_bytes,
                partial(reader.open, reference),
                mtime,
            )
            continue
        manifest: TreeManifest = reader.read_manifest(binding.reference)
        scope = binding.reference.authorization_scope
        for member in manifest.members:
            member_ref = member.reference(scope)
            yield BundleFile(
                f"{top}/{PurePosixPath(name, member.path).as_posix()}",
                member.size_bytes,
                partial(reader.open, member_ref),
                mtime,
            )


def _mtime(published_at: str | None) -> int:
    if not published_at:
        return 0
    try:
        return int(datetime.fromisoformat(published_at).timestamp())
    except ValueError:
        return 0


class _ChunkReader(io.RawIOBase):
    """A readable over an iterator of chunks."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._held = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        """Fill ``buffer`` until it is full or the chunks end."""
        view = memoryview(buffer).cast("B")
        filled = 0
        while filled < len(view):
            while not self._held:
                if (held := next(self._chunks, None)) is None:
                    return filled
                self._held = held
            count = min(len(view) - filled, len(self._held))
            view[filled : filled + count] = self._held[:count]
            self._held = self._held[count:]
            filled += count
        return filled


class _Cancelled(Exception):
    """The response stopped reading the bundle."""


class _WindowWriter(io.RawIOBase):
    """Hands what a writer writes to a bounded queue, in chunks."""

    def __init__(self, window: queue.Queue[bytes | BaseException | None]) -> None:
        self._window = window
        self._pending = bytearray()
        self.cancelled = threading.Event()

    def writable(self) -> bool:
        return True

    def write(self, data: Buffer, /) -> int:
        view = memoryview(data).cast("B")
        self._pending += view
        while len(self._pending) >= _BUNDLE_CHUNK_BYTES:
            self.put(bytes(self._pending[:_BUNDLE_CHUNK_BYTES]))
            del self._pending[:_BUNDLE_CHUNK_BYTES]
        return len(view)

    def put(self, item: bytes | BaseException | None) -> None:
        while not self.cancelled.is_set():
            try:
                self._window.put(item, timeout=_BUNDLE_PUT_TIMEOUT_SEC)
                return
            except queue.Full:
                continue
        raise _Cancelled()

    def finish(self) -> None:
        if self._pending:
            self.put(bytes(self._pending))
            self._pending.clear()


class BundleStream(Iterator[bytes]):
    """A gzip-compressed tar of ``files``, written on a thread of its own.

    The writer runs at most a window ahead of the reader. A file that fails to read,
    or verifies as other bytes than it names, ends the stream with that error before
    the archive's end; closing the stream stops the writer.
    """

    def __init__(self, files: Callable[[], Iterator[BundleFile]], name: str) -> None:
        self._window: queue.Queue[bytes | BaseException | None] = queue.Queue(
            maxsize=_BUNDLE_WINDOW_CHUNKS
        )
        self._writer = _WindowWriter(self._window)
        self._files = files
        self._done = False
        threading.Thread(target=self._write, name=f"bundle-{name}", daemon=True).start()

    def _write(self) -> None:
        writer = self._writer
        try:
            with tarfile.open(fileobj=writer, mode="w|gz") as archive:
                for entry in self._files():
                    info = tarfile.TarInfo(entry.arcname)
                    info.size = entry.size_bytes
                    info.mode = 0o644
                    info.mtime = entry.mtime
                    archive.addfile(info, _ChunkReader(entry.read()))
            writer.finish()
            writer.put(None)
        except _Cancelled:
            return
        except BaseException as exc:
            try:
                writer.put(exc)
            except _Cancelled:
                return

    def __next__(self) -> bytes:
        while not self._done:
            try:
                item = self._window.get(timeout=_BUNDLE_PUT_TIMEOUT_SEC)
            except queue.Empty:
                continue
            if item is None:
                self._done = True
                break
            if isinstance(item, BaseException):
                self._done = True
                raise item
            return item
        raise StopIteration

    def close(self) -> None:
        """Stop reading and let the writer go."""
        self._done = True
        self._writer.cancelled.set()


__all__ = [
    "ArtifactDownload",
    "ArtifactNotFound",
    "BundleFile",
    "artifact_bundle_files",
    "BundleStream",
    "resolve_download",
]
