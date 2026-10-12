"""The immutable-object core every reference-backed fabric value is stored over.

An object is bytes named by their own digest within an authorization scope, written
put-if-absent and read back only under that same scope. The core knows nothing about
what the bytes mean: a caller that needs a meaning keeps its own binding to the
reference, so two kinds of value can share one identity and one backing without either
becoming a name for the other.

Integrity rests on four rules every backend keeps. A writer hashes the exact bytes it
names, so a reference never names bytes nobody hashed. Every write is conditional: an
object already under a digest key is never replaced. Every read verifies size and digest
before its last byte is released. And a write the store refuses because the key is
taken checks the existing object's size. A refusal proves the object exists, not that
its bytes match: an object corrupted at rest under its digest key stays there, and the
next read of it fails verification as a hydration error rather than returning it.
"""

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Protocol

from shared.utils.atomic import temp_path

from .reference import (
    CONTENT_ENCODING_IDENTITY,
    OCTET_STREAM,
    ContentReference,
    DigestAlgorithm,
)


class Hasher(Protocol):
    """An incremental digest over a stream of bytes."""

    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


# The digest algorithms this build can name content by and verify it against.
_HASHERS: dict[DigestAlgorithm, Callable[[], Hasher]] = {
    DigestAlgorithm.SHA256: hashlib.sha256,
}

# How much of an object one streamed read or write moves at a time.
STREAM_CHUNK_BYTES = 1 << 20


def new_hasher(algorithm: DigestAlgorithm = DigestAlgorithm.SHA256) -> Hasher:
    """A fresh incremental digest for an algorithm this build can compute."""
    return _HASHERS[algorithm]()


def content_digest(
    data: bytes, algorithm: DigestAlgorithm = DigestAlgorithm.SHA256
) -> str:
    """The immutable content identity: a hex digest over the bytes."""
    hasher = new_hasher(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


class ContentStoreError(RuntimeError):
    """A content-store write, read, or finalize failed."""


class ContentHydrationError(ContentStoreError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


class ContentUnavailable(ContentHydrationError):
    """The store could not be reached for a read; the content may well be there."""


class WriteCancelled(ContentStoreError):
    """A content write was stopped by its caller before it committed."""


class ContentVerifier:
    """Checks bytes against the reference naming them as they arrive, chunk by chunk.

    Size is checked with the digest because a reader sized its buffers from it, and an
    encoding or algorithm this build cannot verify is refused rather than trusted. Bytes
    past the declared size are refused as they arrive, so a source sending more than
    the reference names is never read to its end to find that out.
    """

    def __init__(self, reference: ContentReference) -> None:
        if (factory := _HASHERS.get(reference.digest_algorithm)) is None:
            raise ContentHydrationError(
                f"unsupported digest algorithm {reference.digest_algorithm}"
            )
        if reference.content_encoding != CONTENT_ENCODING_IDENTITY:
            raise ContentHydrationError(
                f"unsupported content encoding {reference.content_encoding}"
            )
        self._reference = reference
        self._hasher = factory()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def update(self, chunk: bytes) -> None:
        self._size += len(chunk)
        if self._size > self._reference.size_bytes:
            raise ContentHydrationError(
                f"hydrated content overran the {self._reference.size_bytes} bytes "
                f"{self._reference.content_digest} names"
            )
        self._hasher.update(chunk)

    def finish(self) -> None:
        """Raise unless everything seen is exactly the bytes the reference names."""
        if self._size != self._reference.size_bytes:
            raise ContentHydrationError(
                f"hydrated content is {self._size} bytes for a reference naming "
                f"{self._reference.size_bytes}"
            )
        if self._hasher.hexdigest() != self._reference.content_digest:
            raise ContentHydrationError(
                f"hydrated content digest mismatch for "
                f"{self._reference.content_digest}"
            )


def check_existing_size(reference: ContentReference, size: int) -> None:
    """Take an object already stored under a reference's key as the one written, once
    its size shows it is not a truncated copy."""
    if size != reference.size_bytes:
        raise ContentStoreError(
            f"content {reference.content_digest} already stored is {size} bytes "
            f"for a reference naming {reference.size_bytes}"
        )


def verify_content(reference: ContentReference, data: bytes) -> bytes:
    """Return the bytes once they are the ones the reference names, else raise."""
    verifier = ContentVerifier(reference)
    verifier.update(data)
    verifier.finish()
    return data


def verified_chunks(
    reference: ContentReference, chunks: Iterable[bytes]
) -> Iterator[bytes]:
    """Pass a stream through, releasing its last chunk only once the whole verifies.

    A consumer that writes each chunk out as it arrives therefore never completes a
    copy of bytes the reference does not name: the stream raises before its end.
    """
    verifier = ContentVerifier(reference)
    held: bytes | None = None
    for chunk in chunks:
        if not chunk:
            continue
        verifier.update(chunk)
        if held is not None:
            yield held
        held = chunk
    verifier.finish()
    if held is not None:
        yield held


def write_verified(chunks: Iterable[bytes], dest: Path) -> None:
    """Write a verified stream to ``dest`` through a temporary file beside it.

    ``dest`` appears only complete and durable: a stream that fails verification
    leaves nothing behind but the error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(dest.parent)
    try:
        with tmp.open("xb") as fh:
            for chunk in chunks:
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def reference_for(
    scope: str, data: bytes, *, media_type: str = OCTET_STREAM
) -> ContentReference:
    """The reference bytes written under a scope are named by."""
    return ContentReference(
        authorization_scope=scope,
        content_digest=content_digest(data),
        size_bytes=len(data),
        media_type=media_type,
    )


class ScopedObjectStore(Protocol):
    """Opens the store one scope's content lives in.

    The implementer decides what access that scope carries, and the store it returns
    reaches exactly that scope's content.
    """

    def for_scope(self, scope: str) -> "FabricObjectStore": ...


class FabricObjectStore(ABC):
    """Scoped put-if-absent writes and digest-verified hydration.

    ``write`` and ``hydrate`` move one object as one value, for content a caller holds
    whole anyway. ``write_file``, ``open`` and ``hydrate_to`` move it through bounded
    buffers, so an object of any size passes through a process without being held in
    it. A backend overrides the streaming calls; the defaults here go through the
    whole-value ones.
    """

    @abstractmethod
    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        """Store the bytes in a scope the caller is authorized for and name them.

        A digest already present in the scope is left as it is: identical bytes, so the
        first write stands and a repeat costs nothing.
        """

    @abstractmethod
    def fetch(self, reference: ContentReference) -> bytes:
        """Fetch a reference's bytes, raising on a missing or unauthorized read."""

    def hydrate(self, reference: ContentReference) -> bytes:
        """Fetch and verify content against the reference naming it."""
        return verify_content(reference, self.fetch(reference))

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Store a file's bytes in a scope and name them, as ``write`` does.

        The file must not change while it is written. ``cancelled`` is polled between
        the write's steps, and a write it stops leaves nothing in the store.
        """
        return self.write(scope, path.read_bytes(), media_type=media_type)

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        """A reference's bytes in chunks, raising before the last one unless they
        verify."""
        data = self.hydrate(reference)
        for start in range(0, len(data), chunk_bytes):
            yield data[start : start + chunk_bytes]

    def hydrate_to(self, reference: ContentReference, dest: Path) -> None:
        """Write a reference's verified bytes to ``dest``, replacing it whole."""
        write_verified(self.open(reference), dest)
