"""The immutable-object core every reference-backed fabric value is stored over.

An object is bytes named by their own digest within an authorization scope, written
put-if-absent and read back only under that same scope. The core knows nothing about
what the bytes mean: a caller that needs a meaning keeps its own binding to the
reference, so two kinds of value can share one identity and one backing without either
becoming a name for the other.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

from .reference import (
    CONTENT_ENCODING_IDENTITY,
    OCTET_STREAM,
    ContentReference,
    DigestAlgorithm,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# The digest algorithms this build can name content by and verify it against, each as
# the hex digest it computes over an object's bytes.
_DIGESTS: dict[DigestAlgorithm, Callable[[bytes], str]] = {
    DigestAlgorithm.SHA256: _sha256,
}


def content_digest(
    data: bytes, algorithm: DigestAlgorithm = DigestAlgorithm.SHA256
) -> str:
    """The immutable content identity: a hex digest over the bytes."""
    return _DIGESTS[algorithm](data)


class ContentStoreError(RuntimeError):
    """A content-store write, read, or finalize failed."""


class ContentHydrationError(ContentStoreError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


def verify_content(reference: ContentReference, data: bytes) -> bytes:
    """Return the bytes once they are the ones the reference names, else raise.

    Size is checked with the digest because a reader sized its buffers from it, and an
    encoding or algorithm this build cannot verify is refused rather than trusted.
    """
    if (digest := _DIGESTS.get(reference.digest_algorithm)) is None:
        raise ContentHydrationError(
            f"unsupported digest algorithm {reference.digest_algorithm}"
        )
    if reference.content_encoding != CONTENT_ENCODING_IDENTITY:
        raise ContentHydrationError(
            f"unsupported content encoding {reference.content_encoding}"
        )
    if len(data) != reference.size_bytes:
        raise ContentHydrationError(
            f"hydrated content is {len(data)} bytes for a reference naming "
            f"{reference.size_bytes}"
        )
    if digest(data) != reference.content_digest:
        raise ContentHydrationError(
            f"hydrated content digest mismatch for {reference.content_digest}"
        )
    return data


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
    """Scoped put-if-absent writes and digest-verified hydration."""

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
