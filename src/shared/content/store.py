"""The immutable-object core every reference-backed fabric value is stored over.

An object is bytes named by their own digest, written put-if-absent under the writing
principal's scope and read back only under that same scope. The core knows nothing about
what the bytes mean: a caller that needs a meaning wraps this in its own facade and
carries its own reference type, so two kinds of value can share one identity and one
backing without either becoming a name for the other.
"""

import hashlib
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class ObjectWriteAck(BaseModel):
    """What a store acknowledges a write with: the identity it stored the bytes under.

    A writer pairs this with the meaning it already holds to form its own reference; the
    ack itself names no scope and no media type, so it is not one.
    """

    model_config = ConfigDict(frozen=True)

    content_digest: str
    size_bytes: int


def content_digest(data: bytes) -> str:
    """The immutable content identity: a hex sha256 over the bytes."""
    return hashlib.sha256(data).hexdigest()


class ContentStoreError(RuntimeError):
    """A content-store write, read, or finalize failed."""


class ContentHydrationError(ContentStoreError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


class FabricObjectStore(ABC):
    """Put-if-absent by digest, authorized read, and digest-verified hydration."""

    @abstractmethod
    def put_object(self, data: bytes, *, media_type: str) -> str:
        """Store the bytes under the caller's scope and return their digest.

        A digest already present is left as it is: identical bytes, so the first write
        stands and a repeat costs nothing.
        """

    @abstractmethod
    def read(self, digest: str) -> bytes:
        """Fetch content by digest, raising on a missing or unauthorized read."""

    def hydrate_object(self, digest: str) -> bytes:
        """Fetch and verify content against the digest that names it."""
        data = self.read(digest)
        if content_digest(data) != digest:
            raise ContentHydrationError(
                f"hydrated content digest mismatch for {digest}"
            )
        return data
