"""Immutable, content-addressed objects the fabric stores within one scope."""

from .reference import (
    CONTENT_ENCODING_IDENTITY,
    OCTET_STREAM,
    ContentReference,
    DigestAlgorithm,
)
from .store import (
    ContentHydrationError,
    ContentStoreError,
    FabricObjectStore,
    content_digest,
    reference_for,
    verify_content,
)

__all__ = [
    "CONTENT_ENCODING_IDENTITY",
    "OCTET_STREAM",
    "ContentHydrationError",
    "ContentReference",
    "ContentStoreError",
    "DigestAlgorithm",
    "FabricObjectStore",
    "content_digest",
    "reference_for",
    "verify_content",
]
