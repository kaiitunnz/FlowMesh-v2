"""Immutable, content-addressed objects the fabric stores within one scope."""

from .filesystem import FilesystemObjectBacking
from .grant import (
    ContentHydrationGrant,
    ContentOperation,
    GrantRejection,
    HolderGrantGate,
)
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
    "ContentHydrationGrant",
    "ContentOperation",
    "ContentReference",
    "ContentStoreError",
    "DigestAlgorithm",
    "FabricObjectStore",
    "FilesystemObjectBacking",
    "GrantRejection",
    "HolderGrantGate",
    "content_digest",
    "reference_for",
    "verify_content",
]
