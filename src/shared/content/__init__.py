"""Immutable, content-addressed objects the fabric stores within one scope."""

from .access import (
    ContentOperationKind,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ScopedContentCredential,
)
from .config import BACKEND_FILESYSTEM, BACKEND_S3, ObjectStoreConfig
from .filesystem import FilesystemObjectBacking, SharedFilesystemObjectStore
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
    ContentUnavailable,
    FabricObjectStore,
    ScopedObjectStore,
    content_digest,
    reference_for,
    verify_content,
)

__all__ = [
    "BACKEND_FILESYSTEM",
    "BACKEND_S3",
    "ObjectStoreConfig",
    "CONTENT_ENCODING_IDENTITY",
    "OCTET_STREAM",
    "ContentHydrationError",
    "ContentHydrationGrant",
    "ContentOperation",
    "ContentOperationKind",
    "ContentReference",
    "ContentStoreAccess",
    "ContentStoreAccessGrant",
    "ContentStoreError",
    "ContentUnavailable",
    "DigestAlgorithm",
    "FabricObjectStore",
    "FilesystemObjectBacking",
    "ScopedContentCredential",
    "SharedFilesystemObjectStore",
    "ScopedObjectStore",
    "GrantRejection",
    "HolderGrantGate",
    "content_digest",
    "reference_for",
    "verify_content",
]
