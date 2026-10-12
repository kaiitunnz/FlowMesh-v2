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
    STREAM_CHUNK_BYTES,
    ContentHydrationError,
    ContentStoreError,
    ContentUnavailable,
    ContentVerifier,
    FabricObjectStore,
    ScopedObjectStore,
    WriteCancelled,
    content_digest,
    new_hasher,
    reference_for,
    verified_chunks,
    verify_content,
    write_verified,
)

__all__ = [
    "STREAM_CHUNK_BYTES",
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
    "ContentVerifier",
    "DigestAlgorithm",
    "FabricObjectStore",
    "FilesystemObjectBacking",
    "ScopedContentCredential",
    "SharedFilesystemObjectStore",
    "ScopedObjectStore",
    "GrantRejection",
    "HolderGrantGate",
    "WriteCancelled",
    "content_digest",
    "new_hasher",
    "reference_for",
    "verified_chunks",
    "verify_content",
    "write_verified",
]
