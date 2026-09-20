"""How this worker reaches the shared durable store.

Access is taken per authorization scope: the factory here is the seam where a
deployment decides what a worker may reach, and it hands back a store that reaches one
scope's content. The deployment-configured factory opens the store with the credential
the worker was provisioned with, which is the whole fleet's; a factory that mints access
per scope narrows that to the scope alone without anything above it changing.
"""

import logging
from pathlib import Path

from shared.content import (
    OCTET_STREAM,
    FabricObjectStore,
    FilesystemObjectBacking,
    ScopedObjectStore,
)
from shared.content.reference import ContentReference
from shared.content.s3_store import S3ObjectStore

from ..config import ObjectStoreConfig


class SharedFilesystemObjectStore(FabricObjectStore):
    """Content objects on a filesystem every worker in the fleet shares.

    The store a deployment gets by mounting one durable filesystem — CephFS, NFS — at
    the same path on every node. The layout is the digest-indexed one the fabric uses
    everywhere else, so a worker's own copies and the shared source of truth differ only
    in which directory they are in.
    """

    def __init__(self, root: Path) -> None:
        self._objects = FilesystemObjectBacking(root)

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._objects.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._objects.read(
            reference.authorization_scope, reference.content_digest
        )


class DeploymentScopedObjectStore:
    """Opens the shared store with the credential this worker was provisioned with.

    One credential reaches every scope's content, so the isolation a reader sees is the
    one the deployment's own store policy enforces.
    """

    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store

    def for_scope(self, scope: str) -> FabricObjectStore:
        return self._store


def build_shared_store(
    cfg: ObjectStoreConfig, logger: logging.Logger
) -> ScopedObjectStore | None:
    """The shared durable store this deployment runs, or None when it runs none."""
    match cfg.backend:
        case "s3":
            return DeploymentScopedObjectStore(_s3_store(cfg))
        case "filesystem":
            return DeploymentScopedObjectStore(
                SharedFilesystemObjectStore(cfg.filesystem_root)
            )
        case _:
            logger.warning("unknown content store backend %s", cfg.backend)
            return None


def _s3_store(cfg: ObjectStoreConfig) -> S3ObjectStore:
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url or None,
        aws_access_key_id=cfg.access_key or None,
        aws_secret_access_key=cfg.secret_key or None,
        region_name=cfg.region,
        config=Config(signature_version="s3v4"),
    )
    return S3ObjectStore(client, cfg.bucket, prefix=cfg.prefix)
