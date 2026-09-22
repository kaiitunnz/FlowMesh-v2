"""The shared durable store on a filesystem every worker mounts.

One of the two backings a deployment can give the fabric's content: a durable
filesystem — CephFS, NFS — mounted at the same path on every node. The layout is the
digest-indexed one the fabric uses everywhere else, so a worker's own copies and the
shared source of truth differ only in which directory they are in.

Isolation here is the filesystem's. A deployment gets per-scope separation only where
projected identities, mounts, or ACLs enforce it, because a directory name under one
shared worker identity does not.
"""

from pathlib import Path

from shared.content import (
    OCTET_STREAM,
    FabricObjectStore,
    FilesystemObjectBacking,
)
from shared.content.reference import ContentReference


class SharedFilesystemObjectStore(FabricObjectStore):
    """Content objects on a filesystem every worker in the fleet shares."""

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
