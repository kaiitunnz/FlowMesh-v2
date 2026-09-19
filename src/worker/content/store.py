"""The objects a worker holds for the fabric.

A worker that materializes content keeps the bytes itself and reports only the reference
naming them, so a payload reaches a consumer over an authorized transfer rather than
through a server. What the worker holds is therefore live fabric state, not a cache: it
serves a hydration the control plane granted, and it is the only holder of a new object
until another consumer materializes the same bytes in the same scope.

Reclamation is deliberately conservative. An object stays for as long as this worker
runs once it has been reported in a consumer's binding; only a write that never reached
one — an execution that died between writing the bytes and reporting them — is
reclaimed, and only after a grace period long enough that a slow report is not mistaken
for a lost one. Nothing here reads a binding or decides that a consumer is done with an
object: that is the control plane's, and in this slice no one does it.
"""

import time
from pathlib import Path

from shared.content import (
    OCTET_STREAM,
    ContentReference,
    FabricObjectStore,
    FilesystemObjectBacking,
)
from shared.content.filesystem import safe_segment
from shared.utils.atomic import atomic_write_bytes


class WorkerObjectStore(FabricObjectStore):
    """The content this worker wrote and serves, under a local directory."""

    def __init__(self, root: Path, *, orphan_grace_sec: float) -> None:
        self._objects = FilesystemObjectBacking(root)
        self._orphan_grace_sec = orphan_grace_sec

    @property
    def orphan_grace_sec(self) -> float:
        return self._orphan_grace_sec

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._objects.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._objects.read(
            reference.authorization_scope, reference.content_digest
        )

    def holds(self, reference: ContentReference) -> bool:
        return self._objects.holds(
            reference.authorization_scope, reference.content_digest
        )

    def bind(self, reference: ContentReference) -> None:
        """Record that a consumer's binding now names this object.

        Called once the worker has reported the reference, which is what makes the
        object reachable and so keeps it from being swept as an unbound write. The mark
        sits beside the object so a restarted worker does not sweep what it already
        reported.
        """
        atomic_write_bytes(self._bound_path(*self._key(reference)), b"", if_absent=True)

    def _bound_path(self, scope: str, digest: str) -> Path:
        return self._objects.scope_path(scope, "bound") / safe_segment(digest)

    @staticmethod
    def _key(reference: ContentReference) -> tuple[str, str]:
        return reference.authorization_scope, reference.content_digest

    def reclaim_orphans(self, *, in_transfer: frozenset[str] = frozenset()) -> int:
        """Drop writes no binding claims, and return how many went.

        An object is reclaimable only when every one of them holds: no binding named it,
        it is older than the grace period, and no transfer is serving it right now.
        """
        deadline = time.time() - self._orphan_grace_sec
        reclaimed = 0
        for scope, digest in list(self._objects.iter_objects()):
            if digest in in_transfer or self._bound_path(scope, digest).exists():
                continue
            written_at = self._objects.written_at(scope, digest)
            if written_at is None or written_at > deadline:
                continue
            self._objects.remove(scope, digest)
            reclaimed += 1
        return reclaimed
