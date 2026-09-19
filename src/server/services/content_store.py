"""A filesystem-backed content store for reference-backed outcomes.

The store holds worker-materialized content as immutable, content-addressed objects
under a per-scope partition, plus an idempotency index so a re-drive under the same
fabric ``idempotency_key`` resolves the first materialization rather than writing a
second object. It stores opaque bytes and metadata only; it never assembles content into
orchestration state. Writes come from the worker over the content router; the server
never originates a materialization. Isolation is the partition: a read is scoped to the
authorization scope its caller was admitted for, so it never reaches another's content.

It backs many scopes at once: the router binds one per authenticated request rather than
the store holding one.
"""

from pathlib import Path

from shared.content import OCTET_STREAM, ContentReference, FilesystemObjectBacking
from shared.content.filesystem import safe_segment
from shared.outcome import OutcomeManifest
from shared.utils.atomic import atomic_write_bytes


class ServerContentStore:
    """A per-scope, content-addressed immutable store under a local directory."""

    def __init__(self, root: Path) -> None:
        self._objects = FilesystemObjectBacking(root)

    def _idem_path(self, scope: str | None, idempotency_key: str) -> Path:
        return self._objects.scope_path(scope, "idem") / (
            f"{safe_segment(idempotency_key)}.json"
        )

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        """Store bytes in a scope and return the reference naming them."""
        return self._objects.write(scope, data, media_type=media_type)

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        path = self._idem_path(scope, idempotency_key)
        if not path.exists():
            return None
        return OutcomeManifest.model_validate_json(path.read_text())

    def materialize(
        self,
        scope: str,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
        provenance: str | None = None,
    ) -> OutcomeManifest:
        if (found := self.find(scope, idempotency_key)) is not None:
            return found
        manifest = OutcomeManifest(
            content=self.write(scope, data, media_type=media_type),
            provenance=provenance,
            idempotency_key=idempotency_key,
        )
        atomic_write_bytes(
            self._idem_path(scope, idempotency_key),
            manifest.model_dump_json().encode(),
            if_absent=True,
        )
        return manifest

    def read(self, scope: str, digest: str) -> bytes:
        return self._objects.read(scope, digest)

    def fetch(self, reference: ContentReference) -> bytes:
        return self.read(reference.authorization_scope, reference.content_digest)
