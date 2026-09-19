"""An in-memory ``FabricContentStore`` for exercising reference-backed outcomes."""

from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    ContentStoreError,
    reference_for,
)
from shared.outcome import FabricContentStore, OutcomeManifest


class InMemoryContentStore(FabricContentStore):
    """A content-addressed store backed by process memory.

    ``fail_finalize`` simulates an origin crash after acknowledged bytes but before the
    manifest commits, so a retry re-materializes the same content under its key.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}
        self._idm: dict[tuple[str, str], OutcomeManifest] = {}
        self.fail_finalize = False
        self.write_count = 0

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        key = (scope, reference.content_digest)
        if key not in self._objects:
            self._objects[key] = data
            self.write_count += 1
        return reference

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        return self._idm.get((scope, idempotency_key))

    def materialize(
        self, scope: str, idempotency_key: str, data: bytes, *, media_type: str
    ) -> OutcomeManifest:
        if (found := self._idm.get((scope, idempotency_key))) is not None:
            return found
        if self.fail_finalize:
            raise ContentStoreError("simulated crash before manifest commit")
        manifest = OutcomeManifest(
            content=self.write(scope, data, media_type=media_type),
            idempotency_key=idempotency_key,
        )
        self._idm[(scope, idempotency_key)] = manifest
        return manifest

    def fetch(self, reference: ContentReference) -> bytes:
        data = self._objects.get(
            (reference.authorization_scope, reference.content_digest)
        )
        if data is None:
            raise ContentHydrationError(f"no content for {reference.content_digest}")
        return data
