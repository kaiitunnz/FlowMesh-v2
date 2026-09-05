"""An in-memory ``FabricContentStore`` for exercising reference-backed outcomes."""

from shared.outcome import (
    ContentStoreError,
    FabricContentStore,
    OutcomeHydrationError,
    OutcomeManifest,
    content_digest,
)


class InMemoryContentStore(FabricContentStore):
    """A content-addressed store backed by process memory for one principal scope.

    ``fail_finalize`` simulates an origin crash after acknowledged bytes but before the
    manifest commits, so a retry re-materializes the same content under its key.
    """

    def __init__(self, tenant: str | None = "local") -> None:
        self._tenant = tenant
        self._objects: dict[str, bytes] = {}
        self._idm: dict[str, OutcomeManifest] = {}
        self.fail_finalize = False
        self.write_count = 0

    def find(self, idempotency_key: str) -> OutcomeManifest | None:
        return self._idm.get(idempotency_key)

    def materialize(
        self, idempotency_key: str, data: bytes, *, media_type: str
    ) -> OutcomeManifest:
        if (found := self._idm.get(idempotency_key)) is not None:
            return found
        if self.fail_finalize:
            raise ContentStoreError("simulated crash before manifest commit")
        digest = content_digest(data)
        self._objects[digest] = data
        self.write_count += 1
        manifest = OutcomeManifest(
            content_digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            idempotency_key=idempotency_key,
            tenant=self._tenant,
        )
        self._idm[idempotency_key] = manifest
        return manifest

    def read(self, digest: str) -> bytes:
        data = self._objects.get(digest)
        if data is None:
            raise OutcomeHydrationError(f"no content for {digest}")
        return data
