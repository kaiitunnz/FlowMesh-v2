"""An in-memory ``FabricContentStore`` for exercising reference-backed outcomes."""

from shared.outcome import (
    ContentStoreError,
    FabricContentStore,
    OutcomeAccessBinding,
    OutcomeHydrationError,
    OutcomeManifest,
    OutcomeSpool,
    content_digest,
)


class InMemoryContentStore(FabricContentStore):
    """A content-addressed, tenant-scoped store backed by process memory.

    ``fail_finalize`` simulates an origin crash after acknowledged bytes but before the
    manifest commits, so a retry re-materializes the same content under its key.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[str | None, str], bytes] = {}
        self._idm: dict[tuple[str | None, str], OutcomeManifest] = {}
        self.fail_finalize = False
        self.write_count = 0

    def find(self, tenant: str | None, idempotency_key: str) -> OutcomeManifest | None:
        return self._idm.get((tenant, idempotency_key))

    def open_spool(self, tenant: str | None, idempotency_key: str) -> OutcomeSpool:
        return _InMemorySpool(self, tenant, idempotency_key)

    def read(self, tenant: str | None, digest: str) -> bytes:
        data = self._objects.get((tenant, digest))
        if data is None:
            raise OutcomeHydrationError(
                f"no content for {digest} under tenant {tenant}"
            )
        return data


class _InMemorySpool(OutcomeSpool):
    def __init__(
        self, store: InMemoryContentStore, tenant: str | None, idempotency_key: str
    ) -> None:
        self._store = store
        self._tenant = tenant
        self._idm = idempotency_key
        self._buf = bytearray()
        self._cursor = 0

    def append(self, data: bytes) -> int:
        self._buf.extend(data)
        self._cursor += len(data)
        return self._cursor

    def finalize(
        self,
        *,
        media_type: str,
        provenance: str | None,
        access: OutcomeAccessBinding,
    ) -> OutcomeManifest:
        if self._store.fail_finalize:
            raise ContentStoreError("simulated crash before manifest commit")
        data = bytes(self._buf)
        digest = content_digest(data)
        self._store._objects[(access.tenant, digest)] = data
        self._store.write_count += 1
        manifest = OutcomeManifest(
            content_digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            provenance=provenance,
            idempotency_key=self._idm,
            access=access,
        )
        self._store._idm[(access.tenant, self._idm)] = manifest
        return manifest
