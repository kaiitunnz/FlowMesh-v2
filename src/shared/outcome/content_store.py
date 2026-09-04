"""The fabric content-store contract for reference-backed outcomes.

A worker materializes a completed outcome into content-addressed, immutable content and
reports its manifest; a resumed worker hydrates and verifies the manifest before it
injects the value. The store is content-addressed by ``(tenant, content_digest)`` and
keeps a fabric-idempotency index so a re-drive under the same ``idempotency_key`` finds
the already-materialized content rather than rerunning a sampled producer.

The spool primitive is streaming-shaped: ``append`` advances a durable write cursor and
``finalize`` commits the immutable object only once the cursor covers the content. A
single-shot producer appends once and finalizes; a streamed producer appends
incrementally and gates each transport acknowledgement on the returned cursor.
"""

from abc import ABC, abstractmethod

from .manifest import OutcomeAccessBinding, OutcomeManifest, content_digest


class ContentStoreError(RuntimeError):
    """A content-store write, read, or finalize failed."""


class OutcomeHydrationError(ContentStoreError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


class OutcomeSpool(ABC):
    """A durable, idempotency-scoped write spool for one materialization."""

    @abstractmethod
    def append(self, data: bytes) -> int:
        """Append bytes durably; return the write cursor (total bytes committed)."""

    @abstractmethod
    def finalize(
        self,
        *,
        media_type: str,
        provenance: str | None,
        access: OutcomeAccessBinding,
    ) -> OutcomeManifest:
        """Commit the spooled content as an immutable object and return its manifest."""


class FabricContentStore(ABC):
    """A content-addressed, tenant-scoped immutable content store."""

    @abstractmethod
    def find(self, tenant: str | None, idempotency_key: str) -> OutcomeManifest | None:
        """The manifest already materialized under this idempotency key, or None."""

    @abstractmethod
    def open_spool(self, tenant: str | None, idempotency_key: str) -> OutcomeSpool:
        """Open the durable spool for a materialization under this idempotency key."""

    @abstractmethod
    def read(self, tenant: str | None, digest: str) -> bytes:
        """Fetch content by digest for a tenant, raising on a missing or denied read."""

    def materialize(
        self,
        tenant: str | None,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
        provenance: str | None = None,
        owner_subject: str | None = None,
    ) -> OutcomeManifest:
        """Find-or-finalize the content for an idempotency key in one shot.

        A prior materialization under the same key returns its manifest without a
        second write, so a re-drive never reruns the sampled producer.
        """
        if (found := self.find(tenant, idempotency_key)) is not None:
            return found
        spool = self.open_spool(tenant, idempotency_key)
        spool.append(data)
        return spool.finalize(
            media_type=media_type,
            provenance=provenance,
            access=OutcomeAccessBinding(tenant=tenant, owner_subject=owner_subject),
        )

    def hydrate(self, manifest: OutcomeManifest) -> bytes:
        """Fetch and digest-verify the manifest's content before it is injected."""
        data = self.read(manifest.access.tenant, manifest.content_digest)
        if content_digest(data) != manifest.content_digest:
            raise OutcomeHydrationError(
                f"hydrated content digest mismatch for {manifest.content_digest}"
            )
        return data
