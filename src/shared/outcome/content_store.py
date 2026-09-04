"""The worker-facing fabric content-store contract for reference-backed outcomes.

A worker materializes a completed outcome into content-addressed, immutable content and
reports its manifest; a resumed worker hydrates and verifies the manifest before it
injects the value. The store keeps a fabric-idempotency index so a re-drive under the
same ``idempotency_key`` finds the already-materialized content rather than rerunning a
sampled producer. The store partitions content by the authenticated principal, so the
worker names no tenant — its own credential scopes every read and write.
"""

from abc import ABC, abstractmethod

from .manifest import OutcomeManifest, content_digest


class ContentStoreError(RuntimeError):
    """A content-store write, read, or finalize failed."""


class OutcomeHydrationError(ContentStoreError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


class FabricContentStore(ABC):
    """A content-addressed, principal-scoped immutable content store."""

    @abstractmethod
    def find(self, idempotency_key: str) -> OutcomeManifest | None:
        """The manifest already materialized under this idempotency key, or None."""

    @abstractmethod
    def materialize(
        self,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
    ) -> OutcomeManifest:
        """Find-or-commit the content for an idempotency key and return its manifest.

        A prior materialization under the same key returns its manifest without a
        second write, so a re-drive never reruns the sampled producer.
        """

    @abstractmethod
    def read(self, digest: str) -> bytes:
        """Fetch content by digest, raising on a missing or unauthorized read."""

    def hydrate(self, manifest: OutcomeManifest) -> bytes:
        """Fetch and digest-verify the manifest's content before it is injected."""
        data = self.read(manifest.content_digest)
        if content_digest(data) != manifest.content_digest:
            raise OutcomeHydrationError(
                f"hydrated content digest mismatch for {manifest.content_digest}"
            )
        return data
