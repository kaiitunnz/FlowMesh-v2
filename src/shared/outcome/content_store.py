"""The worker-facing fabric content-store contract for reference-backed outcomes.

A worker materializes a completed outcome into content-addressed, immutable content and
reports its manifest; a resumed worker hydrates and verifies the manifest before it
injects the value. Over the fabric's immutable objects this adds one thing: a
fabric-idempotency index, so a re-drive under the same ``idempotency_key`` finds the
already-materialized content rather than rerunning a sampled producer. Both layers
partition content by the authenticated principal, so the worker names no tenant — its
own credential scopes every read and write.
"""

from abc import abstractmethod

from shared.content import ContentHydrationError, FabricObjectStore, content_digest

from .manifest import OutcomeManifest


class OutcomeHydrationError(ContentHydrationError):
    """Fetched content is missing, unauthorized, or fails digest verification."""


class FabricContentStore(FabricObjectStore):
    """Outcome finalization over the fabric's immutable objects."""

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

    def hydrate(self, manifest: OutcomeManifest) -> bytes:
        """Fetch and digest-verify the manifest's content before it is injected."""
        data = self.read(manifest.content_digest)
        if content_digest(data) != manifest.content_digest:
            raise OutcomeHydrationError(
                f"hydrated content digest mismatch for {manifest.content_digest}"
            )
        return data
