"""The worker-facing fabric content-store contract for reference-backed outcomes.

A worker materializes a completed outcome into content-addressed, immutable content and
reports its manifest; a resumed worker hydrates and verifies the manifest's reference
before it injects the value. Over the fabric's immutable objects this adds one thing: a
fabric-idempotency index, so a re-drive under the same ``idempotency_key`` finds the
already-materialized content rather than rerunning a sampled producer.
"""

from abc import abstractmethod

from shared.content import FabricObjectStore

from .manifest import OutcomeManifest


class FabricContentStore(FabricObjectStore):
    """Outcome finalization over the fabric's immutable objects."""

    @abstractmethod
    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        """The manifest already materialized under this idempotency key, or None."""

    @abstractmethod
    def materialize(
        self,
        scope: str,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
    ) -> OutcomeManifest:
        """Find-or-commit the content for an idempotency key and return its manifest.

        A prior materialization under the same key returns its manifest without a
        second write, so a re-drive never reruns the sampled producer.
        """
