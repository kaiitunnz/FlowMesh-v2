"""Which content an outcome finalization already produced.

A producer that may be re-driven must be able to tell whether its outcome was already
materialized, or a second drive re-runs a sampled model and settles a different result
under the same identity. That question is control-plane state: the index here maps a
finalization's ``idm-*`` to the reference it produced, and holds nothing else — never
the bytes, which live in the shared content store, and never a general name for content,
which an idempotency key is not.
"""

from shared.content import ContentReference
from shared.outcome import OutcomeManifest

from ..clients.redis import RedisClient


class FinalizationIndex:
    """The reference each outcome finalization committed, by scope and key."""

    def __init__(self, redis: RedisClient, *, ttl_sec: float = 0.0) -> None:
        self._rds = redis
        self._ttl = ttl_sec

    @staticmethod
    def _key(scope: str, idempotency_key: str) -> str:
        return f"ct:finalization:{scope}:{idempotency_key}"

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        raw = self._rds.sync.get(self._key(scope, idempotency_key))
        return OutcomeManifest.model_validate_json(raw) if raw else None

    def record(
        self,
        scope: str,
        idempotency_key: str,
        content: ContentReference,
        *,
        provenance: str | None = None,
    ) -> OutcomeManifest:
        """Bind a finalization to its content, or return the binding already there.

        The first binding stands: a re-drive that produced the same content records
        nothing new, and one that produced different content still settles as the
        first, so an outcome never changes after it has been committed.
        """
        key = self._key(scope, idempotency_key)
        if (existing := self.find(scope, idempotency_key)) is not None:
            return existing
        manifest = OutcomeManifest(
            content=content, provenance=provenance, idempotency_key=idempotency_key
        )
        self._rds.sync.set_value(key, manifest.model_dump_json())
        if self._ttl:
            self._rds.sync.expire(key, int(self._ttl))
        return manifest
