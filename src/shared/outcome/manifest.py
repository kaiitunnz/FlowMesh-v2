"""The bounded, immutable reference form of a materialized invocation outcome.

A large or streamed provider/service result is materialized into durable, content-
addressed content by the producing worker; only this bounded manifest travels back to
the control plane and lands on the durable boundary. It is the outcome's binding to
content the fabric already names: the reference says which immutable bytes, and the
manifest says they are this finalization's. It carries no bearer URL and no worker-local
path — content is fetched over an authorized fabric data path.

Isolation is the reference's own authorization scope: a resume admitted for that scope
hydrates the content and one admitted for another cannot. ``idempotency_key`` is the
fabric identity the materialization deduplicates under; it names a producer
finalization, never the content.
"""

from pydantic import BaseModel, ConfigDict

from shared.content import ContentReference


class OutcomeManifest(BaseModel):
    """An outcome finalization's binding to durably materialized content."""

    model_config = ConfigDict(frozen=True)

    content: ContentReference
    provenance: str | None = None  # opaque producer tag; audit only, never parsed
    idempotency_key: str | None = None
