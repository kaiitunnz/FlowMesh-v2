"""The bounded, immutable reference form of a materialized invocation outcome.

A large or streamed provider/service result is materialized into durable, content-
addressed content by the producing worker; only this bounded manifest travels back to
the control plane and lands on the durable boundary. It records the content identity
and the metadata a resumed worker needs to hydrate and verify the value, plus the
fabric idempotency key the materialization dedups under. It carries no bearer URL and
no worker-local path: content is fetched by digest over an authorized fabric data path.

Isolation is the content store's own: it partitions content by the authenticated
principal, so a resume authenticated as the same principal hydrates the content and a
different one cannot. ``tenant`` records that scope for audit; it is not a second gate.
"""

import hashlib

from pydantic import BaseModel, ConfigDict


def content_digest(data: bytes) -> str:
    """The immutable content identity: a hex sha256 over the materialized bytes."""
    return hashlib.sha256(data).hexdigest()


class OutcomeManifest(BaseModel):
    """A bounded, immutable reference to durably materialized outcome content."""

    model_config = ConfigDict(frozen=True)

    content_digest: str
    size_bytes: int
    media_type: str
    provenance: str | None = None  # opaque producer tag; audit only, never parsed
    idempotency_key: str | None = None
    tenant: str | None = None  # the principal scope the content was materialized under
