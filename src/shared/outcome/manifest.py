"""The bounded, immutable reference form of a materialized invocation outcome.

A large or streamed provider/service result is materialized into durable, content-
addressed content by the producing worker; only this bounded manifest travels back to
the control plane and lands on the durable boundary. It records the content identity
and the metadata a resumed worker needs to hydrate and verify the value, plus the
fabric idempotency key the materialization dedups under. It carries no bearer URL and
no worker-local path: content is addressed by ``(access.tenant, content_digest)`` and
fetched over an authorized fabric data path.
"""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


def content_digest(data: bytes) -> str:
    """The immutable content identity: a hex sha256 over the materialized bytes."""
    return hashlib.sha256(data).hexdigest()


class OutcomeAccessBinding(BaseModel):
    """The tenant/owner scope a hydration request must satisfy to read the content."""

    model_config = ConfigDict(frozen=True)

    tenant: str | None = None
    owner_subject: str | None = None


class OutcomeManifest(BaseModel):
    """A bounded, immutable reference to durably materialized outcome content."""

    model_config = ConfigDict(frozen=True)

    content_digest: str
    size_bytes: int
    media_type: str
    encoding: str | None = None
    provenance: str | None = None  # opaque producer tag; audit only, never parsed
    idempotency_key: str | None = None
    access: OutcomeAccessBinding = Field(default_factory=OutcomeAccessBinding)
