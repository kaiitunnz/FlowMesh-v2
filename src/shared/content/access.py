"""The authorization a worker reaches the shared content store with.

Direct access to the store is its own thing, separate from both the reference naming an
object and the grant that opens one peer's copy of it. Control mints a
``ContentStoreAccessGrant`` for one dispatched task, in one authorization scope, bound
to the worker incarnation running it and expiring shortly after; the grant says what
access was given and is not itself secret. The material that actually opens the
backend travels
beside it as a ``ScopedContentCredential`` over the worker's authenticated attachment,
and goes nowhere else: not into the ledger, a message, a manifest, a relay frame, a
capsule, an artifact, or a log.

The grant covers a scope rather than a reference, so one issuance serves everything a
task reads and writes, and the scope is therefore the widest the store itself will let
that task reach. It carries no list, delete, or binding operation. Which references a
task may actually use is still decided above, by the consumer bindings control checks
before it hands one over.
"""

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

ACCESS_GRANT_VERSION = 1


class ContentOperationKind(StrEnum):
    """What a grant permits against the shared store, and nothing more."""

    READ = "read"
    WRITE = "write"


class ContentStoreAccessGrant(BaseModel):
    """One task's authorization to reach one scope's content in the shared store.

    ``backend_policy_version`` and ``scope_policy_epoch`` are the policy generations the
    access was cut against, so a rotation fences what was issued before it. The grant
    holds no credential: it describes access, and the material is delivered beside it.
    """

    model_config = ConfigDict(frozen=True)

    grant_id: str
    version: int = ACCESS_GRANT_VERSION
    task_id: str
    authorization_scope: str
    subject: str
    subject_generation: int
    operations: tuple[ContentOperationKind, ...]
    backend_policy_version: str = ""
    scope_policy_epoch: int = 0
    expires_at_epoch: float

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) > self.expires_at_epoch

    def permits(self, operation: ContentOperationKind) -> bool:
        return operation in self.operations


class ScopedContentCredential(BaseModel):
    """Opaque backend session material for one access grant.

    Whatever the backend needs to be opened under the grant's scope — session keys for
    an S3-compatible store, a projected identity for a shared filesystem. It is never
    rendered, persisted, or carried anywhere but the worker attachment it is delivered
    over, so it has no representation and no place in any durable record.
    """

    model_config = ConfigDict(frozen=True)

    material: dict[str, str] = Field(default_factory=dict, repr=False)


class ContentStoreAccess(BaseModel):
    """What control relays to a worker: the grant, and the material that opens it."""

    model_config = ConfigDict(frozen=True)

    grant: ContentStoreAccessGrant
    credential: ScopedContentCredential = Field(repr=False)


__all__ = [
    "ACCESS_GRANT_VERSION",
    "ContentOperationKind",
    "ContentStoreAccess",
    "ContentStoreAccessGrant",
    "ScopedContentCredential",
]
