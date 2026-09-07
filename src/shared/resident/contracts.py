"""Claim-bound admission fences and the replica engine endpoint.

Central control mints the fences from authoritative claim facts; the origin worker
carries them on the data path and the replica worker's claim gate validates them before
engine delivery. They carry no route or credential — the network plane resolves the path
and the replica reaches its co-located engine.
"""

from pydantic import BaseModel, ConfigDict, Field


class ReplicaEndpoint(BaseModel):
    """The reachable address of a materialized replica.

    ``api_key`` never reaches a workflow; it is held out of the durable snapshot
    (``exclude=True``) so no credential is persisted in cleartext, and is re-attached
    from a live probe on rehydrate. ``base_url`` is OpenAI-compatible for the inference
    family. ``interface`` selects the engine route the replica serves (``chat`` or
    ``embedding``).
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key: str | None = Field(default=None, exclude=True)
    protocol: str = "openai"
    interface: str = "chat"


class AdmissionHandoff(BaseModel):
    """A claim-bound pre-``ACCEPTED`` bootstrap fence for one reserved claim.

    A ``RESERVED`` claim authorizes one bootstrap delivery that reaches the selected
    replica incarnation's resident sidecar and obtains an engine enqueue ack. It binds
    the tenant-scoped invocation subject, the fabric ``idm-*`` request identity, the
    selected replica incarnation and listener generation, and an expiry. The origin
    worker carries the resolved route alongside this handoff; the replica claim gate
    validates these bindings and trusts that only the authorized origin reaches its
    route. It never carries the raw engine endpoint or credential. For an adapter-bound
    invocation it also names the adapter to load into a replica slot and select on the
    request; the adapter rides the per-claim handoff, not the base-keyed replica.
    """

    model_config = ConfigDict(frozen=True)

    token: str
    claim_id: str
    invocation_id: str
    idempotency_key: str | None = None
    family: str
    tenant: str | None = None
    origin_id: str | None = None
    replica_id: str
    incarnation: int
    listener_generation: int = 0
    expires_at: str | None = None
    adapter_name: str | None = None
    adapter_source: str | None = None


class RouteAuthorization(BaseModel):
    """The immutable post-``ACCEPTED`` fence for one accepted-claim response stream.

    Issued only after the engine enqueue acknowledgement, it authorizes the response
    stream, cancellation, and backpressure for a single tenant-scoped invocation. It
    stamps no path and is never refreshed. The replica claim gate validates it per
    stream and rejects it once any bound fence — expiry, replica incarnation, listener
    generation, subject, claim, invocation, or request identity — no longer holds. A
    permitted reissue is a fresh successor claim under the same invocation, so the claim
    fence alone rejects a superseded authorization. It carries no bearer credential.
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str
    invocation_id: str
    idempotency_key: str | None = None
    tenant: str | None = None
    origin_id: str | None = None
    replica_id: str
    incarnation: int
    listener_generation: int = 0
    expires_at: str | None = None
