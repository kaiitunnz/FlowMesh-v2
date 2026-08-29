"""The worker-emittable boundary vocabulary shared by the harness contract.

A harness adapter runs on a worker, which cannot import the server. These are the
boundary kinds, denial kinds, and the request an adapter emits at a boundary — the
subset the worker produces, distinct from the server's durable boundary envelope that
adds the fabric-assigned identity (activation, idempotency key, invocation id,
injection target, continuation).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BoundaryEventKind(StrEnum):
    """A fabric-relevant event an operator's boundary signature may emit.

    ``SPAWN_SEAL`` closes an agent's child-init producer: a ``SPAWN`` never seals it, so
    an agent yields ``SPAWN_SEAL`` when it will issue no more children.
    """

    INVOCATION = "invocation"
    SPAWN = "spawn"
    SPAWN_SEAL = "spawn_seal"
    YIELD = "yield"
    EXTERNAL_EFFECT = "external_effect"
    STATE_ACCESS = "state_access"


class DenialKind(StrEnum):
    """Why a definitive dynamic authorization failed, distinct from admission outcomes.

    A denial creates neither a child activation nor a resident claim, and is separate
    from quota/rate/capacity/transport outcomes.
    """

    AUTHORITY = "authority"  # interface outside the effective grant's invoke/delegate
    POLICY = "policy"  # blocked by the pinned policy envelope


class BoundaryRequest(BaseModel):
    """One boundary an adapter emits before it executes, as worker-safe data.

    The adapter names the request kind and its subject (a tool/model ``interface``, a
    ``child_region_ref`` role, or a declared ``state_ref``) plus the stable
    ``call_correlation`` that survives a re-drive and the opaque ``request_payload``.
    The fabric-assigned identity and the durable continuation are the server's to mint;
    the adapter never sets them.
    """

    model_config = ConfigDict(frozen=True)

    kind: BoundaryEventKind
    call_correlation: str | None = None
    interface: str | None = None
    child_region_ref: str | None = None
    child_ref: str | None = None
    request_payload: str | None = None
    state_ref: str | None = None
