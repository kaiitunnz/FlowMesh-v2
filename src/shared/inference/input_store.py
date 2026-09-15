"""Storing and retrieving the one resolved request an inference leaf's inputs prepared.

The worker that resolves a leaf's source writes the request it materialized as an
immutable object and reports a bounded reference to it. Whichever embodiment runs next
hydrates that object and verifies it against the digest naming it, so the run issues the
exact request the resolution produced rather than re-reading upstream syntax. The
reference is its own type: it records an input, which is neither an invocation outcome
nor a declared result, and nothing here names one.
"""

from pydantic import BaseModel, ConfigDict, Field

from shared.content import ContentHydrationError, FabricObjectStore

from .codec import ResolvedCanonicalInferenceRequest
from .source import InputResolutionBinding

RESOLVED_INPUT_MEDIA_TYPE = "application/json"


class ResolvedInputReference(BaseModel):
    """A bounded, immutable reference to one worker-materialized resolved request.

    It carries the identity and the scope a holder needs to fetch and verify the bytes,
    and nothing that would let a reader reach them another way: no bearer URL and no
    worker-local path.
    """

    model_config = ConfigDict(frozen=True)

    content_digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = RESOLVED_INPUT_MEDIA_TYPE
    # The principal scope the object was written under; a read authenticates as it.
    scope: str | None = None


class ResolvedInputMaterialization(BaseModel):
    """One preparation's whole durable result: how it resolved, and to what.

    The two travel together because either alone is unusable — a binding without its
    request cannot be run, and a reference without its binding records nothing about
    what produced it — so they commit as one fact and an object written ahead of that
    commit belongs to no resolution.
    """

    model_config = ConfigDict(frozen=True)

    binding: InputResolutionBinding
    reference: ResolvedInputReference


def write_resolved_input(
    store: FabricObjectStore,
    resolved: ResolvedCanonicalInferenceRequest,
    *,
    scope: str | None = None,
) -> ResolvedInputReference:
    """Store a resolved request and return the reference a later run hydrates it by."""
    data = resolved.model_dump_json().encode()
    digest = store.put_object(data, media_type=RESOLVED_INPUT_MEDIA_TYPE)
    return ResolvedInputReference(
        content_digest=digest, size_bytes=len(data), scope=scope
    )


def hydrate_resolved_input(
    store: FabricObjectStore, reference: ResolvedInputReference
) -> ResolvedCanonicalInferenceRequest:
    """Fetch and verify the request a reference names, before anything issues it."""
    data = store.hydrate_object(reference.content_digest)
    try:
        return ResolvedCanonicalInferenceRequest.model_validate_json(data)
    except ValueError as exc:
        raise ContentHydrationError(
            f"the object at {reference.content_digest} is not a resolved request"
        ) from exc
