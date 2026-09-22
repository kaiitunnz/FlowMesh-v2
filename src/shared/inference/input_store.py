"""Storing and retrieving the one resolved request an inference leaf's inputs prepared.

The worker that resolves a leaf's source writes the request it materialized as an
immutable object and reports a bounded reference to it. Whichever embodiment runs next
hydrates that object and verifies it against the digest naming it, so the run issues the
exact request the resolution produced rather than re-reading upstream syntax. The object
is named by an ordinary content reference; what makes it this leaf's input is the
binding recorded beside it, so nothing here turns a stored object into a declared
result or an invocation outcome.
"""

from pydantic import BaseModel, ConfigDict

from shared.content import ContentHydrationError, ContentReference, FabricObjectStore

from .codec import ResolvedCanonicalInferenceRequest
from .source import InputResolutionBinding

RESOLVED_INPUT_MEDIA_TYPE = "application/json"


class ResolvedInputMaterialization(BaseModel):
    """One preparation's whole durable result: how it resolved, and to what.

    The two travel together because either alone is unusable — a binding without its
    request cannot be run, and a reference without its binding records nothing about
    what produced it — so they commit as one fact and an object written ahead of that
    commit belongs to no resolution.
    """

    model_config = ConfigDict(frozen=True)

    binding: InputResolutionBinding
    reference: ContentReference


def write_resolved_input(
    store: FabricObjectStore, scope: str, resolved: ResolvedCanonicalInferenceRequest
) -> ContentReference:
    """Store a resolved request and return the reference a later run hydrates it by."""
    return store.write(
        scope,
        resolved.model_dump_json().encode(),
        media_type=RESOLVED_INPUT_MEDIA_TYPE,
    )


def hydrate_resolved_input(
    store: FabricObjectStore, reference: ContentReference
) -> ResolvedCanonicalInferenceRequest:
    """Fetch and verify the request a reference names, before anything issues it.

    The media type is checked with the digest: an object of another type in the same
    scope is not this leaf's request, whatever its bytes hash to.
    """
    if reference.media_type != RESOLVED_INPUT_MEDIA_TYPE:
        raise ContentHydrationError(
            f"the object at {reference.content_digest} is not a resolved request"
        )
    data = store.hydrate(reference)
    try:
        return ResolvedCanonicalInferenceRequest.model_validate_json(data)
    except ValueError as exc:
        raise ContentHydrationError(
            f"the object at {reference.content_digest} is not a resolved request"
        ) from exc
