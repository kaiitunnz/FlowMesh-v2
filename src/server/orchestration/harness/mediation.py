"""The mapping between a worker-emitted boundary request and the engine's envelope.

An adapter runs on a worker and emits a worker-safe :class:`BoundaryRequest`; the engine
consumes the server's durable :class:`BoundaryEvent`, which mints the fabric-assigned
identity and carries the continuation. The production path lives in the task runtime,
which routes a worker's returned boundary through this translation into the engine.
"""

from shared.harness import BoundaryRequest

from ..state import BoundaryEvent


def to_boundary_event(
    request: BoundaryRequest, *, continuation: str | None = None
) -> BoundaryEvent:
    """Lift a worker-emitted request into the engine's durable boundary envelope.

    The fabric-assigned identity (activation, idempotency key, invocation id, injection
    target, denial, settled value) stays the engine's to mint; only the continuation
    capsule is supplied here, from the step's returned capsule.
    """
    return BoundaryEvent(
        kind=request.kind,
        call_correlation=request.call_correlation,
        interface=request.interface,
        child_ref=request.child_ref,
        child_region_ref=request.child_region_ref,
        request_payload=request.request_payload,
        state_ref=request.state_ref,
        continuation=continuation,
    )
