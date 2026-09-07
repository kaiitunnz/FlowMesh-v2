"""Caller-neutral capture of a resident boundary's raw request into worker custody."""

from shared.harness import HarnessResult, HarnessResultKind
from shared.resident.wire import resident_request_digest

from .request_store import ResidentRequestStore


def capture_resident_request(
    store: ResidentRequestStore, task_id: str, result: HarnessResult
) -> HarnessResult:
    """Keep a resident request worker-private and emit only its digest.

    The raw request is recorded in resident custody keyed by
    ``(task_id, call_correlation)``, read back only by the origin worker's resident
    driver, and stripped from the boundary that crosses to control. Both the
    agent-episode and the service-leaf episode capture through this one path.
    """
    req = result.request
    assert result.kind is HarnessResultKind.BOUNDARY
    assert req is not None and req.request_payload is not None
    assert req.call_correlation is not None
    store.put(task_id, req.call_correlation, req.request_payload)
    stripped = req.model_copy(
        update={
            "request_payload": None,
            "request_digest": resident_request_digest(req.request_payload),
        }
    )
    return result.model_copy(update={"request": stripped})
