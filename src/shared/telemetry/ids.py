"""Trace and span identity: the workflow-id bijection and derived span ids.

Every telemetry producer must derive the same ``trace_id`` for a given workflow, and a
worker-side child must be able to name a server-synthesized parent span that has not
been exported yet. Both properties come from making the ids pure functions of durable
ids rather than random or process-local values.
"""

import hashlib
import re
from enum import StrEnum

from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from shared.utils.ids import PREFIX_WORKFLOW

_HEX_ONLY = re.compile(r"[^0-9a-f]")


def workflow_to_trace_id_int(workflow_id: str) -> int:
    """Stable 128-bit trace id derived from the workflow id.

    Strips the ``wfl-`` prefix before hex extraction so the prefix's ``f``
    doesn't shift the bit pattern.
    """
    body = workflow_id.lower().removeprefix(f"{PREFIX_WORKFLOW}-")
    hex_only = _HEX_ONLY.sub("", body)
    if not hex_only:
        return 0
    return int(hex_only.zfill(32)[:32], 16)


class SpanIdKind(StrEnum):
    """The long-lived entity a derived span id names.

    Included in the hash input alongside the durable id, which is what prevents a work
    item and its sole attempt from colliding when they share a durable id prefix.
    """

    WORKFLOW = "workflow"
    ACTIVATION = "activation"
    WORK_ITEM = "work_item"
    ATTEMPT = "attempt"
    INVOCATION = "invocation"


def derived_span_id(kind: SpanIdKind, durable_id: str) -> int:
    """Deterministic, non-zero 64-bit span id for a long-lived durable entity.

    Two independent producers computing this for the same ``(kind, durable_id)`` reach
    the same id, which lets a worker-side child name a server-synthesized parent span
    before that span exists as an object. Only workflow, activation, work item, attempt
    and invocation spans are derived this way; everything else uses an ordinary random
    span id.
    """
    salt = b""
    while True:
        digest = hashlib.blake2b(
            f"{kind}:{durable_id}".encode() + salt, digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big")
        if value != 0:
            return value
        salt += b"\x00"


def trace_sampled(ratio: float, trace_id: int) -> bool:
    """Whether ``trace_id`` falls inside the configured sample.

    The decision reads the trace id alone, which every producer derives from the
    workflow id by the same function, so the root, a supervisor and a worker agree on
    whether a workflow is traced without coordinating -- and a trace is never half
    present. Delegates to the SDK's own ratio sampler rather than restating its bound.
    """
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    result = TraceIdRatioBased(ratio).should_sample(
        parent_context=None, trace_id=trace_id, name=""
    )
    return result.decision.is_sampled()
