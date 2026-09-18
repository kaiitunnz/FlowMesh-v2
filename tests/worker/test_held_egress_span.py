"""The held model turn's ``flowmesh.egress`` span and the parent its permit names.

A synchronous-turn-only backend drives its egress through ``egress_now`` on the
facade's own thread, with no ambient context to inherit, so the span's parent can only
come from the ``traceparent`` control stamped on the permit.
"""

import json
from pathlib import Path
from typing import Any

from shared.telemetry.config import TelemetryLevel
from shared.tools.contract import (
    MediatedOperationPermit,
    ToolOperationEnvelope,
    ToolOutcome,
)
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    ModelCompletion,
    ModelRequest,
    model_request_digest,
)
from shared.utils.ids import new_mediated_permit_id
from tests.worker.otel_support import fresh_worker_provider, worker_telemetry
from worker.egress import MediatedEgressSidecar, PendingEgressRequestStore
from worker.executors.mixins import _otel

_WORKER = "wkr-1"
_GEN = 3
_AGENT = "tsk-agent"
_CALL = "t0"
_WORKFLOW = "wfl-0102030405060708090a0b0c0d0e0f10"
_TRACE_ID = "0102030405060708090a0b0c0d0e0f10"
_PARENT_SPAN_ID = "00f1e2d3c4b5a697"
_TRACEPARENT = f"00-{_TRACE_ID}-{_PARENT_SPAN_ID}-01"
_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_REQUEST = ModelRequest(interface=MODEL_INTERFACE, url="http://up/v1", body=_BODY)


class _StubSyncEgress:
    interface = MODEL_INTERFACE

    def digest(self, request: Any) -> str:
        return model_request_digest(request.interface, request.url, request.body)

    def execute(
        self,
        envelope: ToolOperationEnvelope,
        request: Any,
        credential: str | None,
    ) -> ToolOutcome:
        raise AssertionError("a held turn never takes the asynchronous egress path")

    def complete(
        self,
        envelope: ToolOperationEnvelope,
        request: Any,
        credential: str | None,
    ) -> ModelCompletion:
        return ModelCompletion(content="a reply")


def _permit(traceparent: str | None) -> MediatedOperationPermit:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=_AGENT,
        call_correlation=_CALL,
        interface=MODEL_INTERFACE,
        subject=MODEL_INTERFACE,
        invocation_id="inv-1",
        idempotency_key="idm-1",
        request_digest=model_request_digest(MODEL_INTERFACE, "http://up/v1", _BODY),
        target_id=_WORKER,
        target_generation=_GEN,
        deadline_epoch=2_000_000_000.0,
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=1_000_000,
        traceparent=traceparent,
    )


def _egress_spans(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [row for row in rows if row["name"] == "flowmesh.egress"]


def _run_held_turn(spans_path: Path, traceparent: str | None) -> None:
    pending = PendingEgressRequestStore()
    pending.put(_AGENT, _CALL, _REQUEST)
    sidecar = MediatedEgressSidecar(
        pending_requests=pending,
        audience=lambda: (_WORKER, _GEN),
        egresses=(_StubSyncEgress(),),
        outcome_sink=lambda _outcome: None,
    )
    try:
        with _otel.task_trace_context(_WORKFLOW, spans_path):
            result = sidecar.egress_now(_permit(traceparent))
        assert isinstance(result, ModelCompletion)
    finally:
        sidecar.stop()


def test_a_held_turn_opens_an_egress_span_under_the_permits_parent(
    tmp_path: Path,
) -> None:
    spans_path = tmp_path / "held" / "spans.jsonl"
    with fresh_worker_provider(worker_telemetry(TelemetryLevel.FINE)):
        _run_held_turn(spans_path, _TRACEPARENT)

    spans = _egress_spans(spans_path)
    assert len(spans) == 1, "a held model turn must open its own egress span"
    span = spans[0]
    assert span["parent_id"] == f"0x{_PARENT_SPAN_ID}"
    assert span["context"]["trace_id"] == f"0x{_TRACE_ID}"


def test_a_held_turn_opens_no_egress_span_below_fine(tmp_path: Path) -> None:
    spans_path = tmp_path / "coarse" / "spans.jsonl"
    with fresh_worker_provider(worker_telemetry(TelemetryLevel.COARSE)):
        _run_held_turn(spans_path, _TRACEPARENT)

    assert _egress_spans(spans_path) == []
