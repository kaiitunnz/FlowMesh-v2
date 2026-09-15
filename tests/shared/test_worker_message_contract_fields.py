"""The inference contract a dispatched message carries survives its wire encoding.

A worker task message deduplicates its strings on the way out and restores them on the
way in, so the contract fields ride that encoding as nested models rather than as opaque
payloads a reader has to decode itself.
"""

import json
import pickle
from typing import Any

from shared.inference import (
    CanonicalInferenceContract,
    CanonicalInferenceInputSource,
    CanonicalInferenceRequest,
    InferenceSourceKind,
    InputResolutionBinding,
    UpstreamProvenance,
)
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.worker_message import WorkerTaskMessage

_CONTRACT = CanonicalInferenceContract(
    model="Qwen/Qwen3-4B",
    source=CanonicalInferenceInputSource(
        kind=InferenceSourceKind.UPSTREAM,
        node="up",
        path="items.output",
        max_items=4,
    ),
    params={"temperature": 0.7, "max_tokens": 512, "top_k": -1},
)

_BINDING = InputResolutionBinding(
    source_digest="src",
    resolver_version="1",
    request_digest="req",
    cardinality=2,
    upstream=(UpstreamProvenance(node="up", content_digest="c1"),),
    projected_output_tokens=1024,
)

_REQUEST = CanonicalInferenceRequest(
    model="Qwen/Qwen3-4B",
    prompts=("hello", "goodbye"),
    params={"temperature": 0.7, "max_tokens": 512},
)


def _message(**fields: Any) -> WorkerTaskMessage:
    return WorkerTaskMessage(
        task_id="tsk-1",
        workflow_id="wfl-1",
        owner_id="own-1",
        task=TaskEnvelopeStrict.model_validate(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "spec": {
                    "taskType": "inference",
                    "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
                    "data": {"type": "list", "expr": "up.items.output", "max_items": 4},
                },
            }
        ),
        assigned_worker="wrk-1",
        dispatched_at="2026-01-01T00:00:00Z",
        **fields,
    )


def _round_trip(msg: WorkerTaskMessage) -> WorkerTaskMessage:
    wire = json.loads(json.dumps(msg.model_dump(mode="json")))
    return WorkerTaskMessage.model_validate(wire)


def test_the_contract_fields_survive_the_wire_encoding() -> None:
    msg = _message(
        declared_contract=_CONTRACT,
        recorded_resolution=_BINDING,
        resolved_contract=_REQUEST,
    )
    back = _round_trip(msg)

    assert back.declared_contract == _CONTRACT
    assert back.recorded_resolution == _BINDING
    assert back.resolved_contract == _REQUEST
    assert back.declared_contract is not None
    assert back.declared_contract.source.kind is InferenceSourceKind.UPSTREAM
    assert back.resolved_contract is not None
    assert back.resolved_contract.prompts == ("hello", "goodbye")


def test_a_message_deduplicates_the_strings_its_contract_carries() -> None:
    # The dedup encoding reaches into the nested models rather than stopping at them.
    wire = _message(declared_contract=_CONTRACT, resolved_contract=_REQUEST).model_dump(
        mode="json"
    )

    assert set(wire) == {"content", "data"}
    assert wire["data"]["declared_contract"]["source"]["node"].keys() == {
        "__dedup_ref__"
    }
    assert wire["data"]["resolved_contract"]["prompts"][0].keys() == {"__dedup_ref__"}
    assert "hello" in wire["content"].values()


def test_a_message_declaring_no_contract_round_trips() -> None:
    back = _round_trip(_message())

    assert back.declared_contract is None
    assert back.recorded_resolution is None
    assert back.resolved_contract is None


def test_a_worker_resolving_its_contract_keeps_it_across_the_executor_boundary() -> (
    None
):
    # The origin worker sets the resolved request on the message it hands its executor,
    # which runs in another process.
    msg = _round_trip(_message(declared_contract=_CONTRACT))
    msg.resolved_contract = _REQUEST

    handed = pickle.loads(pickle.dumps(msg))
    assert handed.resolved_contract == _REQUEST
    assert handed.declared_contract == _CONTRACT
