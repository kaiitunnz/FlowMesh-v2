"""The ten ``mediated_op`` frame kinds, each proven for arrival.

Three carriers (``permit``, ``resident_handoff``, ``resident_frame``) must reach the
worker-side router carrying their traceparent (or ``tp``, for ``resident_frame``); the
other seven must carry nothing, ever -- adding a carrier to one of them would be a
misattribution (``resident_sidecar_bind`` in particular is replica-lifecycle, not
invocation-scoped, and sounds like it should carry one).

Every kind here is driven through the real ``TaskListener._handle_message`` rebuild of
the Redis pub/sub JSON, or the ``enqueue_local`` bypass ``resident_frame`` actually
uses, and then through the real gRPC ``Struct`` round trip -- the two places a model
round-trip test would miss a drop. A test that only asserts ``MediatedOpMessage``
round-trips through Pydantic is not a gate here; the assertion is the dict that would
reach ``Runner._route_mediated_op`` or ``ResidentLaneHost.route``.

The kind set is re-derived from the worker-side router
(``runner.py::_route_mediated_op`` plus ``resident/lane_host.py::route``), not from the
call sites -- ``resident/service.py`` passes ``frame_kind`` positionally, so grepping
for ``frame_kind=`` finds only four of the ten.
"""

import asyncio
import threading
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from server.supervisor.services.task_listener import TaskListener
from server.utils.helpers import TSQueue
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.schemas.command import MediatedOpMessage

_TP = "00-11111111111111111111111111111111-2222222222222222-01"
_WORKER = "wkr-1"


def _struct_from_payload(payload: dict[str, Any]) -> Struct:
    struct = Struct()
    struct.update(payload)
    return struct


def _payload_from_struct(struct: Struct) -> dict[str, Any]:
    return MessageToDict(struct, preserving_proto_field_name=True)


def _through_grpc_struct(payload: dict[str, Any]) -> dict[str, Any]:
    """The real gRPC Struct round trip: JSON-shaped dict -> proto Struct -> dict."""
    return _payload_from_struct(_struct_from_payload(payload))


def _through_task_listener(frame_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The real Redis pub/sub JSON rebuild every non-``resident_frame`` kind crosses.

    ``_handle_message`` schedules the enqueue via ``run_coroutine_threadsafe``, which
    needs a loop actually running on another thread to execute -- exactly the
    production shape, reproduced here rather than faked.
    """
    listener = TaskListener.__new__(TaskListener)
    listener._qs = {_WORKER: TSQueue()}

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        listener._loop = loop
        data = MediatedOpMessage(
            worker_id=_WORKER, frame_kind=frame_kind, payload=payload
        ).model_dump(mode="json")
        listener._handle_message(data)
        rebuilt = asyncio.run_coroutine_threadsafe(
            listener._qs[_WORKER].get(), loop
        ).result(timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    assert rebuilt["kind"] == "mediated_op"
    assert rebuilt["frame_kind"] == frame_kind
    return rebuilt["payload"]


def _arrived(frame_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The exact payload dict that would reach the worker-side router, both layers
    crossed."""
    if frame_kind == "resident_frame":
        # enqueue_local bypasses TaskListener._handle_message's JSON rebuild entirely
        # (the node-local resident bridge shortcut), but still crosses the gRPC Struct
        # layer once StreamTasks drains the queue.
        return _through_grpc_struct(payload)
    rebuilt = _through_task_listener(frame_kind, payload)
    return _through_grpc_struct(rebuilt)


# The three carriers, and the key each names its stamp with in the arrived payload.
_CARRIERS: dict[str, str] = {
    "permit": "traceparent",
    "resident_handoff": "traceparent",
    "resident_frame": "tp",
}

# The seven NONE kinds, with a representative payload shape drawn from the real
# minting call sites in runtime.py / resident/service.py.
_NONE_PAYLOADS: dict[str, dict[str, Any]] = {
    "deny": {
        "agent_task_id": "tsk-agent",
        "call_correlation": "m0",
        "reason": "model turn egress denied",
    },
    "reap": {"agent_task_id": "tsk-agent", "call_correlation": "m0"},
    "resident_authorization": {
        "call_correlation": "call-1",
        "auth": {"claim_id": "scl-1"},
    },
    "resident_sidecar_bind": {
        "replica_id": "rpl-1",
        "incarnation": 1,
        "listener_generation": 1,
        "serve_task_id": None,
        "binding_generation": None,
        "engine": {"base_url": "http://engine/v1", "model": "m", "api_key": None},
    },
    "resident_reap": {"task_id": "tsk-1", "call_correlation": "call-1"},
    "resident_sidecar_reap": {"invocation_id": "inv-1"},
    "resident_adapter_unload": {"replica_id": "rpl-1", "adapter_name": "lora-1"},
}


def _permit_payload(traceparent: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "permit_id": "mop-1",
        "agent_task_id": "tsk-agent",
        "call_correlation": "m0",
        "interface": "search/v1",
        "subject": "search/v1",
        "invocation_id": "inv-1",
        "idempotency_key": "idm-1",
        "request_digest": "d",
        "target_id": "wkr-1",
        "target_generation": 3,
        "deadline_epoch": 2_000_000_000.0,
        "max_results": 1,
        "timeout_sec": 10.0,
        "result_char_cap": 4000,
    }
    if traceparent is not None:
        payload["traceparent"] = traceparent
    return payload


def _resident_handoff_payload(traceparent: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": "tsk-1",
        "call_correlation": "call-1",
        "session_id": "rly-1",
        "handoff": {"token": "hnd-1"},
        "carriage_plan": {"session_id": "rly-1"},
    }
    if traceparent is not None:
        payload["traceparent"] = traceparent
    return payload


def _resident_frame_payload(tp: str | None) -> dict[str, Any]:
    frame = RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        payload=b"body",
        tp=tp,
    )
    return frame.to_wire()


_CARRIER_BUILDERS = {
    "permit": _permit_payload,
    "resident_handoff": _resident_handoff_payload,
    "resident_frame": _resident_frame_payload,
}


def test_each_carrier_kind_arrives_with_its_stamp() -> None:
    for kind, key in _CARRIERS.items():
        payload = _CARRIER_BUILDERS[kind](_TP)
        arrived = _arrived(kind, payload)
        assert arrived.get(key) == _TP, f"{kind} lost its {key} stamp across the chain"


def test_each_carrier_kind_omits_the_stamp_when_telemetry_is_off() -> None:
    for kind, key in _CARRIERS.items():
        payload = _CARRIER_BUILDERS[kind](None)
        arrived = _arrived(kind, payload)
        assert key not in arrived, f"{kind} emitted a {key} field with telemetry off"


def test_every_none_kind_never_carries_a_traceparent_or_tp_key() -> None:
    assert set(_NONE_PAYLOADS) | set(_CARRIERS) == {
        "permit",
        "deny",
        "reap",
        "resident_handoff",
        "resident_authorization",
        "resident_sidecar_bind",
        "resident_reap",
        "resident_sidecar_reap",
        "resident_adapter_unload",
        "resident_frame",
    }, "the kind set must match the worker-side router exactly (ten kinds)"
    for kind, payload in _NONE_PAYLOADS.items():
        arrived = _arrived(kind, payload)
        assert "traceparent" not in arrived, f"{kind} must never carry a traceparent"
        assert "tp" not in arrived, f"{kind} must never carry a tp field"


def test_resident_sidecar_bind_specifically_carries_nothing() -> None:
    # The one kind most likely to be stamped by mistake: it sounds like it opens
    # flowmesh.engine_request, but it is replica-lifecycle (bound once, serving many
    # invocations from many workflows) and must never be invocation-scoped.
    payload = _NONE_PAYLOADS["resident_sidecar_bind"]
    arrived = _arrived("resident_sidecar_bind", payload)
    assert "traceparent" not in arrived
    assert "tp" not in arrived


def test_the_worker_side_router_recognizes_every_kind_this_test_covers() -> None:
    """Cross-check against the real router so a kind renamed or removed there fails
    here instead of silently going untested."""
    import inspect

    source = inspect.getsource(__import__("worker.runner", fromlist=["Runner"]).Runner)
    for kind in ("permit", "deny", "reap"):
        assert f'"{kind}"' in source, f"runner.py no longer routes {kind!r}"

    from worker.resident.lane_host import ResidentLaneHost

    route_source = inspect.getsource(ResidentLaneHost.route)
    for kind in (
        "resident_handoff",
        "resident_authorization",
        "resident_sidecar_bind",
        "resident_reap",
        "resident_sidecar_reap",
        "resident_adapter_unload",
        "resident_frame",
    ):
        assert f'"{kind}"' in route_source, f"lane_host.route no longer routes {kind!r}"
