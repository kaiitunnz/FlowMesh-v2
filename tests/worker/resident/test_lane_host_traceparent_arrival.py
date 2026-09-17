"""The ``resident_handoff`` frame's ``traceparent`` key reaches the origin driver.

Its home is a payload dict, not a declared model field, so a model round-trip proves
nothing about arrival -- the frame is unpacked into ``ResidentOriginRequest`` field by
field. This drives that unpack through the real ``ResidentLaneHost._begin`` and asserts
the value on the object the origin driver receives, which is what
``flowmesh.transport.*`` opens its span from.
"""

from typing import Any
from unittest.mock import MagicMock

from worker.resident.lane_host import ResidentLaneHost
from worker.resident.origin_driver import ResidentOriginRequest

_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def _handoff_dict() -> dict[str, Any]:
    return {
        "token": "tok-1",
        "claim_id": "scl-1",
        "invocation_id": "inv-1",
        "family": "default",
        "replica_id": "rpl-1",
        "incarnation": 1,
    }


def _carriage_plan_dict() -> dict[str, Any]:
    return {"session_id": "ssn-1"}


def _build_host() -> ResidentLaneHost:
    return ResidentLaneHost(
        push_frame=lambda frame: None,
        report_ack=lambda ack: None,
        report_outcome=lambda outcome: None,
        content_store=None,
        peek_request=lambda task_id, call_correlation: "raw-request",
        delete_request=lambda task_id, call_correlation: None,
    )


def test_resident_handoff_traceparent_reaches_the_origin_request() -> None:
    host = _build_host()
    host.start()
    try:
        assert host._origin is not None
        captured = MagicMock()
        host._origin.begin = captured  # type: ignore[method-assign]

        host._begin(
            {
                "task_id": "tsk-1",
                "call_correlation": "call-1",
                "session_id": "ssn-1",
                "handoff": _handoff_dict(),
                "carriage_plan": _carriage_plan_dict(),
                "traceparent": _TRACEPARENT,
            }
        )

        captured.assert_called_once()
        request = captured.call_args.args[0]
        assert isinstance(request, ResidentOriginRequest)
        assert request.traceparent == _TRACEPARENT
    finally:
        host.stop()


def test_resident_handoff_with_no_traceparent_key_leaves_it_none() -> None:
    """Telemetry off upstream omits the key entirely (zero bytes on the wire)."""
    host = _build_host()
    host.start()
    try:
        assert host._origin is not None
        captured = MagicMock()
        host._origin.begin = captured  # type: ignore[method-assign]

        host._begin(
            {
                "task_id": "tsk-1",
                "call_correlation": "call-1",
                "session_id": "ssn-1",
                "handoff": _handoff_dict(),
                "carriage_plan": _carriage_plan_dict(),
            }
        )

        request = captured.call_args.args[0]
        assert request.traceparent is None
    finally:
        host.stop()
