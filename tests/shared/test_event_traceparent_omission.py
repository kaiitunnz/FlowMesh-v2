"""An event without trace context puts no trace field on any wire.

Several paths re-serialize an event they are only relaying, so stripping the field at
one of them leaves the others emitting an explicit null -- and one of those fires per
relay frame. Dropping it in the serializer is what makes the absence hold everywhere,
including on a path added later.
"""

import json

from shared.schemas.event import TaskEvent, WorkerEvent, serialize_event

_TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_an_event_without_context_carries_no_trace_field() -> None:
    payload = serialize_event(TaskEvent(type="TASK_SUCCEEDED", task_id="tsk-1"))

    assert "traceparent" not in payload


def test_an_event_with_context_still_carries_it() -> None:
    payload = serialize_event(
        TaskEvent(type="TASK_SUCCEEDED", task_id="tsk-1", traceparent=_TP)
    )

    assert payload["traceparent"] == _TP


def test_a_relayed_event_stays_free_of_the_field() -> None:
    """A relay re-serializes what it decoded, so the omission survives a round trip."""
    original = WorkerEvent(type="WORKER_READY", worker_id="wkr-1")

    relayed = json.loads(json.dumps(serialize_event(original)))
    again = serialize_event(WorkerEvent.model_validate(relayed))

    assert "traceparent" not in relayed
    assert "traceparent" not in again
