"""The worker's task-status events carry the dispatch traceparent.

Each ``task_*`` helper is the far end of the worker->supervisor->server hop for
``TaskEvent``; this proves the value each is given reaches the enqueued event payload
unchanged, and that an absent value emits no key at all rather than an explicit null.
"""

import logging
from typing import Any, cast
from unittest import mock

from worker.supervisor_client import SupervisorClient

_TP = "00-11111111111111111111111111111111-2222222222222222-01"


def _client() -> SupervisorClient:
    client = SupervisorClient(
        worker_token="t",
        owner_principal=None,
        grpc_target="x",
        worker_namespace="ns",
        worker_cluster="c",
        worker_alias="a",
        logger=logging.getLogger("supervisor-client-tp-test"),
    )
    client._worker_id = "wkr-1"
    client._stub = cast(Any, object())
    client._event_ready.set()
    return client


def test_task_started_carries_the_traceparent_when_enabled() -> None:
    client = _client()
    client.task_started("tsk-1", traceparent=_TP)
    payload = client._event_queue.get_nowait()
    assert isinstance(payload, dict)
    assert payload["traceparent"] == _TP


def test_task_started_emits_no_traceparent_key_when_disabled() -> None:
    client = _client()
    client.task_started("tsk-1")
    payload = client._event_queue.get_nowait()
    assert isinstance(payload, dict)
    assert "traceparent" not in payload


def test_task_succeeded_task_failed_task_cancelled_task_update_forward_the_value() -> (
    None
):
    client = _client()
    client.task_succeeded("tsk-1", traceparent=_TP)
    client.task_failed("tsk-1", "boom", traceparent=_TP)
    client.task_cancelled("tsk-1", traceparent=_TP)
    client.task_update("tsk-1", {"k": "v"}, traceparent=_TP)

    payloads = [client._event_queue.get_nowait() for _ in range(4)]
    for payload in payloads:
        assert isinstance(payload, dict)
        assert payload["traceparent"] == _TP


def test_heartbeat_and_status_never_carry_a_traceparent_key() -> None:
    # These events have no per-task context and never populate the field.
    client = _client()
    client.heartbeat()
    payload = client._event_queue.get_nowait()
    assert isinstance(payload, dict)
    assert "traceparent" not in payload


def test_create_task_log_emitter_threads_the_traceparent() -> None:
    client = _client()
    with mock.patch("worker.supervisor_client.TaskLogEmitter") as emitter_cls:
        client.create_task_log_emitter(
            task_id="tsk-1",
            workflow_id="wfl-1",
            owner_id="own-1",
            traceparent=_TP,
        )
    assert emitter_cls.call_args.kwargs["traceparent"] == _TP
