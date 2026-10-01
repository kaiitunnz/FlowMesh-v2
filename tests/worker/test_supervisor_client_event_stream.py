"""An event stream that closes keeps the events it had not sent for the next one."""

import threading
from typing import Any

from google.protobuf.json_format import MessageToDict

from shared.schemas.event import WorkerEvent
from tests.worker.test_supervisor_client_dispatch_id import _client


def _heartbeat(worker_id: str) -> WorkerEvent:
    return WorkerEvent(type="HEARTBEAT", worker_id=worker_id, ts="t")


def test_an_event_a_closed_stream_pulls_goes_to_the_next_stream() -> None:
    client = _client()
    client._shutdown.clear()
    closed = threading.Event()
    stale = client._event_messages(closed)
    # gRPC pulls a call's request iterator on its own thread, which is still waiting
    # on the queue when the call ends.
    pulled: list[Any] = []
    puller = threading.Thread(target=lambda: pulled.extend(stale), daemon=True)
    puller.start()
    closed.set()

    client._enqueue_event(_heartbeat("wkr-1"))
    puller.join(timeout=5)
    assert not puller.is_alive()

    client._event_queue.put(client._EVENT_SENTINEL)
    sent = list(client._event_messages(threading.Event()))

    assert pulled == []
    assert [
        MessageToDict(message.payload, preserving_proto_field_name=True)["type"]
        for message in sent
    ] == ["HEARTBEAT"]


def test_a_heartbeat_names_the_task_of_its_dispatch() -> None:
    client = _client()
    client._shutdown.clear()

    client.heartbeat(dispatch_id="dsp-1", task_id="tsk-1")
    client._event_queue.put(client._EVENT_SENTINEL)
    [sent] = list(client._event_messages(threading.Event()))

    event = MessageToDict(sent.payload, preserving_proto_field_name=True)
    assert (event["dispatch_id"], event["payload"]["task_id"]) == ("dsp-1", "tsk-1")
