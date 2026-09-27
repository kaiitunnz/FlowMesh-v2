"""Control-originated task events ride the durable task-event stream."""

import json
import logging
from unittest.mock import MagicMock

from server.clients.redis import TASK_EVENT_STREAM_KEY
from server.services.task_events import TaskEventPublisher
from shared.schemas.event import TaskEvent

_EVENT = TaskEvent(type="TASK_FAILED", task_id="tsk-1", worker_id="wkr-1", error="x")


def _publisher(redis: MagicMock) -> tuple[TaskEventPublisher, list[TaskEvent]]:
    applied: list[TaskEvent] = []
    publisher = TaskEventPublisher(redis, logging.getLogger("task-events"))
    publisher.set_fallback(applied.append)
    return publisher, applied


def test_an_event_is_published_onto_the_task_event_stream() -> None:
    redis = MagicMock()
    publisher, applied = _publisher(redis)

    publisher.publish(_EVENT)

    (key, fields), _ = redis.xadd_telemetry.call_args
    assert key == TASK_EVENT_STREAM_KEY
    assert json.loads(fields["payload"])["task_id"] == "tsk-1"
    assert applied == []


def test_an_event_the_stream_does_not_take_is_applied_directly() -> None:
    redis = MagicMock()
    redis.xadd_telemetry.side_effect = ConnectionError("telemetry redis down")
    publisher, applied = _publisher(redis)

    publisher.publish(_EVENT)

    assert applied == [_EVENT]


def test_a_direct_apply_that_fails_does_not_raise() -> None:
    redis = MagicMock()
    redis.xadd_telemetry.side_effect = ConnectionError("telemetry redis down")
    publisher = TaskEventPublisher(redis, logging.getLogger("task-events"))

    def _fail(event: TaskEvent) -> None:
        raise ConnectionError("control redis down")

    publisher.set_fallback(_fail)

    publisher.publish(_EVENT)
