import json
import logging
from collections.abc import Callable

from shared.schemas.event import TaskEvent, serialize_event

from ..clients.redis import (
    TASK_EVENT_STREAM_KEY,
    TASK_EVENT_STREAM_MAXLEN,
    SyncRedisClient,
)


class TaskEventPublisher:
    """Publishes a task event control originates onto the durable task-event stream.

    The event monitor handles it as it does a worker's report: in stream order, and
    again when a handling's durable write fails. An event the stream does not take is
    applied directly through the fallback, when one is set.
    """

    def __init__(self, redis_client: SyncRedisClient, logger: logging.Logger) -> None:
        self._redis = redis_client
        self._logger = logger
        self._fallback: Callable[[TaskEvent], None] | None = None

    def set_fallback(self, apply: Callable[[TaskEvent], None]) -> None:
        """Set the handler that applies an event directly when its publish fails."""
        self._fallback = apply

    def publish(self, event: TaskEvent) -> None:
        try:
            payload = json.dumps(serialize_event(event), ensure_ascii=False)
            self._redis.xadd_telemetry(
                TASK_EVENT_STREAM_KEY,
                {"payload": payload},
                maxlen=TASK_EVENT_STREAM_MAXLEN,
            )
            return
        except Exception as exc:
            self._logger.error(
                "Failed to publish %s for %s: %s", event.type, event.task_id, exc
            )
        if self._fallback is None:
            return
        try:
            self._fallback(event)
        except Exception as exc:
            self._logger.error(
                "Failed to apply %s for %s directly: %s", event.type, event.task_id, exc
            )


__all__ = ["TaskEventPublisher"]
