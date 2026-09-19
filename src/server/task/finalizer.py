"""Workflow completion: the single path that closes a finished workflow.

A workflow closes once — its ``flowmesh.workflow`` span is emitted and its log stream
is sealed — whichever way its last task settled. A worker-reported terminal reaches
here through the task-event handler; a terminal the control plane settles without
publishing an event reaches here through the runtime's own notification.
"""

import json
import logging
import threading

from ..clients.redis import (
    SyncRedisClient,
    workflow_key,
    workflow_log_closed_key,
    workflow_log_stream_key,
    workflow_tasks_key,
)
from ..orchestration.telemetry import WorkflowSpanEmitter
from ..schemas.logs import LogEvent
from ..utils.time import now_iso, ts_to_iso
from .models import WorkflowSettlement
from .runtime import TaskRuntime


class WorkflowFinalizer:
    """Closes each completed workflow exactly once, serialized across its callers.

    Callers name a workflow; the finalizer decides whether it is complete and, if so,
    emits its span and closes its log stream. Serializing the whole decision keeps two
    callers from both reading an unclosed workflow and both closing it.
    """

    def __init__(
        self,
        redis_client: SyncRedisClient,
        runtime: TaskRuntime,
        logger: logging.Logger,
        workflow_span_emitter: WorkflowSpanEmitter,
        log_stream_ttl_sec: int = 0,
    ) -> None:
        self._redis_client = redis_client
        self._runtime = runtime
        self._logger = logger
        self._workflow_span_emitter = workflow_span_emitter
        self._log_stream_ttl_sec = max(0, int(log_stream_ttl_sec))
        self._finalize_lock = threading.Lock()
        self._requests: set[str] = set()
        self._requests_cv = threading.Condition()

    def request(self, workflow_id: str) -> None:
        """Ask for a workflow to be closed when it is complete.

        Cheap and thread-safe: the runtime calls this from its own settle paths, and
        the finalizer's thread does the work. A request for a workflow that is not
        complete, or is already closed, resolves to nothing.
        """
        if not workflow_id:
            return
        with self._requests_cv:
            self._requests.add(workflow_id)
            self._requests_cv.notify_all()

    def run(self, stop_event: threading.Event, poll_interval: float = 0.5) -> None:
        """Drain requests until stopped."""
        while not stop_event.is_set():
            self.drain(poll_interval)

    def drain(self, timeout: float = 0.0) -> None:
        """Close every workflow requested so far.

        The queue is handed over before any workflow is examined, so nothing here
        holds it while reading the runtime -- a caller settling a task inside the
        scheduler lock must never wait on this thread.
        """
        for workflow_id in self._take_requests(timeout):
            self.close(workflow_id)

    def _take_requests(self, timeout: float) -> set[str]:
        with self._requests_cv:
            self._requests_cv.wait_for(lambda: bool(self._requests), timeout)
            pending, self._requests = self._requests, set()
            return pending

    def close(self, workflow_id: str) -> None:
        """Close the workflow if it is complete; otherwise do nothing."""
        if not workflow_id:
            return
        with self._finalize_lock:
            try:
                settlement = self._settlement_if_complete(workflow_id)
            except Exception as exc:
                self._logger.debug(
                    "Failed to evaluate workflow completion for %s: %s",
                    workflow_id,
                    exc,
                )
                return
            if settlement is None:
                return
            self._emit_workflow_span(workflow_id, settlement)
            self._close_log_stream(workflow_id)

    def close_task_workflow(self, task_id: str) -> None:
        """Close the workflow owning a task that just settled."""
        record = self._runtime.get_record(task_id)
        if record is not None:
            self.close(record.workflow_id)

    def _settlement_if_complete(self, workflow_id: str) -> WorkflowSettlement | None:
        """The workflow's settlement when it is complete and unclosed, else None."""
        if self._redis_client.exists(workflow_log_closed_key(workflow_id)):
            return None
        if self._redis_client.set_members(workflow_tasks_key(workflow_id)):
            return None
        if not self._redis_client.exists(workflow_key(workflow_id)):
            return None
        settlement = self._runtime.workflow_settlement(workflow_id)
        return settlement if settlement.settled else None

    def _emit_workflow_span(
        self, workflow_id: str, settlement: WorkflowSettlement
    ) -> None:
        # Telemetry never decides whether the log stream closes: this reads the
        # workflow record and emits a span, and a failure in either must not strand a
        # completed workflow's stream open.
        try:
            submitted_at = self._runtime.workflow_submitted_at(workflow_id)
            # The workflow's own last finish, not the clock: this runs once per
            # workflow but may run again after a restart, and a durable end makes the
            # re-emitted span identical to the first.
            finished_ts = settlement.finished_ts
            if submitted_at is None or finished_ts is None:
                missing = [
                    name
                    for name, value in (
                        ("submission time", submitted_at),
                        ("finish time", finished_ts),
                    )
                    if value is None
                ]
                self._logger.warning(
                    "Omitting the workflow span for %s: no durable %s",
                    workflow_id,
                    " or ".join(missing),
                )
                return
            self._workflow_span_emitter.emit(
                workflow_id, submitted_at, ts_to_iso(finished_ts)
            )
        except Exception as exc:
            self._logger.debug(
                "Failed to emit the workflow span for %s: %s", workflow_id, exc
            )

    def _close_log_stream(self, workflow_id: str) -> None:
        event = LogEvent(
            ts=now_iso(),
            workflow_id=workflow_id,
            level="INFO",
            stream="system",
            source="server",
            message="Workflow log stream closed.",
        )
        payload = event.model_dump(exclude_none=True)
        payload["type"] = "LOG_STREAM_CLOSED"
        encoded = json.dumps(payload, ensure_ascii=False)
        stream_key = workflow_log_stream_key(workflow_id)
        closed_key = workflow_log_closed_key(workflow_id)
        try:
            self._redis_client.xadd_telemetry(
                stream_key,
                {"payload": encoded, "workflow_id": workflow_id},
            )
            self._redis_client.set_value(closed_key, "1")
            if self._log_stream_ttl_sec:
                self._redis_client.expire_telemetry(
                    stream_key, self._log_stream_ttl_sec
                )
                self._redis_client.expire(closed_key, self._log_stream_ttl_sec)
        except Exception as exc:
            self._logger.debug(
                "Failed to append log sentinel for workflow %s: %s", workflow_id, exc
            )
