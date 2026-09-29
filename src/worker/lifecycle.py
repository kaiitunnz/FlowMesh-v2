# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.content import ContentReference
from shared.schemas.event import TaskFailureKind
from shared.schemas.worker import SSHLimits, WorkerCapabilities
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from shared.utils.time import now_iso

from .egress import PendingEgressRequestStore
from .power import PowerMonitor
from .resident import ResidentRequestStore
from .supervisor_client import SupervisorClient

if TYPE_CHECKING:
    from .content import WorkerContentPlane
    from .model_turn import ResponsesFacade

# The wait an unregister gets for its event stream even once the stop budget is spent;
# the budget leaves this much before the stop kills the worker.
_UNREGISTER_FLOOR_SEC = 2.0


class Lifecycle:
    def __init__(
        self,
        client: SupervisorClient,
        hb_sec: int,
        hb_ttl_sec: int,
        hb_file: Path,
        cost_per_hour: float,
        power_monitor: PowerMonitor | None = None,
    ):
        self.client = client
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.hb_file = hb_file
        self.cost_per_hour = cost_per_hour
        self.power_monitor = power_monitor or PowerMonitor()
        self.pending_egress_requests = PendingEgressRequestStore()
        self.resident_requests = ResidentRequestStore()
        # The worker-local Responses facade held Codex episodes run their model turns
        # through, built by the runner once the worker id is known and read by the
        # agent-episode executor to bind a codex adapter.
        self.responses_facade: ResponsesFacade | None = None
        # This worker's content plane: started once the worker id is known, read by
        # whatever reaches fabric content, and stopped when the worker shuts down.
        self.content_plane: WorkerContentPlane | None = None
        self._stop_event = threading.Event()
        self._started_ts: float | None = None
        # What this worker last reported: its status and the dispatch it concerns.
        # Heartbeats repeat it, so a report the registry missed or took out of order
        # is restored within one heartbeat.
        self._status_lock = threading.Lock()
        self._status = WorkerStatus.STARTING
        self._dispatch_id: str | None = None
        self._draining = threading.Event()

    @property
    def worker_id(self) -> str:
        return self.client.worker_id

    def _metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
            metrics["uptime_sec"] = uptime
            metrics["accrued_cost_usd"] = (self.cost_per_hour / 3600.0) * uptime
        try:
            la = os.getloadavg()
            metrics["loadavg"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
        except Exception:
            pass
        try:
            power_sample = self.power_monitor.sample()
        except Exception:
            power_sample = None
        if power_sample:
            metrics["power"] = power_sample
        try:
            power_summary = self.power_monitor.summary()
        except Exception:
            power_summary = None
        if power_summary:
            metrics["power_summary"] = power_summary
            energy_total = power_summary.get("estimated_energy_kwh")
            if isinstance(energy_total, (int, float)):
                metrics["estimated_energy_kwh"] = energy_total
        return metrics

    def start(
        self,
        env: dict[str, Any],
        hardware: WorkerHardware,
        capabilities: WorkerCapabilities,
        ssh_limits: SSHLimits | None,
        tags: list[str],
    ):
        self._started_ts = time.time()
        try:
            initial_power = self.power_monitor.sample()
        except Exception:
            initial_power = None
        self.client.register(
            status=WorkerStatus.STARTING,
            started_at=now_iso(),
            pid=os.getpid(),
            env=env,
            hardware=hardware,
            capabilities=capabilities,
            ssh_limits=ssh_limits,
            tags=tags,
            cost_per_hour=self.cost_per_hour,
            power_metrics=initial_power,
        )
        self.client.start()
        self._report(WorkerStatus.IDLE, None, {})
        self.client.on_event_stream_ready(self._report_again)
        self._touch_hb_file()
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _hb_loop(self):
        while not self._stop_event.is_set():
            metrics = self._metrics()
            try:
                # Under the status lock, so no heartbeat carries a status older than
                # a report already sent.
                with self._status_lock:
                    self.client.heartbeat(
                        ttl_sec=self.hb_ttl_sec,
                        metrics=metrics,
                        status=self._status,
                        dispatch_id=self._dispatch_id,
                    )
            except Exception:
                pass
            self._touch_hb_file()
            self._stop_event.wait(self.hb_sec)

    def set_busy(self, task_id: str) -> None:
        self._report(
            WorkerStatus.BUSY, self.client.dispatch_id(task_id), {"task_id": task_id}
        )

    def set_idle(self, task_id: str) -> None:
        self._report(
            WorkerStatus.IDLE, self.client.dispatch_id(task_id), {"last_task": task_id}
        )

    def begin_draining(self) -> None:
        """Refuse every later status report; safe from a signal handler."""
        self._draining.set()

    def set_draining(self) -> None:
        """Report the worker busy for as long as it runs, so it takes no further task
        while it shuts down."""
        self._draining.set()
        with self._status_lock:
            self._report_locked(WorkerStatus.BUSY, self._dispatch_id, {})

    def _report_again(self) -> None:
        """Report the last status again, which an outage may have dropped."""
        with self._status_lock:
            self._report_locked(self._status, self._dispatch_id, {})

    def _report(
        self, status: WorkerStatus, dispatch_id: str | None, extra: dict[str, Any]
    ) -> None:
        with self._status_lock:
            if not self._draining.is_set():
                self._report_locked(status, dispatch_id, extra)

    def _report_locked(
        self, status: WorkerStatus, dispatch_id: str | None, extra: dict[str, Any]
    ) -> None:
        self._status, self._dispatch_id = status, dispatch_id
        try:
            self.client.set_status(status, extra, dispatch_id)
        except Exception:
            pass

    def set_failed(
        self,
        task_id: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        retryable: bool = True,
        failure_kind: TaskFailureKind | None = None,
        unavailable_inputs: tuple[ContentReference, ...] = (),
    ):
        try:
            self.client.task_failed(
                task_id,
                error=error,
                metadata=metadata,
                retryable=retryable,
                failure_kind=failure_kind,
                unavailable_inputs=unavailable_inputs,
            )
        except Exception:
            pass

    def set_succeeded(self, task_id: str, metadata: dict[str, Any] | None = None):
        try:
            self.client.task_succeeded(task_id, metadata=metadata)
        except Exception:
            pass

    def set_cancelled(self, task_id: str, metadata: dict[str, Any] | None = None):
        try:
            self.client.task_cancelled(task_id, metadata=metadata)
        except Exception:
            pass

    def notify_task_update(self, task_id: str, payload: dict[str, Any]) -> None:
        try:
            self.client.task_update(task_id, payload)
        except Exception:
            pass

    def notify_task_started(
        self,
        task_id: str,
        task_type: str | None,
        dispatched_at: str | None,
        started_at: str,
    ) -> None:
        try:
            self.client.task_started(
                task_id,
                task_type=task_type,
                dispatched_at=dispatched_at,
                started_at=started_at,
            )
        except Exception:
            pass

    def stop(self) -> None:
        self.client.stop()

    def held_boundaries(self) -> list[tuple[str, str]]:
        """The boundaries whose raw requests this worker holds, by task and call; each
        is dropped once control commits its outcome."""
        return (
            self.pending_egress_requests.occurrences()
            + self.resident_requests.occurrences()
        )

    def start_content_plane(self, plane: "WorkerContentPlane | None") -> None:
        """Own the worker's content plane from here to shutdown."""
        self.content_plane = plane
        if plane is not None:
            plane.start()

    def shutdown(self, graceful: bool, deadline: float | None = None) -> None:
        """Unregister the worker; `graceful` marks a shutdown it was asked for, and
        `deadline` is the monotonic time by which it must have unregistered."""

        def left() -> float | None:
            return None if deadline is None else max(0.0, deadline - time.monotonic())

        self._stop_event.set()
        if self.content_plane is not None:
            # Before unregistering: draining the lane cancels the transfers it serves,
            # and those frames leave over the attachment unregistering closes.
            try:
                remaining = left()
                self.content_plane.stop(10.0 if remaining is None else remaining)
            except Exception:
                pass
        try:
            self.power_monitor.sample()
        except Exception:
            pass
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
        accrued_cost = (
            (self.cost_per_hour / 3600.0) * uptime if uptime is not None else None
        )
        summary = self.power_monitor.summary()
        try:
            self.client.unregister(
                graceful,
                cost_per_hour=self.cost_per_hour,
                uptime_sec=uptime,
                accrued_cost_usd=accrued_cost,
                power_summary=summary,
                timeout=(
                    None
                    if (remaining := left()) is None
                    else max(remaining, _UNREGISTER_FLOOR_SEC)
                ),
            )
        except Exception:
            pass
        self.client.shutdown()
        self._remove_hb_file()

    def _touch_hb_file(self) -> None:
        hb_file = self.hb_file
        hb_file.parent.mkdir(parents=True, exist_ok=True)
        hb_file.touch()

    def _remove_hb_file(self) -> None:
        self.hb_file.unlink(missing_ok=True)
