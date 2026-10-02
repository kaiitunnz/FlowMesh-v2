# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.content import ContentReference
from shared.schemas.event import TaskFailureKind
from shared.schemas.worker import SSHLimits, WorkerCapabilities
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from shared.utils.time import now_iso

from .egress import PendingEgressRequestStore
from .gpu_availability import DeviceAvailability, GpuAvailabilityMonitor
from .power import PowerMonitor
from .resident import ResidentRequestStore
from .ssh_relay import SshEndpointRegistry, SshRelayLane
from .supervisor_client import SupervisorClient

if TYPE_CHECKING:
    from .content import WorkerContentPlane
    from .model_turn import ResponsesFacade

logger = logging.getLogger(__name__)

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
        gpu_monitor: GpuAvailabilityMonitor | None = None,
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
        # The loopback endpoints this worker's executors published for relaying, and
        # the lane that serves relayed SSH connections against them.
        self.ssh_endpoints = SshEndpointRegistry()
        self.ssh_relay: SshRelayLane | None = None
        self._stop_event = threading.Event()
        self._started_ts: float | None = None
        # What this worker last reported: its status, and the dispatch it concerns
        # with that dispatch's task. Heartbeats repeat it, so a report the registry
        # missed or took out of order is restored within one heartbeat.
        self._status_lock = threading.Lock()
        self._status = WorkerStatus.STARTING
        self._dispatch_id: str | None = None
        self._task_id: str | None = None
        self._draining = threading.Event()
        self._last_task_end = 0.0
        self._gpu_monitor = gpu_monitor
        self._abandon_running: Callable[[str | None], None] | None = None
        self._gpu_executor_probe: Callable[[], bool] | None = None
        if gpu_monitor is not None:
            cfg = gpu_monitor.config
            logger.info(
                "foreign-GPU detection on: a device is reported unavailable above "
                "%d MiB used with nothing of ours loaded (%d consecutive checks, "
                "%.0fs grace after a task)",
                cfg.threshold_mib,
                cfg.consecutive,
                cfg.grace_sec,
            )

    @property
    def worker_id(self) -> str:
        return self.client.worker_id

    def dispatch_for(self, task_id: str) -> str | None:
        """The dispatch a task's work runs for: the one running it, or the one a
        boundary it holds off-lane was captured under."""
        return (
            self.client.dispatch_id(task_id)
            or self.pending_egress_requests.dispatch_of(task_id)
            or self.resident_requests.dispatch_of(task_id)
        )

    def set_abandon_handler(self, abandon: Callable[[str | None], None]) -> None:
        """Register how the dispatch running when the worker re-registers is given
        up."""
        self._abandon_running = abandon

    def _on_reregistered(self, abandoned_dispatch: str | None) -> None:
        """Leave everything bound to the previous registration behind: the dispatch it
        ran, the requests its boundaries captured, and the content it held."""
        if (abandon := self._abandon_running) is not None:
            abandon(abandoned_dispatch)
        self.pending_egress_requests.clear()
        self.resident_requests.clear()
        if (plane := self.content_plane) is not None:
            plane.rebind(self.client.worker_id, self.client.incarnation)

    def set_gpu_executor_probe(self, probe: Callable[[], bool]) -> None:
        """Register a probe reporting whether a GPU-using executor is loaded; a
        reading taken while one is warm includes the worker's own model."""
        self._gpu_executor_probe = probe

    def gpu_availability(self) -> dict[str, DeviceAvailability]:
        """Per-device availability as this worker last reported it, the reading
        placement uses."""
        monitor = self._gpu_monitor
        return {} if monitor is None else monitor.snapshot()

    def live_gpu_availability(self) -> dict[str, DeviceAvailability]:
        """Per-device availability from the reading just taken only, so a caller that
        refuses work on it never refuses on a stale latch."""
        monitor = self._gpu_monitor
        return {} if monitor is None else monitor.live_snapshot()

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
        if (monitor := self._gpu_monitor) is not None:
            # Sent even when empty: an empty map is how the worker says its devices
            # are no longer known to be held, and omitting it would leave the
            # server's last reading latched with nothing able to clear it.
            metrics["gpu_availability"] = {
                uuid: {"available": device.available, "free_bytes": device.free_bytes}
                for uuid, device in monitor.snapshot().items()
            }
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
        self._report(WorkerStatus.IDLE, None, None, {})
        self.client.on_event_stream_ready(self._report_again)
        self.client.on_reregistered(self._on_reregistered)
        self._touch_hb_file()
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _hb_loop(self):
        while not self._stop_event.is_set():
            # Observed before the metrics, so the heartbeat carries this beat's
            # reading rather than the previous one's.
            try:
                self._observe_gpu()
            except Exception:
                logger.debug("GPU availability observation failed", exc_info=True)
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
                        task_id=self._task_id,
                    )
            except Exception:
                pass
            self._touch_hb_file()
            self._stop_event.wait(self.hb_sec)

    def _observe_gpu(self) -> None:
        """Feed the availability monitor one observation.

        A reading is trusted only when nothing of the worker's own can be in it: no
        task running, no GPU-using executor still warm, and past the grace window in
        which a finished task's subprocess may still be releasing memory. Before the
        runner registers its probe nothing is measured. A task starting during the one
        NVML read can contribute to it, but ``consecutive`` readings must agree before
        a device flips, so one such reading cannot move its state.
        """
        monitor = self._gpu_monitor
        if monitor is None:
            return
        probe = self._gpu_executor_probe
        with self._status_lock:
            idle = self._status is WorkerStatus.IDLE and not self._draining.is_set()
            past_grace = time.time() - self._last_task_end >= monitor.config.grace_sec
        monitor.observe(idle and past_grace and probe is not None and not probe())

    def set_busy(self, task_id: str) -> None:
        self._report(WorkerStatus.BUSY, self.client.dispatch_id(task_id), task_id, {})

    def set_idle(self, task_id: str) -> None:
        with self._status_lock:
            self._last_task_end = time.time()
            if self._draining.is_set():
                # A draining worker sends no report; clearing the task keeps its
                # heartbeat from naming it.
                self._task_id = None
                return
        self._report(
            WorkerStatus.IDLE,
            self.client.dispatch_id(task_id),
            task_id,
            {"last_task": task_id},
        )

    def begin_draining(self) -> None:
        """Refuse every later status report; safe from a signal handler."""
        self._draining.set()

    def set_draining(self) -> None:
        """Report the worker busy for as long as it runs, so it takes no further task
        while it shuts down."""
        self._draining.set()
        with self._status_lock:
            # Past its last task, a drain report names that task's dispatch but no
            # running task.
            running = self._task_id if self._status is WorkerStatus.BUSY else None
            self._report_locked(WorkerStatus.BUSY, self._dispatch_id, running, {})

    def _report_again(self) -> None:
        """Report the last status again, which an outage may have dropped."""
        with self._status_lock:
            self._report_locked(self._status, self._dispatch_id, self._task_id, {})

    def _report(
        self,
        status: WorkerStatus,
        dispatch_id: str | None,
        task_id: str | None,
        extra: dict[str, Any],
    ) -> None:
        with self._status_lock:
            if not self._draining.is_set():
                self._report_locked(status, dispatch_id, task_id, extra)

    def _report_locked(
        self,
        status: WorkerStatus,
        dispatch_id: str | None,
        task_id: str | None,
        extra: dict[str, Any],
    ) -> None:
        self._status, self._dispatch_id, self._task_id = status, dispatch_id, task_id
        if status is WorkerStatus.BUSY and task_id is not None:
            extra = {**extra, "task_id": task_id}
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

    def start_ssh_relay(self, lane: SshRelayLane) -> None:
        """Own the worker's SSH relay lane from here to shutdown."""
        self.ssh_relay = lane
        lane.start()

    def shutdown(self, graceful: bool, deadline: float | None = None) -> None:
        """Unregister the worker; `graceful` marks a shutdown it was asked for, and
        `deadline` is the monotonic time by which it must have unregistered."""

        def left() -> float | None:
            return None if deadline is None else max(0.0, deadline - time.monotonic())

        self._stop_event.set()
        if self.ssh_relay is not None:
            # Its connections' cancels leave over the attachment unregistering closes.
            try:
                remaining = left()
                self.ssh_relay.stop(10.0 if remaining is None else remaining)
            except Exception:
                pass
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
