"""EventMonitor worker-event handling."""

import json
import logging
from unittest.mock import MagicMock

import pytest

from server.registries.worker import ReportOutcome, StatusReport, WorkerRegistry
from server.services.monitoring import EventMonitor
from shared.schemas.event import WorkerEvent, parse_event
from shared.schemas.worker import WorkerStatus


def _monitor(worker_registry: MagicMock) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test-monitor"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=worker_registry,
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
    )


def _heartbeat(worker_id: str) -> WorkerEvent:
    return WorkerEvent(type="HEARTBEAT", worker_id=worker_id, payload={"ttl_sec": 120})


def _status(worker_id: str) -> WorkerEvent:
    return WorkerEvent(type="STATUS", worker_id=worker_id, payload={})


@pytest.mark.parametrize(
    ("event", "write"),
    [
        (_heartbeat("wkr-1"), "update_worker_hb"),
        (_status("wkr-1"), "set_worker_status"),
    ],
    ids=["heartbeat", "status"],
)
def test_an_event_from_an_unknown_worker_is_reported_and_dropped(
    event: WorkerEvent, write: str, caplog: pytest.LogCaptureFixture
) -> None:
    registry = MagicMock()
    getattr(registry, write).return_value = StatusReport(ReportOutcome.UNKNOWN)

    with caplog.at_level(logging.WARNING, logger="test-monitor"):
        _monitor(registry)._handle_worker_event(event)

    assert "unknown worker wkr-1" in caplog.text


class TestServerOriginStatusEvents:
    """A status event the server published is not written back to the registry."""

    def test_server_origin_status_is_not_reapplied(self) -> None:
        registry = MagicMock()
        monitor = _monitor(registry)
        event = WorkerEvent(
            type="STATUS",
            worker_id="wkr-1",
            status=WorkerStatus.BUSY,
            origin="server",
        )
        monitor._handle_worker_event(event)
        registry.set_worker_status.assert_not_called()

    def test_a_worker_origin_status_is_applied(self) -> None:
        registry = MagicMock()
        registry.set_worker_status.return_value = StatusReport(ReportOutcome.APPLIED)
        monitor = _monitor(registry)
        event = WorkerEvent(
            type="STATUS",
            worker_id="wkr-1",
            status=WorkerStatus.IDLE,
            dispatch_id="dsp-1",
            origin="worker",
        )
        monitor._handle_worker_event(event)
        registry.set_worker_status.assert_called_once_with(
            "wkr-1", WorkerStatus.IDLE, event.ts, {}, "dsp-1"
        )

    def test_registry_status_updates_round_trip_as_server_origin(self) -> None:
        rds = MagicMock()
        rds.sync.eval.return_value = 1
        WorkerRegistry(rds).reserve_worker("wkr-1", "tsk-1", "dsp-1")
        _, raw = rds.sync.publish_telemetry.call_args.args
        event = parse_event(json.loads(raw))
        assert isinstance(event, WorkerEvent)
        assert event.origin == "server"

        registry = MagicMock()
        _monitor(registry)._handle_worker_event(event)
        registry.set_worker_status.assert_not_called()


class TestFencedIdle:
    """An IDLE for an earlier dispatch frees the worker only once the dispatch it is
    reserved for is no longer in flight."""

    def _fenced(self, in_flight: bool) -> tuple[MagicMock, MagicMock]:
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(
            ReportOutcome.FENCED, "tsk-2", "dsp-2"
        )
        runtime = MagicMock()
        runtime.dispatch_in_flight.return_value = in_flight
        monitor = _monitor(registry)
        monitor._runtime = runtime
        monitor._handle_worker_event(
            WorkerEvent(
                type="HEARTBEAT",
                worker_id="wkr-1",
                status=WorkerStatus.IDLE,
                dispatch_id="dsp-1",
                payload={"ttl_sec": 120},
            )
        )
        runtime.dispatch_in_flight.assert_called_once_with("tsk-2", "dsp-2", "wkr-1")
        return registry, runtime

    def test_a_reservation_whose_dispatch_is_in_flight_stands(self) -> None:
        registry, _ = self._fenced(in_flight=True)
        registry.release_worker.assert_not_called()

    def test_a_reservation_whose_dispatch_was_lost_is_released(self) -> None:
        registry, _ = self._fenced(in_flight=False)
        registry.release_worker.assert_called_once_with("wkr-1", "dsp-2")


def test_a_heartbeat_carries_the_workers_status_to_the_registry() -> None:
    registry = MagicMock()
    registry.update_worker_hb.return_value = StatusReport(ReportOutcome.APPLIED)
    event = WorkerEvent(
        type="HEARTBEAT",
        worker_id="wkr-1",
        status=WorkerStatus.BUSY,
        dispatch_id="dsp-1",
        payload={"ttl_sec": 90},
    )

    _monitor(registry)._handle_worker_event(event)

    registry.update_worker_hb.assert_called_once_with(
        "wkr-1", event.ts, 90, WorkerStatus.BUSY, "dsp-1"
    )


class TestHeartbeatCarriesGpuAvailability:
    def _availability_heartbeat(self, worker_id: str) -> WorkerEvent:
        return WorkerEvent(
            type="HEARTBEAT",
            worker_id=worker_id,
            payload={"ttl_sec": 120},
            metrics={
                "gpu_availability": {"GPU-held": {"available": False, "free_bytes": 8}}
            },
        )

    def test_availability_is_recorded(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(ReportOutcome.APPLIED)
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_called_once_with(
            "wkr-1", {"GPU-held": {"available": False, "free_bytes": 8}}
        )

    def test_an_empty_map_is_recorded(self) -> None:
        # It clears a stale reading; only an absent key means no opinion.
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(ReportOutcome.APPLIED)
        event = self._availability_heartbeat("wkr-1")
        event.metrics["gpu_availability"] = {}
        _monitor(registry)._handle_worker_event(event)
        registry.record_gpu_availability.assert_called_once_with("wkr-1", {})

    def test_unknown_worker_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(ReportOutcome.UNKNOWN)
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_not_called()

    def test_a_recording_failure_does_not_escape(self) -> None:
        # Scheduling advice must never cost the worker its liveness update.
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(ReportOutcome.APPLIED)
        registry.record_gpu_availability.side_effect = RuntimeError("redis down")
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.update_worker_hb.assert_called_once()

    def test_heartbeat_without_availability_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = StatusReport(ReportOutcome.APPLIED)
        _monitor(registry)._handle_worker_event(_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_not_called()
