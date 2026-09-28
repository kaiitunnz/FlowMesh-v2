"""EventMonitor worker-event handling."""

import json
import logging
from unittest.mock import MagicMock

import pytest

from server.registries.worker import WorkerRegistry
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
    getattr(registry, write).return_value = False

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

    def test_worker_origin_status_is_still_applied(self) -> None:
        registry = MagicMock()
        registry.set_worker_status.return_value = True
        monitor = _monitor(registry)
        event = WorkerEvent(
            type="STATUS",
            worker_id="wkr-1",
            status=WorkerStatus.IDLE,
            origin="worker",
        )
        monitor._handle_worker_event(event)
        registry.set_worker_status.assert_called_once()

    def test_registry_status_updates_round_trip_as_server_origin(self) -> None:
        rds = MagicMock()
        WorkerRegistry(rds).update_worker_status("wkr-1", WorkerStatus.BUSY)
        _, raw = rds.sync.publish_telemetry.call_args.args
        event = parse_event(json.loads(raw))
        assert isinstance(event, WorkerEvent)
        assert event.origin == "server"

        registry = MagicMock()
        _monitor(registry)._handle_worker_event(event)
        registry.set_worker_status.assert_not_called()
