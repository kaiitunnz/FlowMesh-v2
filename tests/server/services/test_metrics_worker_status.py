"""The metrics recorder tracks a worker's status from its heartbeats as well as its
status reports, since a heartbeat repeats the status a dropped report carried."""

import logging
from pathlib import Path

from server.services.metrics import MetricsRecorder
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus


def test_a_heartbeat_carrying_a_status_updates_it(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path, logging.getLogger("metrics-status"))

    recorder.record_worker_event(
        WorkerEvent(
            type="HEARTBEAT",
            worker_id="wkr-1",
            status=WorkerStatus.BUSY,
            payload={"ttl_sec": 120},
        )
    )
    recorder.record_worker_event(
        WorkerEvent(type="HEARTBEAT", worker_id="wkr-1", payload={"ttl_sec": 120})
    )

    assert recorder._worker_meta["wkr-1"]["status"] is WorkerStatus.BUSY
