"""The control-stage store is read under the lock its writers hold.

A task event builds the metrics snapshot, which walks the control-stage store,
while stages land from the dispatcher and ledger-drive threads. Only a lock
shared by both sides keeps the walk consistent, so the control here is threaded:
a same-thread reproduction would raise with or without the lock, the recorder's
lock being reentrant.
"""

import logging
import pathlib
import tempfile
import threading

import pytest

from server.services.metrics import MetricsRecorder
from server.services.profiling import ControlPlaneStage, StageWindow
from shared.schemas.event import TaskEvent

_EVENTS = 300


@pytest.fixture
def recorder() -> MetricsRecorder:
    return MetricsRecorder(
        pathlib.Path(tempfile.mkdtemp()),
        logging.getLogger("control-profiling-concurrency"),
        enable_control_profiling=True,
    )


def test_task_events_and_control_stages_run_concurrently(
    recorder: MetricsRecorder,
) -> None:
    stop = threading.Event()
    failures: list[BaseException] = []

    def record_stages() -> None:
        index = 0
        try:
            while not stop.is_set():
                # Rising ids keep new workflows arriving, so the store grows and
                # eventually evicts while the snapshot walks it.
                recorder.record_control_stage(
                    ControlPlaneStage.DS_DRIVE,
                    StageWindow.POST_START,
                    0.001,
                    workflow_id=f"wfl-{index}",
                )
                index += 1
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    def record_events() -> None:
        try:
            for index in range(_EVENTS):
                recorder.record_task_event(
                    TaskEvent(type="TASK_SUBMITTED", task_id=f"tsk-{index}")
                )
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)
        finally:
            stop.set()

    writer = threading.Thread(target=record_stages)
    reader = threading.Thread(target=record_events)
    writer.start()
    reader.start()
    reader.join(timeout=60)
    stop.set()
    writer.join(timeout=60)

    assert not failures, f"concurrent access raised: {failures[0]!r}"
    assert recorder.control_plane_breakdown()["workflows"]
