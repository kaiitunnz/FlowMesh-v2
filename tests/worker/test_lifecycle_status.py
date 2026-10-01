"""A worker reports its status with the dispatch it concerns, and repeats both on
every heartbeat."""

import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from shared.schemas.worker import WorkerStatus
from worker.lifecycle import Lifecycle


def _lifecycle(tmp_path: Path) -> tuple[Lifecycle, MagicMock]:
    client = MagicMock()
    client.dispatch_id.side_effect = lambda task_id: f"dsp-{task_id}"
    lifecycle = Lifecycle(
        cast(Any, client),
        hb_sec=5,
        hb_ttl_sec=15,
        hb_file=tmp_path / "hb",
        cost_per_hour=0.0,
    )
    return lifecycle, client


def _heartbeat(lifecycle: Lifecycle, client: MagicMock) -> dict[str, Any]:
    """Run one pass of the heartbeat loop and return what it sent."""
    stop = cast(Any, lifecycle)._stop_event
    client.heartbeat.side_effect = lambda **_: stop.set()
    lifecycle._hb_loop()
    stop.clear()
    return client.heartbeat.call_args.kwargs


def test_a_status_report_names_the_dispatch_it_concerns(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)

    lifecycle.set_busy("tsk-1")
    lifecycle.set_idle("tsk-1")

    assert [call.args for call in client.set_status.call_args_list] == [
        (WorkerStatus.BUSY, {"task_id": "tsk-1"}, "dsp-tsk-1"),
        (WorkerStatus.IDLE, {"last_task": "tsk-1"}, "dsp-tsk-1"),
    ]


def test_a_heartbeat_repeats_the_last_reported_status(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)

    lifecycle.set_busy("tsk-1")
    busy = _heartbeat(lifecycle, client)
    lifecycle.set_idle("tsk-1")
    idle = _heartbeat(lifecycle, client)

    assert (busy["status"], busy["dispatch_id"]) == (WorkerStatus.BUSY, "dsp-tsk-1")
    assert (idle["status"], idle["dispatch_id"]) == (WorkerStatus.IDLE, "dsp-tsk-1")


def test_a_draining_worker_reports_and_heartbeats_busy(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)
    lifecycle.set_busy("tsk-1")
    lifecycle.set_idle("tsk-1")

    lifecycle.set_draining()

    assert client.set_status.call_args.args[0] is WorkerStatus.BUSY
    assert _heartbeat(lifecycle, client)["status"] is WorkerStatus.BUSY


def test_a_heartbeat_never_trails_a_report_sent_before_it(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)
    lifecycle.set_busy("tsk-1")
    sent: list[WorkerStatus] = []
    reporting = threading.Event()
    heartbeat_done = threading.Event()

    def slow_report(status: WorkerStatus, *_: Any) -> None:
        reporting.set()
        # A heartbeat racing the report waits for it rather than sending the status
        # the report is replacing.
        heartbeat_done.wait(timeout=0.2)
        sent.append(status)

    def heartbeat(**kwargs: Any) -> None:
        sent.append(kwargs["status"])
        heartbeat_done.set()
        cast(Any, lifecycle)._stop_event.set()

    client.set_status.side_effect = slow_report
    client.heartbeat.side_effect = heartbeat
    reporter = threading.Thread(target=lifecycle.set_idle, args=("tsk-1",))
    reporter.start()
    reporting.wait(timeout=2.0)
    lifecycle._hb_loop()
    reporter.join(timeout=2.0)

    assert sent == [WorkerStatus.IDLE, WorkerStatus.IDLE]


def test_a_draining_worker_never_reports_itself_idle(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)
    lifecycle.set_busy("tsk-1")
    lifecycle.set_draining()

    lifecycle.set_idle("tsk-1")

    assert client.set_status.call_args.args[0] is WorkerStatus.BUSY
    assert _heartbeat(lifecycle, client)["status"] is WorkerStatus.BUSY


def test_a_reconnected_event_stream_gets_the_last_status_again(
    tmp_path: Path,
) -> None:
    lifecycle, client = _lifecycle(tmp_path)
    lifecycle.start({}, cast(Any, None), cast(Any, None), None, [])
    lifecycle.set_busy("tsk-1")
    cast(Any, lifecycle)._stop_event.set()
    [(on_ready,), _] = client.on_event_stream_ready.call_args

    on_ready()

    assert client.set_status.call_args.args == (WorkerStatus.BUSY, {}, "dsp-tsk-1")


def test_a_heartbeat_names_the_task_of_the_dispatch_it_runs(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)

    lifecycle.set_busy("tsk-1")

    assert _heartbeat(lifecycle, client)["task_id"] == "tsk-1"


def test_a_drain_past_its_last_task_names_no_running_task(tmp_path: Path) -> None:
    lifecycle, client = _lifecycle(tmp_path)
    lifecycle.set_busy("tsk-1")
    lifecycle.set_idle("tsk-1")

    lifecycle.set_draining()

    assert _heartbeat(lifecycle, client)["task_id"] is None
