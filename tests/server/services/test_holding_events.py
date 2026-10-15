"""The events thread hands a worker's holding reports and the report number its
heartbeat names to the content authority, and survives a report it cannot read."""

from typing import Any
from unittest.mock import MagicMock

from shared.content import HoldingReport
from shared.schemas.event import WorkerEvent
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import FakeRegistry
from tests.server.task.test_v2_orchestration import _runtime as _real_runtime


def _monitor_with_authority() -> tuple[Any, MagicMock]:
    monitor = _monitor(_real_runtime(FakeRegistry()))
    authority = MagicMock()
    monitor._content_authority = authority
    return monitor, authority


def test_a_holding_report_reaches_the_authority_as_sent() -> None:
    monitor, authority = _monitor_with_authority()
    report = HoldingReport(
        seq=3, added=(("org", "a" * 64),), dropped=(("org", "b" * 64),)
    )

    monitor._handle_worker_event(
        WorkerEvent(
            type="CONTENT_HOLDING",
            worker_id="wkr-1",
            payload={"report": report.model_dump(mode="json")},
        )
    )

    authority.on_holding.assert_called_once_with("wkr-1", report)


def test_an_unreadable_holding_report_is_dropped_and_the_next_one_handled() -> None:
    monitor, authority = _monitor_with_authority()

    monitor._handle_worker_event(
        WorkerEvent(type="CONTENT_HOLDING", worker_id="wkr-1", payload={"held": []})
    )
    monitor._handle_worker_event(
        WorkerEvent(
            type="CONTENT_HOLDING",
            worker_id="wkr-1",
            payload={"report": HoldingReport(seq=1, whole=True).model_dump()},
        )
    )

    authority.on_holding.assert_called_once_with(
        "wkr-1", HoldingReport(seq=1, whole=True)
    )


def test_a_heartbeat_hands_over_the_holding_report_number_it_names() -> None:
    monitor, authority = _monitor_with_authority()

    monitor._handle_worker_event(
        WorkerEvent(type="HEARTBEAT", worker_id="wkr-1", payload={"holding_seq": 4.0})
    )
    monitor._handle_worker_event(
        WorkerEvent(type="HEARTBEAT", worker_id="wkr-1", payload={})
    )

    authority.on_heartbeat.assert_called_once_with("wkr-1", 4)
