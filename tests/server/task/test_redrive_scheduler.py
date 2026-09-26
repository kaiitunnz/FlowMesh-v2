"""One pending re-drive per workflow, backing off while the store stays away."""

import logging
import threading
from typing import Any

import pytest

from server.task.redrive import StoreRedriveScheduler


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _scheduler(
    fire: Any, clock: _Clock, logger: logging.Logger | None = None
) -> StoreRedriveScheduler:
    return StoreRedriveScheduler(
        fire,
        logger or logging.getLogger("redrive"),
        base_delay_sec=1.0,
        max_delay_sec=8.0,
        warn_after=3,
        clock=clock,
        run_thread=False,
    )


def test_repeated_requests_share_one_pending_re_drive() -> None:
    fired: list[str] = []
    clock = _Clock()
    scheduler = _scheduler(fired.append, clock)
    for _ in range(5):
        scheduler.schedule("wfl-1")
    clock.now = 1.0
    assert scheduler.run_due() == ["wfl-1"]
    assert fired == ["wfl-1"]


def test_a_workflow_still_waiting_backs_off_to_the_ceiling() -> None:
    clock = _Clock()
    scheduler: StoreRedriveScheduler

    def still_away(workflow_id: str) -> None:
        scheduler.schedule(workflow_id)

    scheduler = _scheduler(still_away, clock)
    scheduler.schedule("wfl-1")
    gaps: list[float] = []
    last = 0.0
    for _ in range(6):
        while not scheduler.run_due():
            clock.now += 0.5
        gaps.append(clock.now - last)
        last = clock.now
    assert gaps == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_one_that_gets_through_starts_over(caplog: pytest.LogCaptureFixture) -> None:
    clock = _Clock()
    away = {"on": True}
    scheduler: StoreRedriveScheduler

    def fire(workflow_id: str) -> None:
        if away["on"]:
            scheduler.schedule(workflow_id)

    scheduler = _scheduler(fire, clock)
    scheduler.schedule("wfl-1")
    for _ in range(3):
        clock.now += 8.0
        scheduler.run_due()
    assert "still waiting on the content store" in caplog.text

    away["on"] = False
    clock.now += 8.0
    scheduler.run_due()
    assert not scheduler.pending("wfl-1")
    scheduler.schedule("wfl-1")
    clock.now += 1.0
    assert scheduler.run_due() == ["wfl-1"]


def test_a_settled_or_stopped_workflow_never_fires() -> None:
    fired: list[str] = []
    clock = _Clock()
    scheduler = _scheduler(fired.append, clock)
    scheduler.schedule("wfl-1")
    scheduler.settle("wfl-1")
    scheduler.schedule("wfl-2")
    scheduler.stop()
    scheduler.schedule("wfl-3")
    clock.now = 100.0
    assert scheduler.run_due() == [] and fired == []


def test_the_scheduler_thread_fires_and_stops() -> None:
    fired = threading.Event()
    scheduler = StoreRedriveScheduler(
        lambda _workflow_id: fired.set(),
        logging.getLogger("redrive"),
        base_delay_sec=0.01,
    )
    scheduler.schedule("wfl-1")
    assert fired.wait(2.0)
    scheduler.stop()


def test_a_drive_now_fires_at_once_and_keeps_the_backoff() -> None:
    clock = _Clock()
    fired: list[str] = []
    scheduler: StoreRedriveScheduler

    def away_once(workflow_id: str) -> None:
        fired.append(workflow_id)
        if len(fired) == 1:
            scheduler.schedule(workflow_id)

    scheduler = _scheduler(away_once, clock)
    scheduler.drive_now("wfl-1")
    assert scheduler.run_due() == ["wfl-1"]
    # The first drive found the store away: its backoff starts at the base delay.
    assert scheduler.run_due() == []
    scheduler.drive_now("wfl-1")
    assert scheduler.run_due() == ["wfl-1"]
    assert fired == ["wfl-1", "wfl-1"]
    assert not scheduler.pending("wfl-1")


def test_a_drive_now_moves_a_waiting_re_drive_up() -> None:
    clock = _Clock()
    fired: list[str] = []
    scheduler = _scheduler(fired.append, clock)
    scheduler.schedule("wfl-1")
    assert scheduler.run_due() == []
    scheduler.drive_now("wfl-1")
    assert scheduler.run_due() == ["wfl-1"]
    clock.now = 5.0
    assert scheduler.run_due() == []
    assert fired == ["wfl-1"]
