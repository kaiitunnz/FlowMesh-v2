"""One pending retry per workflow, kept only while the fired callback asks for it."""

import logging
from collections.abc import Callable

from server.task.workflow_retry import WorkflowRetryScheduler


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _scheduler(fire: Callable[[str], None], clock: _Clock) -> WorkflowRetryScheduler:
    return WorkflowRetryScheduler(
        fire,
        logging.getLogger("retry"),
        base_delay_sec=1.0,
        max_delay_sec=4.0,
        clock=clock,
        run_thread=False,
    )


def test_a_retry_that_schedules_again_stays_pending_with_backoff() -> None:
    clock = _Clock()
    scheduler: WorkflowRetryScheduler
    fired: list[float] = []

    def still_failing(workflow_id: str) -> None:
        fired.append(clock.now)
        scheduler.schedule(workflow_id)

    scheduler = _scheduler(still_failing, clock)
    scheduler.schedule("wfl-1")
    while len(fired) < 4:
        clock.now += 0.5
        scheduler.run_due()
    assert fired == [1.0, 3.0, 7.0, 11.0]
    assert scheduler.pending("wfl-1")


def test_a_retry_that_gets_through_settles_and_restarts_its_backoff() -> None:
    clock = _Clock()
    fired: list[str] = []
    scheduler = _scheduler(fired.append, clock)
    scheduler.schedule("wfl-1")
    clock.now = 1.0
    assert scheduler.run_due() == ["wfl-1"]
    assert not scheduler.pending("wfl-1")
    scheduler.schedule("wfl-1")
    clock.now = 1.5
    assert scheduler.run_due() == []
    clock.now = 2.0
    assert scheduler.run_due() == ["wfl-1"]


def test_a_failing_callback_settles_and_drive_now_fires_at_once() -> None:
    clock = _Clock()

    def boom(_workflow_id: str) -> None:
        raise RuntimeError("boom")

    scheduler = _scheduler(boom, clock)
    scheduler.drive_now("wfl-1")
    assert scheduler.run_due() == ["wfl-1"]
    assert not scheduler.pending("wfl-1")
