import threading

import pytest

from server.reclamation.pacing import Pacer


class _Waits(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float] = []

    def wait(self, timeout: float | None = None) -> bool:
        assert timeout is not None
        self.waits.append(timeout)
        return self.is_set()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_pacer_rests_in_proportion_to_its_work_after_every_few_units() -> None:
    stop, clock = _Waits(), _Clock()
    pacer = Pacer(stop, 3, duty=0.1, min_pause_sec=0.001, clock=clock)

    for _ in range(2):
        clock.now += 0.01
        assert not pacer.tick()
    assert stop.waits == []
    clock.now += 0.01
    assert not pacer.tick()

    assert stop.waits == [pytest.approx(0.27)]


def test_a_pacer_rests_at_least_its_minimum() -> None:
    stop, clock = _Waits(), _Clock()
    pacer = Pacer(stop, 1, duty=0.1, min_pause_sec=0.005, clock=clock)

    pacer.tick()

    assert stop.waits == [0.005]


def test_a_pacer_at_full_duty_never_rests_past_its_minimum() -> None:
    stop, clock = _Waits(), _Clock()
    pacer = Pacer(stop, 1, duty=1.0, min_pause_sec=0, clock=clock)

    clock.now += 5
    pacer.tick()

    assert stop.waits == [0]


def test_a_set_stop_ends_the_work_at_the_next_unit() -> None:
    stop, clock = _Waits(), _Clock()
    pacer = Pacer(stop, 100, clock=clock)
    stop.set()

    assert pacer.tick()
    assert pacer.pause()
