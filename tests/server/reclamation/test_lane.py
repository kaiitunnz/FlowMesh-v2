"""The publication lane: bounded, coalescing, and fair to a quiet workflow."""

import threading
from collections.abc import Callable

from server.reclamation.lane import Admission, PublicationLane


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Gate:
    """Jobs that block until released, one at a time."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.running: list[str] = []
        self.done: list[str] = []
        self.open: set[str] = set()

    def job(self, name: str) -> Callable[[], None]:
        def run() -> None:
            with self.cv:
                self.running.append(name)
                self.cv.notify_all()
                self.cv.wait_for(lambda: name in self.open, timeout=5)
                self.done.append(name)
                self.cv.notify_all()

        return run

    def finish(self, name: str) -> None:
        with self.cv:
            self.open.add(name)
            self.cv.notify_all()
            assert self.cv.wait_for(lambda: name in self.done, timeout=5)

    def wait_running(self, name: str) -> None:
        with self.cv:
            assert self.cv.wait_for(lambda: name in self.running, timeout=5)


def _settle(lane: PublicationLane) -> None:
    with lane._cv:
        assert lane._cv.wait_for(lambda: not lane._held, timeout=5)


def test_a_held_publication_coalesces_and_a_full_lane_turns_one_away() -> None:
    gate = _Gate()
    lane = PublicationLane(threads=1, capacity=1, clock=_Clock())
    lane.start()
    try:
        assert lane.offer("a", "wfl-1", gate.job("a")) is Admission.ADMITTED
        gate.wait_running("a")
        assert lane.offer("a", "wfl-1", gate.job("a")) is Admission.COALESCED
        assert lane.offer("b", "wfl-1", gate.job("b")) is Admission.BUSY
        gate.finish("a")
        # The freed slot is reserved for the waiting publication's next offer.
        assert lane.offer("b", "wfl-1", gate.job("b")) is Admission.ADMITTED
        gate.wait_running("b")
        gate.finish("b")
    finally:
        lane.stop(1.0)


def test_a_quiet_workflow_is_admitted_however_hot_another_runs() -> None:
    clock = _Clock()
    gate = _Gate()
    lane = PublicationLane(threads=1, capacity=1, clock=clock)
    lane.start()
    try:
        assert lane.offer("hot-0", "hot", gate.job("hot-0")) is Admission.ADMITTED
        gate.wait_running("hot-0")
        # The quiet publication arrives while the lane is full, then the hot workflow
        # keeps offering new publications ahead of every resend of it.
        assert lane.offer("quiet", "quiet", gate.job("quiet")) is Admission.BUSY
        for index in range(1, 5):
            assert lane.offer(f"hot-{index}", "hot", gate.job("x")) is Admission.BUSY
        gate.finish("hot-0")
        _settle(lane)
        for index in range(1, 5):
            assert lane.offer(f"hot-{index}", "hot", gate.job("x")) is Admission.BUSY
        assert lane.offer("quiet", "quiet", gate.job("quiet")) is Admission.ADMITTED
        gate.wait_running("quiet")
        gate.finish("quiet")
        _settle(lane)
        # Turns alternate across workflows, so the hot workflow's oldest is next.
        assert lane.offer("hot-1", "hot", gate.job("hot-1")) is Admission.ADMITTED
        gate.finish("hot-1")
    finally:
        lane.stop(1.0)


def test_a_waiting_publication_keeps_its_place_across_offers() -> None:
    clock = _Clock()
    gate = _Gate()
    lane = PublicationLane(threads=1, capacity=1, clock=clock)
    lane.start()
    try:
        lane.offer("first", "wfl-1", gate.job("first"))
        gate.wait_running("first")
        assert lane.offer("old", "wfl-2", gate.job("old")) is Admission.BUSY
        assert lane.offer("new", "wfl-2", gate.job("new")) is Admission.BUSY
        # Offering again does not move it behind a younger one of its workflow.
        assert lane.offer("old", "wfl-2", gate.job("old")) is Admission.BUSY
        gate.finish("first")
        _settle(lane)
        assert lane.offer("new", "wfl-2", gate.job("new")) is Admission.BUSY
        assert lane.offer("old", "wfl-2", gate.job("old")) is Admission.ADMITTED
        gate.finish("old")
    finally:
        lane.stop(1.0)


def test_a_reservation_whose_source_is_gone_lapses_to_the_next() -> None:
    clock = _Clock()
    gate = _Gate()
    lane = PublicationLane(threads=1, capacity=1, reservation_sec=10.0, clock=clock)
    lane.start()
    try:
        lane.offer("first", "wfl-1", gate.job("first"))
        gate.wait_running("first")
        assert lane.offer("gone", "wfl-2", gate.job("gone")) is Admission.BUSY
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.BUSY
        gate.finish("first")
        _settle(lane)
        # The slot is the cancelled source's until its reservation lapses.
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.BUSY
        clock.now = 11.0
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.ADMITTED
        gate.finish("live")
        _settle(lane)
        assert lane.depth() == (0, 0, 0)
    finally:
        lane.stop(1.0)


def test_a_failing_publication_frees_its_slot() -> None:
    lane = PublicationLane(threads=1, capacity=1, clock=_Clock())
    lane.start()
    ran = threading.Event()

    def fail() -> None:
        raise RuntimeError("boom")

    try:
        assert lane.offer("a", "wfl-1", fail) is Admission.ADMITTED
        _settle(lane)
        assert lane.offer("b", "wfl-1", ran.set) is Admission.ADMITTED
        assert ran.wait(5)
    finally:
        lane.stop(1.0)


def test_a_waiting_publication_its_source_stops_offering_lapses() -> None:
    clock = _Clock()
    gate = _Gate()
    lane = PublicationLane(threads=1, capacity=1, waiting_sec=60.0, clock=clock)
    lane.start()
    try:
        lane.offer("first", "wfl-1", gate.job("first"))
        gate.wait_running("first")
        for key in ("gone-a", "gone-b"):
            assert lane.offer(key, "wfl-2", gate.job(key)) is Admission.BUSY
        clock.now = 30.0
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.BUSY
        assert lane.depth()[1] == 3

        clock.now = 61.0
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.BUSY
        assert lane.depth() == (1, 1, 0)
        gate.finish("first")
        _settle(lane)
        assert lane.offer("live", "wfl-3", gate.job("live")) is Admission.ADMITTED
    finally:
        lane.stop(1.0)
