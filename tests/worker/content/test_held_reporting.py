"""What a worker tells control about its cache: its whole holding when it attaches or
is asked, and each copy it keeps or drops, in numbered reports that follow the disk."""

import threading
import time
from pathlib import Path

from shared.content import HOLDING_RESYNC_FRAME, HoldingReport
from worker.content import ContentLaneHost, HoldingReporter, WorkerContentCache


class _Reports:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.reports: list[HoldingReport] = []
        self._fail = fail_first
        self.arrived = threading.Event()

    def __call__(self, report: HoldingReport) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("event stream not ready")
        self.reports.append(report)
        self.arrived.set()

    @property
    def seqs(self) -> list[int]:
        return [report.seq for report in self.reports]


def _cache(
    tmp_path: Path, reports: _Reports, **kwargs: float
) -> tuple[WorkerContentCache, HoldingReporter]:
    reporter = HoldingReporter(reports, flush_after_sec=60.0)
    cache = WorkerContentCache(
        tmp_path / "content",
        retain_sec=kwargs.get("retain", 0.0),
        max_bytes=int(kwargs.get("max_bytes", 0)),
        holding=reporter,
    )
    return cache, reporter


def _lane(cache: WorkerContentCache) -> ContentLaneHost:
    return ContentLaneHost(
        store=cache,
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id, peer_only, request_id: None,
        worker_id="wkr-1",
        generation=2,
    )


def test_a_restarted_worker_reports_the_copies_still_on_its_disk_whole(
    tmp_path: Path,
) -> None:
    first = WorkerContentCache(tmp_path / "content")
    kept = [first.write("local", f"object-{index}".encode()) for index in range(5)]

    reports = _Reports()
    cache, _ = _cache(tmp_path, reports)
    lane = _lane(cache)
    lane.start()
    try:
        assert lane.report_held() == 5
    finally:
        lane.stop()
    (report,) = reports.reports
    assert (report.seq, report.whole, report.dropped) == (1, True, ())
    assert set(report.added) == {("local", r.content_digest) for r in kept}


def test_each_kept_and_dropped_copy_leaves_in_the_next_numbered_report(
    tmp_path: Path,
) -> None:
    reports = _Reports()
    cache, reporter = _cache(tmp_path, reports)
    cache.report_whole(restart=True)
    a = cache.write("local", b"a")
    b = cache.write("local", b"b")
    cache.drop(a)
    reporter.stop()

    assert reports.seqs == [1, 2]
    report = reports.reports[1]
    # A copy kept and dropped before the report left is named once, as dropped.
    assert report.added == (("local", b.content_digest),)
    assert report.dropped == (("local", a.content_digest),)
    assert cache.holding_seq == 2


def test_an_evicted_copy_is_reported_dropped(tmp_path: Path) -> None:
    reports = _Reports()
    cache, reporter = _cache(tmp_path, reports, retain=1.0)
    old = cache.write("local", b"old")
    time.sleep(1.1)

    assert cache.evict() == [("local", old.content_digest)]
    reporter.stop()

    assert reports.reports[-1].dropped == (("local", old.content_digest),)


def test_a_copy_evicted_before_its_note_is_never_reported_kept(
    tmp_path: Path,
) -> None:
    reports = _Reports()
    cache, reporter = _cache(tmp_path, reports)
    backing = cache._objects
    touch = backing.touch

    def evicted_meanwhile(scope: str, digest: str) -> None:
        touch(scope, digest)
        backing.remove(scope, digest)

    backing.touch = evicted_meanwhile  # type: ignore[method-assign]
    cache.write("local", b"gone at once")
    reporter.stop()

    assert reports.reports == []


def test_a_full_batch_leaves_without_waiting(tmp_path: Path) -> None:
    reports = _Reports()
    reporter = HoldingReporter(reports, flush_after_sec=60.0, max_changes=3)
    for index in range(3):
        reporter.kept(("local", f"{index:064x}"))

    assert reports.seqs == [1]
    assert len(reports.reports[0].added) == 3
    reporter.stop()


def test_a_report_that_could_not_be_sent_still_takes_its_number(
    tmp_path: Path,
) -> None:
    reports = _Reports(fail_first=True)
    reporter = HoldingReporter(reports, flush_after_sec=0.01)
    reporter.kept(("local", "a" * 64))
    time.sleep(0.2)
    reporter.kept(("local", "b" * 64))
    reporter.stop()

    # Control sees report 2 follow nothing it applied and asks for the whole holding.
    assert reports.seqs == [2]


def test_a_new_registration_numbers_its_whole_report_first(tmp_path: Path) -> None:
    reports = _Reports()
    cache, _ = _cache(tmp_path, reports)
    cache.report_whole(restart=True)
    cache.report_whole()
    lane = _lane(cache)
    lane.start()
    try:
        lane.rebind("wkr-7", 7)
    finally:
        lane.stop()

    assert [(r.seq, r.whole) for r in reports.reports] == [
        (1, True),
        (2, True),
        (1, True),
    ]


def test_control_asking_for_the_whole_holding_gets_it(tmp_path: Path) -> None:
    reports = _Reports()
    cache, _ = _cache(tmp_path, reports)
    kept = cache.write("local", b"kept")
    lane = _lane(cache)
    lane.start()
    try:
        assert lane.route(HOLDING_RESYNC_FRAME, {})
        assert reports.arrived.wait(5.0)
    finally:
        lane.stop()

    whole = [report for report in reports.reports if report.whole]
    assert whole and whole[0].added == (("local", kept.content_digest),)
