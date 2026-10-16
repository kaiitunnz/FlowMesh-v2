"""The artifact uploader: it stores what a success staged, reports each step until
control answers, and treats another registration's custody as evidence only."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentReference, ContentUnavailable, reference_for
from shared.tasks.artifact_upload import UploadDisposition, UploadReport, UploadSettled
from worker.content.custody import ArtifactCustody, UploadJournal
from worker.content.uploads import (
    EVIDENCE_RESEND_SEC,
    EVIDENCE_SCAN_SEC,
    RESEND_SEC,
    ArtifactUploader,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Ports:
    """What the uploader reaches: the store, and control's event stream."""

    def __init__(self) -> None:
        self.stored: list[tuple[str, str, bytes]] = []
        self.successes: list[tuple[str, str]] = []
        self.reports: list[tuple[str, str, UploadReport, str | None]] = []
        self.outage = 0
        self.attempts = 0
        self.wrong = False
        self.explode = 0
        self.drop_reports = False
        self.poisoned: set[bytes] = set()
        self.stored_at_report: list[list[bytes]] = []
        self.cv = threading.Condition()

    def upload(
        self, task_id: str, scope: str, path: Path, media_type: str
    ) -> ContentReference:
        with self.cv:
            self.attempts += 1
            self.cv.notify_all()
        if self.explode:
            self.explode -= 1
            raise RuntimeError("unexpected")
        if self.outage:
            self.outage -= 1
            raise ContentUnavailable("store down")
        data = path.read_bytes()
        if data in self.poisoned:
            raise RuntimeError("unexpected")
        with self.cv:
            self.stored.append((task_id, scope, data))
            self.cv.notify_all()
        reference = reference_for(scope, data)
        if self.wrong:
            return reference.model_copy(update={"size_bytes": 1})
        return reference

    def resend(self, task_id: str, dispatch_id: str, metadata: dict[str, Any]) -> None:
        with self.cv:
            self.successes.append((task_id, dispatch_id))
            self.cv.notify_all()

    def report(
        self, task_id: str, dispatch_id: str, report: UploadReport, error: str | None
    ) -> None:
        if self.drop_reports:
            return
        with self.cv:
            self.reports.append((task_id, dispatch_id, report, error))
            self.stored_at_report.append([data for _, _, data in self.stored])
            self.cv.notify_all()

    def wait(self, predicate: Any, timeout: float = 5.0) -> None:
        with self.cv:
            assert self.cv.wait_for(predicate, timeout), "timed out"


def _uploader(
    root: Path,
    ports: _Ports,
    clock: Any = None,
    alias: str = "w1",
    worker_id: str = "wkr-1",
    custody: ArtifactCustody | None = None,
) -> ArtifactUploader:
    kwargs = {"clock": clock} if clock is not None else {}
    return ArtifactUploader(
        custody or ArtifactCustody(root),
        alias,
        identity=lambda: (worker_id, 1),
        upload=ports.upload,
        resend_success=ports.resend,
        report=ports.report,
        **kwargs,
    )


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / "out" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _stage_file(
    uploader: ArtifactUploader, journal: UploadJournal, task_id: str, source: Path
) -> ContentReference:
    with source.open("rb") as opened:
        return uploader.stage(journal, task_id, opened)


def _stage(
    uploader: ArtifactUploader,
    tmp_path: Path,
    files: dict[str, bytes],
    task_id: str = "tsk-1",
    journal: UploadJournal | None = None,
    submit: bool = True,
) -> UploadJournal:
    journal = journal or uploader.begin("dsp-1", "org")
    for name, data in files.items():
        _stage_file(uploader, journal, task_id, _write(tmp_path, name, data))
    if submit:
        uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    return journal


def _settled(disposition: UploadDisposition, task_id: str = "tsk-1") -> UploadSettled:
    return UploadSettled(task_id=task_id, dispatch_id="dsp-1", disposition=disposition)


def _nudge(uploader: ArtifactUploader, clock: _Clock, seconds: float) -> None:
    clock.now += seconds
    with uploader._cv:
        uploader._cv.notify_all()


def test_staged_artifacts_are_stored_then_reported_uploaded(tmp_path) -> None:
    ports = _Ports()
    uploader = _uploader(tmp_path / "content", ports)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"})

    ports.wait(lambda: ports.reports)

    assert sorted(data for _, _, data in ports.stored) == [b"alpha", b"beta"]
    assert ports.reports == [("tsk-1", "dsp-1", UploadReport.UPLOADED, None)]
    assert uploader.dispatch_of("tsk-1") == "dsp-1"
    uploader.stop(1.0)


def test_nothing_is_written_for_a_dispatch_that_stages_nothing(tmp_path) -> None:
    root = tmp_path / "content"
    uploader = _uploader(root, _Ports())

    uploader.abandon(uploader.begin("dsp-1", "org"))

    assert not root.exists()


def test_custody_ends_only_when_control_settles_the_upload(tmp_path) -> None:
    ports = _Ports()
    root = tmp_path / "content"
    uploader = _uploader(root, ports)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})
    ports.wait(lambda: ports.reports)
    custody = ArtifactCustody(root)

    uploader.settle(_settled(UploadDisposition.PENDING))
    assert list(custody.journals())

    uploader.settle(_settled(UploadDisposition.DURABLE))
    assert list(custody.journals()) == []
    assert uploader.dispatch_of("tsk-1") is None
    uploader.stop(1.0)


def test_unanswered_reports_are_sent_again_on_a_pace(tmp_path) -> None:
    ports = _Ports()
    clock = _Clock()
    uploader = _uploader(tmp_path / "content", ports, clock)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})
    ports.wait(lambda: len(ports.reports) == 1)

    _nudge(uploader, clock, RESEND_SEC + 1)
    ports.wait(lambda: len(ports.reports) == 2 and ports.successes)

    uploader.settle(_settled(UploadDisposition.DURABLE))
    _nudge(uploader, clock, RESEND_SEC + 1)
    uploader.stop(1.0)
    assert len(ports.reports) == 2


@pytest.mark.parametrize("unuploadable", [False, True])
def test_a_pending_answer_has_a_lost_receipt_reported_again(
    tmp_path, unuploadable: bool
) -> None:
    ports = _Ports()
    ports.drop_reports = True
    ports.wrong = unuploadable
    clock = _Clock()
    uploader = _uploader(tmp_path / "content", ports, clock)
    uploader.start()
    journal = _stage(uploader, tmp_path, {"a.bin": b"alpha"})
    ports.wait(lambda: ports.attempts >= 1)
    with uploader._cv:
        assert uploader._cv.wait_for(
            lambda: journal.tasks["tsk-1"].uploaded
            or journal.tasks["tsk-1"].failed is not None,
            5.0,
        )
    ports.drop_reports = False

    uploader.settle(_settled(UploadDisposition.PENDING))

    ports.wait(lambda: ports.reports)
    expected = UploadReport.UNUPLOADABLE if unuploadable else UploadReport.UPLOADED
    assert ports.reports[0][2] is expected
    uploader.stop(0.5)


def test_a_store_outage_is_retried_and_never_reported_lost(tmp_path) -> None:
    ports = _Ports()
    ports.outage = 2
    clock = _Clock()
    uploader = _uploader(tmp_path / "content", ports, clock)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})

    for attempt in (1, 2):
        ports.wait(lambda: ports.attempts >= attempt)
        assert ports.reports == []
        _nudge(uploader, clock, 61)
    ports.wait(lambda: ports.reports)

    assert ports.reports[0][2] is UploadReport.UPLOADED
    uploader.stop(1.0)


def test_an_unexpected_failure_never_stops_the_uploads(tmp_path) -> None:
    ports = _Ports()
    ports.explode = 1
    clock = _Clock()
    uploader = _uploader(tmp_path / "content", ports, clock)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})
    ports.wait(lambda: ports.attempts >= 1)

    _nudge(uploader, clock, 2)
    ports.wait(lambda: ports.reports)

    assert ports.reports[0][2] is UploadReport.UPLOADED
    uploader.stop(1.0)


def test_a_journal_that_cannot_be_written_reports_what_is_owed_unfinishable(
    tmp_path,
) -> None:
    class _Full(ArtifactCustody):
        full = False

        def record(self, journal: UploadJournal) -> None:
            if self.full:
                raise OSError(28, "No space left on device")
            super().record(journal)

    ports = _Ports()
    custody = _Full(tmp_path / "content")
    uploader = _uploader(tmp_path / "content", ports, custody=custody)
    journal = uploader.begin("dsp-1", "org")
    _stage(uploader, tmp_path, {"a.bin": b"alpha"}, journal=journal, submit=False)
    _stage(uploader, tmp_path, {"b.bin": b"beta"}, "tsk-2", journal, submit=False)
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    custody.full = True
    uploader.start()

    ports.wait(lambda: len({r[0] for r in ports.reports}) == 2)

    reports = {task_id: (report, error) for task_id, _, report, error in ports.reports}
    assert reports["tsk-1"] == (UploadReport.UPLOADED, None)
    report, error = reports["tsk-2"]
    assert report is UploadReport.UNUPLOADABLE and error is not None
    assert "journal" in error
    assert uploader._thread is not None and uploader._thread.is_alive()
    uploader.stop(0.5)


def test_a_success_that_cannot_be_recorded_ends_its_custody(tmp_path) -> None:
    class _Failing(ArtifactCustody):
        def record(self, journal: UploadJournal) -> None:
            if journal.task_id:
                raise OSError(28, "No space left on device")
            super().record(journal)

    root = tmp_path / "content"
    uploader = _uploader(root, _Ports(), custody=_Failing(root))
    journal = _stage(uploader, tmp_path, {"a.bin": b"alpha"}, submit=False)

    with pytest.raises(OSError):
        uploader.submit(journal, "tsk-1", {"artifacts_pending": True})

    assert not ArtifactCustody(root).publication_dir("org", "dsp-1").exists()


def test_a_custody_copy_that_changed_reports_the_upload_unfinishable(
    tmp_path,
) -> None:
    ports = _Ports()
    ports.wrong = True
    uploader = _uploader(tmp_path / "content", ports)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})

    ports.wait(lambda: ports.reports)

    task_id, dispatch_id, report, error = ports.reports[0]
    assert (task_id, dispatch_id, report) == (
        "tsk-1",
        "dsp-1",
        UploadReport.UNUPLOADABLE,
    )
    assert error is not None and "changed" in error
    uploader.stop(1.0)


def test_one_unuploadable_member_fails_only_the_tasks_that_own_it(tmp_path) -> None:
    ports = _Ports()
    root = tmp_path / "content"
    uploader = _uploader(root, ports)
    journal = uploader.begin("dsp-1", "org")
    _stage(
        uploader,
        tmp_path,
        {"a.bin": b"alpha", "b.bin": b"beta"},
        journal=journal,
        submit=False,
    )
    _stage(
        uploader,
        tmp_path,
        {"c.bin": b"gamma"},
        task_id="tsk-2",
        journal=journal,
        submit=False,
    )
    gone = journal.tasks["tsk-1"].members[0]
    ArtifactCustody(root).source(journal, gone).unlink()
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    uploader.start()

    ports.wait(lambda: len({r[0] for r in ports.reports}) == 2)

    reports = {task_id: (report, error) for task_id, _, report, error in ports.reports}
    assert reports["tsk-2"] == (UploadReport.UPLOADED, None)
    report, error = reports["tsk-1"]
    assert report is UploadReport.UNUPLOADABLE and error is not None
    assert sorted(data for _, _, data in ports.stored) == [b"beta", b"gamma"]
    uploader.stop(1.0)


def test_a_failed_task_is_reported_once_its_other_files_are_stored(tmp_path) -> None:
    ports = _Ports()
    root = tmp_path / "content"
    uploader = _uploader(root, ports)
    journal = _stage(
        uploader, tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"}, submit=False
    )
    ArtifactCustody(root).source(journal, journal.tasks["tsk-1"].members[0]).unlink()
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    uploader.start()

    ports.wait(lambda: ports.reports)

    assert ports.reports[0][2] is UploadReport.UNUPLOADABLE
    assert ports.stored_at_report[0] == [b"beta"]
    uploader.stop(1.0)


def test_a_pending_answer_reports_no_failed_task_still_uploading(tmp_path) -> None:
    ports = _Ports()
    ports.poisoned.add(b"beta")
    root = tmp_path / "content"
    clock = _Clock()
    uploader = _uploader(root, ports, clock)
    journal = _stage(
        uploader, tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"}, submit=False
    )
    ArtifactCustody(root).source(journal, journal.tasks["tsk-1"].members[0]).unlink()
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    uploader.start()
    ports.wait(lambda: ports.attempts >= 2)
    assert journal.tasks["tsk-1"].failed is not None

    uploader.settle(_settled(UploadDisposition.PENDING))
    _nudge(uploader, clock, 1)
    time.sleep(0.3)

    assert ports.reports == []
    uploader.stop(0.2)


def test_an_unexpected_failure_backs_off_only_its_own_dispatch(tmp_path) -> None:
    ports = _Ports()
    ports.poisoned.add(b"poison")
    uploader = _uploader(tmp_path / "content", ports, _Clock())
    _stage(uploader, tmp_path, {"a.bin": b"poison"})
    _stage(
        uploader,
        tmp_path,
        {"b.bin": b"fine"},
        task_id="tsk-2",
        journal=uploader.begin("dsp-2", "org"),
    )
    uploader.start()

    ports.wait(lambda: ports.reports)

    assert ports.reports[0][:3] == ("tsk-2", "dsp-2", UploadReport.UPLOADED)
    uploader.stop(0.5)


def test_a_member_that_fails_unexpectedly_holds_up_none_of_its_dispatch(
    tmp_path,
) -> None:
    ports = _Ports()
    ports.poisoned.add(b"poison")
    uploader = _uploader(tmp_path / "content", ports)
    journal = uploader.begin("dsp-1", "org")
    _stage_file(uploader, journal, "tsk-1", _write(tmp_path, "a.bin", b"poison"))
    _stage_file(uploader, journal, "tsk-2", _write(tmp_path, "b.bin", b"fine"))
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    uploader.start()

    ports.wait(lambda: ports.reports)
    uploader.stop(0)

    assert [data for _, _, data in ports.stored] == [b"fine"]
    assert ports.reports[0][:3] == ("tsk-2", "dsp-1", UploadReport.UPLOADED)


def test_another_workers_custody_is_kept_while_a_task_it_names_is_unanswered(
    tmp_path,
) -> None:
    root = tmp_path / "content"
    owner = _uploader(root, _Ports(), alias="w2", worker_id="wkr-2")
    journal = _stage(owner, tmp_path, {"a.bin": b"alpha"}, submit=False)
    ports = _Ports()
    mine = _uploader(root, ports, _Clock())
    mine.start()
    ports.wait(lambda: ports.reports)

    _stage(owner, tmp_path, {"b.bin": b"beta"}, task_id="tsk-2", journal=journal)
    mine.settle(_settled(UploadDisposition.REJECTED))
    assert list(ArtifactCustody(root).journals())

    mine.settle(_settled(UploadDisposition.REJECTED, task_id="tsk-2"))
    assert list(ArtifactCustody(root).journals()) == []
    mine.stop(0.5)


def test_a_crashed_peers_unreported_custody_is_released_once_control_rejects_it(
    tmp_path,
) -> None:
    root = tmp_path / "content"
    _stage(
        _uploader(root, _Ports(), alias="w2", worker_id="wkr-2"),
        tmp_path,
        {"a.bin": b"alpha"},
        submit=False,
    )
    ports = _Ports()
    clock = _Clock()
    mine = _uploader(root, ports, clock)
    mine.start()
    ports.wait(lambda: ports.reports)

    mine.settle(_settled(UploadDisposition.REJECTED))
    _nudge(mine, clock, EVIDENCE_SCAN_SEC + 1)
    time.sleep(0.2)
    mine.stop(0.5)

    assert list(ArtifactCustody(root).journals()) == []
    assert mine._publications == {}
    assert len(ports.reports) == 1


def test_an_earlier_registrations_custody_is_reported_and_never_uploaded(
    tmp_path,
) -> None:
    root = tmp_path / "content"
    ports = _Ports()
    first = _uploader(root, ports, worker_id="wkr-0")
    _stage(first, tmp_path, {"a.bin": b"alpha"})

    later = _uploader(root, ports)
    later.start()
    ports.wait(lambda: ports.reports)

    assert ports.stored == []
    assert ports.reports[0][2] is UploadReport.ABANDONED
    assert later.dispatch_of("tsk-1") is None

    later.settle(_settled(UploadDisposition.LOST))
    assert list(ArtifactCustody(root).journals()) == []
    later.stop(1.0)


def test_another_workers_custody_is_asked_about_and_released_once_settled(
    tmp_path,
) -> None:
    root = tmp_path / "content"
    ports = _Ports()
    clock = _Clock()
    _stage(
        _uploader(root, _Ports(), alias="w2", worker_id="wkr-2"),
        tmp_path,
        {"a.bin": b"alpha"},
    )

    mine = _uploader(root, ports, clock)
    mine.start()
    ports.wait(lambda: ports.reports)
    assert ports.reports[0][2] is UploadReport.ABANDONED

    mine.settle(_settled(UploadDisposition.PENDING))
    _nudge(mine, clock, RESEND_SEC + 1)
    time.sleep(0.2)
    assert len(ports.reports) == 1 and list(ArtifactCustody(root).journals())

    _nudge(mine, clock, EVIDENCE_RESEND_SEC)
    ports.wait(lambda: len(ports.reports) == 2)
    mine.settle(_settled(UploadDisposition.DURABLE))
    assert list(ArtifactCustody(root).journals()) == []
    assert ports.stored == []
    mine.stop(0.5)


def test_a_rebound_worker_stops_uploading_what_it_staged(tmp_path) -> None:
    ports = _Ports()
    ports.outage = 1_000
    uploader = _uploader(tmp_path / "content", ports)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})

    uploader.rebind()

    assert uploader.dispatch_of("tsk-1") is None
    ports.wait(lambda: any(r[2] is UploadReport.ABANDONED for r in ports.reports))
    uploader.stop(0.5)


@pytest.mark.parametrize("finishes", [True, False])
def test_stop_drains_owed_uploads_within_its_budget(tmp_path, finishes) -> None:
    ports = _Ports()
    ports.outage = 0 if finishes else 1_000
    uploader = _uploader(tmp_path / "content", ports)
    uploader.start()
    _stage(uploader, tmp_path, {"a.bin": b"alpha"})

    uploader.stop(1.0)

    assert bool(ports.stored) is finishes


def test_a_large_tree_is_bookkept_in_linear_time(tmp_path) -> None:
    ports = _Ports()
    uploader = _uploader(tmp_path / "content", ports)
    files = {f"f{i}.bin": str(i).encode() for i in range(4000)}
    _stage(uploader, tmp_path, files)
    uploader.start()

    started = time.monotonic()
    ports.wait(lambda: ports.reports, timeout=30.0)

    assert len(ports.stored) == 4000
    assert time.monotonic() - started < 10.0
    uploader.stop(1.0)
