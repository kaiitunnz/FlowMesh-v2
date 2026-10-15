"""The artifact uploader: it stores what a success staged, reports each step until
control answers, and treats an earlier registration's custody as evidence only."""

import threading
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentReference, ContentUnavailable, reference_for
from shared.tasks.artifact_upload import UploadDisposition, UploadReport, UploadSettled
from worker.content.custody import ArtifactCustody
from worker.content.uploads import ArtifactUploader


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
        self.cv = threading.Condition()

    def upload(
        self, task_id: str, scope: str, path: Path, media_type: str
    ) -> ContentReference:
        with self.cv:
            self.attempts += 1
            self.cv.notify_all()
        if self.outage:
            self.outage -= 1
            raise ContentUnavailable("store down")
        data = path.read_bytes()
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
        with self.cv:
            self.reports.append((task_id, dispatch_id, report, error))
            self.cv.notify_all()

    def wait(self, predicate: Any, timeout: float = 5.0) -> None:
        with self.cv:
            assert self.cv.wait_for(predicate, timeout), "timed out"


def _uploader(
    root: Path, ports: _Ports, clock: Any = None, alias: str = "w1"
) -> ArtifactUploader:
    kwargs = {"clock": clock} if clock is not None else {}
    return ArtifactUploader(
        ArtifactCustody(root),
        alias,
        identity=lambda: ("wkr-1", 1),
        upload=ports.upload,
        resend_success=ports.resend,
        report=ports.report,
        **kwargs,
    )


def _stage(
    uploader: ArtifactUploader, tmp_path: Path, files: dict[str, bytes]
) -> ContentReference:
    journal = uploader.begin("dsp-1", "org")
    last = None
    for name, data in files.items():
        path = tmp_path / "out" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        last = uploader.stage(journal, "tsk-1", path)
    uploader.submit(journal, "tsk-1", {"artifacts_pending": True})
    assert last is not None
    return last


def _settled(disposition: UploadDisposition) -> UploadSettled:
    return UploadSettled(task_id="tsk-1", dispatch_id="dsp-1", disposition=disposition)


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

    clock.now += 31
    with uploader._cv:
        uploader._cv.notify_all()
    ports.wait(lambda: len(ports.reports) == 2 and ports.successes)

    uploader.settle(_settled(UploadDisposition.DURABLE))
    clock.now += 31
    with uploader._cv:
        uploader._cv.notify_all()
    uploader.stop(1.0)
    assert len(ports.reports) == 2


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
        clock.now += 61
        with uploader._cv:
            uploader._cv.notify_all()
    ports.wait(lambda: ports.reports)

    assert ports.reports[0][2] is UploadReport.UPLOADED
    uploader.stop(1.0)


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


def test_an_earlier_registrations_custody_is_reported_and_never_uploaded(
    tmp_path,
) -> None:
    root = tmp_path / "content"
    ports = _Ports()
    first = _uploader(root, ports)
    journal = first.begin("dsp-1", "org")
    path = tmp_path / "a.bin"
    path.write_bytes(b"alpha")
    first.stage(journal, "tsk-1", path)
    first._custody.record(journal.model_copy(update={"task_id": "tsk-1"}))

    later = _uploader(root, ports)
    later.start()
    ports.wait(lambda: ports.reports)

    assert ports.stored == []
    assert ports.reports[0][2] is UploadReport.ABANDONED
    assert later.dispatch_of("tsk-1") is None

    later.settle(_settled(UploadDisposition.LOST))
    assert list(ArtifactCustody(root).journals()) == []
    later.stop(1.0)


def test_another_workers_custody_on_the_node_is_left_alone(tmp_path) -> None:
    root = tmp_path / "content"
    ports = _Ports()
    other = _uploader(root, _Ports(), alias="w2")
    _stage(other, tmp_path, {"a.bin": b"alpha"})

    mine = _uploader(root, ports, alias="w1")
    mine.start()
    mine.stop(0.5)

    assert ports.reports == []
    assert list(ArtifactCustody(root).journals())


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
