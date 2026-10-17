"""The publication outbox: it keeps what each publication names, repeats the report
until control answers it for good, and stores a refused object again from its copy."""

import threading
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentReference, reference_for
from shared.content.filesystem import SharedFilesystemObjectStore
from shared.tasks.publication import (
    BUSY_REASON,
    Publication,
    PublicationAnswer,
    PublicationDisposition,
)
from worker.content.publications import (
    BUSY_SEC,
    RESEND_SEC,
    PublicationOutbox,
)

_SCOPE = "org-a"


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Ports:
    def __init__(self, root: Path) -> None:
        self.cv = threading.Condition()
        self.sent: list[tuple[str, str, str]] = []
        self.uploaded: list[tuple[str, bytes]] = []
        self.store = SharedFilesystemObjectStore(root / "store")
        self.fail_upload = False

    def send(
        self, event_type: str, task_id: str, dispatch_id: str, payload: dict[str, Any]
    ) -> None:
        with self.cv:
            self.sent.append((event_type, task_id, dispatch_id))
            self.cv.notify_all()

    def upload(
        self, task_id: str, scope: str, path: Path, media_type: str
    ) -> ContentReference:
        if self.fail_upload:
            raise OSError("store down")
        data = path.read_bytes()
        with self.cv:
            self.uploaded.append((task_id, data))
            self.cv.notify_all()
        return self.store.write(scope, data, media_type=media_type)

    def wait(self, predicate: Any) -> None:
        with self.cv:
            assert self.cv.wait_for(predicate, timeout=5), "timed out"


@pytest.fixture
def rig(tmp_path: Path) -> Any:
    ports = _Ports(tmp_path)
    clock = _Clock()
    outbox = PublicationOutbox(
        tmp_path / "outbox", lambda: "wkr-1", ports.send, ports.upload, clock=clock
    )
    outbox.start()
    yield outbox, ports, clock, tmp_path
    outbox.stop(1.0)


def _nudge(outbox: PublicationOutbox, clock: _Clock, seconds: float) -> None:
    clock.now += seconds
    with outbox._cv:
        outbox._cv.notify_all()


def _success(reference: ContentReference) -> dict[str, Any]:
    return {"result_reference": reference.model_dump(mode="json")}


def _answer(
    reference: ContentReference,
    disposition: PublicationDisposition,
    digests: list[str] | None = None,
    reason: str | None = None,
) -> PublicationAnswer:
    publication = Publication.of(
        "wkr-1", "dsp-1", "tsk-1", "TASK_SUCCEEDED", _success(reference)
    )
    assert publication is not None
    return PublicationAnswer(
        publication=publication,
        disposition=disposition,
        digests=digests or [],
        reason=reason,
    )


def _kept_store(outbox: PublicationOutbox, ports: _Ports) -> Any:
    return outbox.keeping("dsp-1", ports.store)


def test_a_report_is_repeated_until_committed_and_its_copies_go(rig) -> None:
    outbox, ports, clock, tmp_path = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.end("dsp-1")
    assert outbox.owed() == 1
    assert outbox.dispatch_of("tsk-1") == "dsp-1"

    _nudge(outbox, clock, RESEND_SEC + 1)
    ports.wait(lambda: len(ports.sent) == 1)
    outbox.answer(_answer(reference, PublicationDisposition.COMMITTED))
    assert outbox.owed() == 0
    assert not (tmp_path / "outbox" / "dsp-1").exists()
    _nudge(outbox, clock, RESEND_SEC + 1)
    assert len(ports.sent) == 1


def test_a_rejected_report_ends_too(rig) -> None:
    outbox, ports, _, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(_answer(reference, PublicationDisposition.REJECTED))
    assert outbox.owed() == 0


def test_copies_stay_while_the_dispatch_runs(rig) -> None:
    outbox, ports, _, tmp_path = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(_answer(reference, PublicationDisposition.COMMITTED))
    assert (tmp_path / "outbox" / "dsp-1" / reference.content_digest).exists()
    outbox.end("dsp-1")
    assert not (tmp_path / "outbox" / "dsp-1").exists()


def test_a_missing_object_is_stored_again_from_its_copy_then_reported(rig) -> None:
    outbox, ports, _, tmp_path = rig
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact bytes")
    reference = _kept_store(outbox, ports).write_file(_SCOPE, source)
    source.unlink()
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(
            reference,
            PublicationDisposition.REMATERIALIZE,
            [reference.content_digest],
        )
    )
    ports.wait(lambda: ports.uploaded and ports.sent)
    assert ports.uploaded == [("tsk-1", b"artifact bytes")]
    assert outbox.owed() == 1


def test_a_store_failure_while_restoring_is_retried(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    ports.fail_upload = True
    outbox.answer(
        _answer(
            reference, PublicationDisposition.REMATERIALIZE, [reference.content_digest]
        )
    )
    threading.Event().wait(0.2)
    assert ports.sent == []
    ports.fail_upload = False
    _nudge(outbox, clock, 2.0)
    ports.wait(lambda: ports.uploaded and ports.sent)


def test_a_retirement_is_waited_out_before_the_object_is_stored_again(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(
            reference,
            PublicationDisposition.RETRY_RETIREMENT,
            [reference.content_digest],
        )
    )
    threading.Event().wait(0.2)
    assert ports.uploaded == []
    _nudge(outbox, clock, 1.5)
    ports.wait(lambda: ports.uploaded and ports.sent)


def test_a_busy_lane_is_offered_again_shortly(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(_answer(reference, PublicationDisposition.RETRY, reason=BUSY_REASON))
    _nudge(outbox, clock, 2 * BUSY_SEC + 0.1)
    ports.wait(lambda: len(ports.sent) == 1)


def test_an_answer_about_another_version_is_ignored(rig) -> None:
    outbox, ports, _, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(reference_for(_SCOPE, b"other"), PublicationDisposition.COMMITTED)
    )
    assert outbox.owed() == 1


def test_a_new_registration_forgets_what_the_last_reported(rig) -> None:
    outbox, ports, _, tmp_path = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.clear()
    assert outbox.owed() == 0
    assert not (tmp_path / "outbox" / "dsp-1").exists()


def test_a_report_naming_nothing_stored_is_not_owed(rig) -> None:
    outbox, _, _, _ = rig
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", {"value": 1})
    outbox.reported("TASK_UPDATE", "tsk-1", "dsp-1", {"progress": 0.5})
    assert outbox.owed() == 0
