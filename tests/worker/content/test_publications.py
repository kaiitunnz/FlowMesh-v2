"""The publication outbox: it keeps what each publication names, repeats the report
until control answers it for good, and stores a refused object again from its copy."""

import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentReference, StoreRefusal, reference_for
from shared.content.filesystem import SharedFilesystemObjectStore
from shared.tasks.artifact_binding import ArtifactBinding, step_revision
from shared.tasks.publication import (
    CHECKPOINT_FIELD,
    Publication,
    PublicationAnswer,
    PublicationDisposition,
)
from worker.content import publications
from worker.content.artifacts import ArtifactPublisher
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
        self.upload_attempts = 0
        self.failed: list[tuple[str, str, str]] = []
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
        with self.cv:
            self.upload_attempts += 1
            self.cv.notify_all()
        if self.fail_upload:
            raise OSError("store down")
        data = path.read_bytes()
        with self.cv:
            self.uploaded.append((task_id, data))
            self.cv.notify_all()
        return self.store.write(scope, data, media_type=media_type)

    def fail(self, task_id: str, dispatch_id: str, error: str) -> None:
        with self.cv:
            self.failed.append((task_id, dispatch_id, error))
            self.cv.notify_all()

    def wait(self, predicate: Any) -> None:
        with self.cv:
            assert self.cv.wait_for(predicate, timeout=5), "timed out"


@pytest.fixture
def rig(tmp_path: Path) -> Any:
    ports = _Ports(tmp_path)
    clock = _Clock()
    outbox = PublicationOutbox(
        tmp_path / "outbox",
        lambda: "wkr-1",
        ports.send,
        ports.upload,
        ports.fail,
        clock=clock,
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
    answer: PublicationDisposition | StoreRefusal,
    digests: list[str] | None = None,
    reason: str | None = None,
) -> PublicationAnswer:
    publication = Publication.of(
        "wkr-1", "dsp-1", "tsk-1", "TASK_SUCCEEDED", _success(reference)
    )
    assert publication is not None
    return PublicationAnswer(
        publication=publication,
        disposition=(
            PublicationDisposition.REFUSED
            if isinstance(answer, StoreRefusal)
            else answer
        ),
        refusal=answer if isinstance(answer, StoreRefusal) else None,
        digests=digests or [],
        reason=reason,
    )


def _kept_store(outbox: PublicationOutbox, ports: _Ports) -> Any:
    return outbox.keeping("dsp-1", ports.store)


def _kept(tmp_path: Path) -> set[str]:
    """The digests the outbox holds a copy of for dsp-1."""
    directory = tmp_path / "outbox" / "dsp-1"
    if not directory.exists():
        return set()
    return {path.name.split(".")[0] for path in directory.iterdir()}


def _until(predicate: Any) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        threading.Event().wait(0.01)


def _due_in(outbox: PublicationOutbox) -> float:
    """How long until the next report is due."""
    with outbox._cv:
        return outbox._wait_locked()


def test_a_report_is_repeated_until_committed_and_its_copies_go(rig) -> None:
    outbox, ports, clock, tmp_path = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.end("dsp-1")
    assert outbox.dispatch_of("tsk-1") == "dsp-1"

    _nudge(outbox, clock, RESEND_SEC + 1)
    ports.wait(lambda: len(ports.sent) == 1)
    outbox.answer(_answer(reference, PublicationDisposition.COMMITTED))
    assert outbox.dispatch_of("tsk-1") is None
    assert not (tmp_path / "outbox" / "dsp-1").exists()
    _nudge(outbox, clock, RESEND_SEC + 1)
    assert len(ports.sent) == 1


def test_a_rejected_report_ends_too(rig) -> None:
    outbox, ports, _, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(_answer(reference, PublicationDisposition.REJECTED))
    assert outbox.dispatch_of("tsk-1") is None


def test_a_committed_publications_copies_go_while_the_dispatch_runs(rig) -> None:
    outbox, ports, _, tmp_path = rig
    store = _kept_store(outbox, ports)
    reference = store.write(_SCOPE, b"result")
    unreported = store.write(_SCOPE, b"not reported yet")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    assert _kept(tmp_path) == {reference.content_digest, unreported.content_digest}

    outbox.answer(_answer(reference, PublicationDisposition.COMMITTED))

    assert _kept(tmp_path) == {unreported.content_digest}
    outbox.end("dsp-1")
    assert not (tmp_path / "outbox" / "dsp-1").exists()


def _checkpoint(
    outbox: PublicationOutbox, ports: _Ports, tmp_path: Path, step: int, weights: bytes
) -> tuple[ArtifactBinding, dict[str, Any]]:
    """Publish one checkpoint of a shared config and its own weights, and report it."""
    snapshot = tmp_path / f"step-{step}"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"{}")
    (snapshot / "weights.bin").write_bytes(weights)
    publisher = ArtifactPublisher(_kept_store(outbox, ports), _SCOPE)
    binding, _ = publisher.publish_tree("checkpoints", snapshot, step_revision(step))
    shutil.rmtree(snapshot)
    payload = {CHECKPOINT_FIELD: [binding.model_dump(mode="json")]}
    outbox.reported("TASK_UPDATE", "tsk-1", "dsp-1", payload)
    return binding, payload


def _committed(payload: dict[str, Any]) -> PublicationAnswer:
    publication = Publication.of("wkr-1", "dsp-1", "tsk-1", "TASK_UPDATE", payload)
    assert publication is not None
    return PublicationAnswer(
        publication=publication, disposition=PublicationDisposition.COMMITTED
    )


def test_a_rotating_run_keeps_only_the_checkpoints_still_unanswered(rig) -> None:
    outbox, ports, _, tmp_path = rig
    config = reference_for(_SCOPE, b"{}").content_digest
    published = []
    for step in range(1, 6):
        binding, payload = _checkpoint(
            outbox, ports, tmp_path, step, b"w" * (1 << 16) + bytes([step])
        )
        weights = reference_for(_SCOPE, b"w" * (1 << 16) + bytes([step]))
        published.append((binding, payload, weights.content_digest))
        if step > 1:
            previous, previous_payload, _ = published[-2]
            outbox.answer(_committed(previous_payload))
        assert _kept(tmp_path) == {
            binding.reference.content_digest,
            weights.content_digest,
            config,
        }

    outbox.answer(_committed(published[-1][1]))
    assert _kept(tmp_path) == set()
    assert outbox.dispatch_of("tsk-1") is None


def test_a_member_stored_again_for_an_unreported_checkpoint_outlives_its_last(
    rig,
) -> None:
    outbox, ports, _, tmp_path = rig
    _, first = _checkpoint(outbox, ports, tmp_path, 1, b"weights-1")
    config = reference_for(_SCOPE, b"{}")
    _kept_store(outbox, ports).write(_SCOPE, b"{}")

    outbox.answer(_committed(first))

    assert _kept(tmp_path) == {config.content_digest}


def test_a_missing_object_is_stored_again_from_its_copy_then_reported(rig) -> None:
    outbox, ports, clock, tmp_path = rig
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact bytes")
    reference = _kept_store(outbox, ports).write_file(_SCOPE, source)
    source.unlink()
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(
            reference,
            StoreRefusal.REMATERIALIZE,
            [reference.content_digest],
        )
    )
    _nudge(outbox, clock, 1.0)
    ports.wait(lambda: ports.uploaded and ports.sent)
    assert ports.uploaded == [("tsk-1", b"artifact bytes")]
    assert outbox.dispatch_of("tsk-1") == "dsp-1"


def test_a_store_failure_while_restoring_is_retried(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    ports.fail_upload = True
    outbox.answer(
        _answer(reference, StoreRefusal.REMATERIALIZE, [reference.content_digest])
    )
    _nudge(outbox, clock, 1.0)
    ports.wait(lambda: ports.upload_attempts == 1)
    _until(lambda: _due_in(outbox) == pytest.approx(2.0))
    assert ports.sent == []
    ports.fail_upload = False
    _nudge(outbox, clock, 2.0)
    ports.wait(lambda: ports.uploaded and ports.sent)


def test_a_retirement_is_waited_out_and_reported_again_storing_nothing(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(
            reference,
            StoreRefusal.RETRY_RETIREMENT,
            [reference.content_digest],
        )
    )
    assert _due_in(outbox) == pytest.approx(1.0)
    _nudge(outbox, clock, 1.0)
    ports.wait(lambda: len(ports.sent) == 1)
    assert ports.upload_attempts == 0


def test_a_missing_object_is_stored_again_only_after_a_backoff(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    for wait in (1.0, 2.0, 4.0):
        outbox.answer(
            _answer(
                reference,
                StoreRefusal.REMATERIALIZE,
                [reference.content_digest],
            )
        )
        assert _due_in(outbox) == pytest.approx(wait)
        sent = len(ports.sent)
        _nudge(outbox, clock, wait)
        ports.wait(lambda: len(ports.sent) == sent + 1)


@pytest.mark.parametrize("copy", ["not kept", "gone", "truncated"])
def test_an_object_with_no_intact_copy_fails_its_dispatch(rig, copy: str) -> None:
    outbox, ports, clock, tmp_path = rig
    store = _kept_store(outbox, ports)
    reference = store.write(_SCOPE, b"result")
    named = reference
    if copy == "not kept":
        named = reference_for(_SCOPE, b"never written")
    else:
        (kept,) = (tmp_path / "outbox" / "dsp-1").iterdir()
        if copy == "gone":
            kept.unlink()
        else:
            kept.write_bytes(b"res")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(reference, StoreRefusal.REMATERIALIZE, [named.content_digest])
    )
    _nudge(outbox, clock, 1.0)

    ports.wait(lambda: ports.failed and ports.sent)
    ((task_id, dispatch_id, error),) = ports.failed
    assert (task_id, dispatch_id) == ("tsk-1", "dsp-1")
    assert error.startswith("publication_unrepairable")
    assert ports.upload_attempts == 0
    assert outbox.dispatch_of("tsk-1") == "dsp-1"


def test_a_copy_that_fails_midway_leaves_nothing_kept(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox, ports, _, tmp_path = rig

    def torn(source: Any, copy: Any, length: int) -> None:
        copy.write(source.read(3))
        raise OSError("disk full")

    monkeypatch.setattr(publications, "link_opened", lambda *_: False)
    monkeypatch.setattr(publications.shutil, "copyfileobj", torn)
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact bytes")
    with source.open("rb") as opened:
        _kept_store(outbox, ports).write_completed(_SCOPE, opened)

    assert _kept(tmp_path) == set()
    with outbox._cv:
        assert outbox._dispatches["dsp-1"].kept == {}


def test_nothing_is_kept_for_a_dispatch_that_ended(rig) -> None:
    outbox, ports, _, tmp_path = rig
    store = _kept_store(outbox, ports)
    outbox.end("dsp-1")

    outbox.keep("dsp-1", reference_for(_SCOPE, b"late"), b"late")
    store.write(_SCOPE, b"later")

    assert outbox._dispatches == {}
    assert not (tmp_path / "outbox" / "dsp-1").exists()


def test_copies_are_removed_outside_the_outbox_lock(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox, ports, _, _ = rig
    held: list[bool] = []
    remove = publications.remove_below

    def removing(path: Path) -> None:
        free: list[bool] = []

        def probe() -> None:
            if outbox._cv.acquire(blocking=False):
                outbox._cv.release()
                free.append(True)

        prober = threading.Thread(target=probe)
        prober.start()
        prober.join()
        held.append(not free)
        remove(path)

    monkeypatch.setattr(publications, "remove_below", removing)
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(_answer(reference, PublicationDisposition.COMMITTED))
    outbox.end("dsp-1")

    assert held == [False, False]


def test_a_busy_lane_is_offered_again_shortly(rig) -> None:
    outbox, ports, clock, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    # However often the lane is full, the report waits no longer than a busy pause.
    for sent in range(1, 7):
        outbox.answer(_answer(reference, StoreRefusal.BUSY))
        _nudge(outbox, clock, 2 * BUSY_SEC + 0.1)
        ports.wait(lambda: len(ports.sent) == sent)


def test_an_answer_about_another_version_is_ignored(rig) -> None:
    outbox, ports, _, _ = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.answer(
        _answer(reference_for(_SCOPE, b"other"), PublicationDisposition.COMMITTED)
    )
    assert outbox.dispatch_of("tsk-1") == "dsp-1"


def test_a_new_registration_forgets_what_the_last_reported(rig) -> None:
    outbox, ports, _, tmp_path = rig
    reference = _kept_store(outbox, ports).write(_SCOPE, b"result")
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", _success(reference))
    outbox.clear()
    assert outbox.dispatch_of("tsk-1") is None
    assert not (tmp_path / "outbox" / "dsp-1").exists()


def test_a_report_naming_nothing_stored_is_not_owed(rig) -> None:
    outbox, _, _, _ = rig
    outbox.reported("TASK_SUCCEEDED", "tsk-1", "dsp-1", {"value": 1})
    outbox.reported("TASK_UPDATE", "tsk-1", "dsp-1", {"progress": 0.5})
    assert outbox.dispatch_of("tsk-1") is None
