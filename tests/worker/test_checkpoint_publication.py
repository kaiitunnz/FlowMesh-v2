"""A trainer's saved checkpoints publish as revisions while it runs, from snapshots its
rotation cannot remove."""

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from shared.content import (
    ContentStoreError,
    SharedFilesystemObjectStore,
    TreeManifest,
    WriteCancelled,
)
from shared.tasks.artifact_binding import ArtifactBinding
from shared.utils.nofollow import PathRefused
from worker.content import artifacts, checkpoints
from worker.content.artifacts import (
    close_spool,
    open_spool,
    spool_checkpoint,
    spool_failed,
    spool_is_open,
    spooled_steps,
)
from worker.content.checkpoints import CheckpointPublishFailed, CheckpointUploader
from worker.executors.utils.checkpoint_spool import CheckpointSpool


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    (checkpoint / "sub").mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(f"model-{step}".encode())
    (checkpoint / "sub" / "optimizer.pt").write_bytes(b"optimizer")
    return checkpoint


def test_a_snapshot_survives_the_trainer_rotating_its_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", 10)
    spool = tmp_path / "spool"

    snapshot = spool_checkpoint(checkpoint, spool, 10)
    for path in sorted(checkpoint.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()

    assert snapshot is not None
    assert (snapshot / "checkpoint-10" / "model.bin").read_bytes() == b"model-10"
    assert spooled_steps(spool) == [(10, snapshot)]
    assert spool_checkpoint(checkpoint, spool, 10) is None


def test_a_half_written_snapshot_is_not_spooled(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    (spool / ".step-3.partial" / "checkpoint-3").mkdir(parents=True)

    assert spooled_steps(spool) == []


def _open_spool(spool: Path) -> None:
    spool.mkdir(parents=True, exist_ok=True)
    open_spool(spool)


def _uploader(
    tmp_path: Path, store: Any, reports: list[ArtifactBinding]
) -> CheckpointUploader:
    return CheckpointUploader(
        store,
        "org",
        tmp_path / "spool",
        reports.append,
        logging.getLogger("test"),
        poll_sec=0.01,
        backoff_sec=0.0,
    )


def test_each_spooled_checkpoint_publishes_once_as_its_step(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    reports: list[ArtifactBinding] = []
    uploader = _uploader(tmp_path, store, reports)
    uploader.start()
    for step in (20, 10):
        spool_checkpoint(
            _checkpoint(tmp_path / "checkpoints", step), tmp_path / "spool", step
        )

    published = uploader.drain()

    assert [(b.name, b.revision) for b in published] == [
        ("checkpoints", "step-10"),
        ("checkpoints", "step-20"),
    ]
    assert sorted(b.revision for b in reports) == ["step-10", "step-20"]
    manifest = TreeManifest.decode(store.hydrate(published[0].reference))
    assert [m.path for m in manifest.members] == [
        "checkpoint-10/model.bin",
        "checkpoint-10/sub/optimizer.pt",
    ]
    assert not (tmp_path / "spool").exists()


def test_a_checkpoint_the_store_refuses_fails_the_drain(tmp_path: Path) -> None:
    class _Refusing(SharedFilesystemObjectStore):
        def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
            raise ContentStoreError("store unreachable")

    uploader = _uploader(tmp_path, _Refusing(tmp_path / "cas"), [])
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader.start()

    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()


def test_an_aborted_upload_leaves_no_spool(tmp_path: Path) -> None:
    uploader = _uploader(tmp_path, SharedFilesystemObjectStore(tmp_path / "cas"), [])
    uploader.start()
    (tmp_path / "spool" / ".step-1.partial").mkdir(parents=True)

    uploader.abort()

    assert not (tmp_path / "spool").exists()


@pytest.mark.parametrize("zero", [True, False])
def test_only_the_first_rank_snapshots_a_save(tmp_path: Path, zero: bool) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, 7)
    state = TrainerState(global_step=7)
    state.is_world_process_zero = zero

    _open_spool(tmp_path / "spool")

    CheckpointSpool(tmp_path / "spool").on_save(
        TrainingArguments(output_dir=root.as_posix(), report_to=[]),
        state,
        TrainerControl(),
    )

    assert bool(spooled_steps(tmp_path / "spool")) is zero


def test_a_save_with_no_uploader_spools_nothing(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, 7)

    _save(tmp_path / "spool", root, 7)

    assert spooled_steps(tmp_path / "spool") == []


class _Flaky(SharedFilesystemObjectStore):
    def __init__(self, root: Path, failures: int) -> None:
        super().__init__(root)
        self.failures = failures

    def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
        if self.failures:
            self.failures -= 1
            raise ContentStoreError("store unreachable")
        return super().write_file(scope, path, **kwargs)


def _save(spool: Path, root: Path, step: int, **kwargs: Any) -> TrainerControl:
    state = TrainerState(global_step=step)
    state.is_world_process_zero = True
    control = TrainerControl()
    CheckpointSpool(spool, poll_sec=0.01, **kwargs).on_save(
        TrainingArguments(output_dir=root.as_posix(), report_to=[]), state, control
    )
    return control


def test_a_transient_store_failure_is_retried(tmp_path: Path) -> None:
    uploader = _uploader(tmp_path, _Flaky(tmp_path / "cas", failures=2), [])
    uploader.start()
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)

    assert [b.revision for b in uploader.drain()] == ["step-1"]


def test_a_persistent_failure_stops_training_at_the_next_save(tmp_path: Path) -> None:
    uploader = _uploader(tmp_path, _Flaky(tmp_path / "cas", failures=99), [])
    uploader.start()
    root = tmp_path / "checkpoints"
    _checkpoint(root, 1)

    assert not _save(tmp_path / "spool", root, 1).should_training_stop
    deadline = time.monotonic() + 10
    while not spool_failed(tmp_path / "spool") and time.monotonic() < deadline:
        time.sleep(0.01)
    _checkpoint(root, 2)

    assert _save(tmp_path / "spool", root, 2).should_training_stop
    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()


def test_a_save_waits_while_too_many_checkpoints_wait_to_publish(
    tmp_path: Path,
) -> None:
    spool = tmp_path / "spool"
    _open_spool(spool)
    root = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        _checkpoint(root, step)
        spool_checkpoint(root / f"checkpoint-{step}", spool, step)
    _checkpoint(root, 4)
    saved = threading.Event()

    def save() -> None:
        _save(spool, root, 4, max_pending=2)
        saved.set()

    saving = threading.Thread(target=save)
    saving.start()

    assert not saved.wait(0.3)
    for step in (1, 2):
        shutil.rmtree(spool / f"step-{step}")
    assert saved.wait(5)
    saving.join()
    assert [step for step, _ in spooled_steps(spool)] == [3, 4]


class _Endless(SharedFilesystemObjectStore):
    """A store whose writes run until the writer cancels them."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writes = 0

    def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
        self.writes += 1
        cancelled = kwargs["cancelled"]
        while not cancelled():
            time.sleep(0.01)
        raise WriteCancelled("cancelled")


def test_an_abort_interrupts_the_upload_under_way(tmp_path: Path) -> None:
    store = _Endless(tmp_path / "cas")
    uploader = _uploader(tmp_path, store, [])
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader.start()
    while not store.writes:
        time.sleep(0.01)

    started = time.monotonic()
    uploader.abort()

    assert time.monotonic() - started < 2
    assert store.writes == 1


def test_a_cancelled_task_stops_its_upload_without_retrying(tmp_path: Path) -> None:
    store = _Endless(tmp_path / "cas")
    ended = threading.Event()
    uploader = CheckpointUploader(
        store,
        "org",
        tmp_path / "spool",
        lambda binding: None,
        logging.getLogger("test"),
        cancelled=ended.is_set,
        poll_sec=0.01,
        backoff_sec=0.0,
    )
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader.start()
    while not store.writes:
        time.sleep(0.01)

    ended.set()
    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()
    assert store.writes == 1


def test_a_spool_planted_as_a_link_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "spool").symlink_to(outside)
    uploader = _uploader(tmp_path, SharedFilesystemObjectStore(tmp_path / "cas"), [])

    with pytest.raises(PathRefused):
        uploader.start()
    assert list(outside.iterdir()) == []


def test_a_cancelled_upload_lets_the_trainer_finish(tmp_path: Path) -> None:
    out = tmp_path / "out"
    cancelled = threading.Event()
    uploader = CheckpointUploader(
        SharedFilesystemObjectStore(tmp_path / "cas"),
        "org",
        tmp_path / "spool",
        lambda binding: None,
        logging.getLogger("test"),
        cancelled=cancelled.is_set,
        poll_sec=0.01,
        backoff_sec=0.0,
    )
    uploader.start()
    cancelled.set()
    finished = threading.Event()

    def train() -> None:
        for step in (1, 2, 3, 4):
            _checkpoint(out, step)
            _save(tmp_path / "spool", out, step)
        finished.set()

    threading.Thread(target=train, daemon=True).start()

    assert finished.wait(10), "a save waits on an uploader that stopped"
    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()


def test_an_uploader_that_stops_closes_its_spool(tmp_path: Path) -> None:
    uploader = _uploader(tmp_path, _Flaky(tmp_path / "cas", failures=99), [])
    uploader.start()
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader._thread.join(10)

    assert not uploader._thread.is_alive()
    assert spool_failed(tmp_path / "spool")
    assert not spool_is_open(tmp_path / "spool")
    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()


def test_a_spool_its_uploader_stopped_refreshing_goes_stale(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    _open_spool(spool)
    marker = spool / ".uploading"
    assert spool_is_open(spool)

    past = time.time() - artifacts.SPOOL_STALE_SEC - 1
    os.utime(marker, (past, past))

    assert not spool_is_open(spool)


def test_a_running_uploader_keeps_its_spool_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifacts, "SPOOL_STALE_SEC", 0.2)
    uploader = _uploader(tmp_path, SharedFilesystemObjectStore(tmp_path / "cas"), [])
    uploader.start()
    time.sleep(0.5)

    assert spool_is_open(tmp_path / "spool")
    uploader.drain()


def test_a_failed_or_closed_spool_takes_no_snapshot(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    root = tmp_path / "checkpoints"
    _open_spool(spool)
    (spool / ".failed").write_text("store unreachable")
    _checkpoint(root, 1)

    assert _save(spool, root, 1).should_training_stop
    close_spool(spool)
    (spool / ".failed").unlink()
    _checkpoint(root, 2)
    _save(spool, root, 2)

    assert spooled_steps(spool) == []


class _Stubborn(SharedFilesystemObjectStore):
    """A store whose writes ignore cancellation until released."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.entered = threading.Event()
        self.release = threading.Event()

    def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
        self.entered.set()
        self.release.wait(10)
        kwargs.pop("cancelled", None)
        return super().write_file(scope, path, **kwargs)


def test_an_upload_outliving_its_abort_leaves_a_reused_descriptor_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoints, "_ABORT_WAIT_SEC", 0.1)
    store = _Stubborn(tmp_path / "cas")
    uploader = _uploader(tmp_path, store, [])
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader.start()
    assert store.entered.wait(10)
    descriptor = int(uploader._spool.name)

    uploader.abort()
    other = tmp_path / "other"
    _checkpoint(other / "step-1", 1)
    opened: list[int] = []
    try:
        while (fd := os.open(other, os.O_RDONLY)) < descriptor:
            opened.append(fd)
        opened.append(fd)
        if fd != descriptor:
            pytest.skip("the closed descriptor was not reused")
        store.release.set()
        uploader._thread.join(10)
    finally:
        for fd in opened:
            os.close(fd)

    assert (other / "step-1" / "checkpoint-1" / "model.bin").exists()
