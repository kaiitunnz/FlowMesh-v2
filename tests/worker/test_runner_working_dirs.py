"""A starting worker clears the working directories it left behind, and never those of
a worker sharing its results directory."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import TrainerControl, TrainerState, TrainingArguments

from shared.content import SharedFilesystemObjectStore
from worker.content.artifact_inputs import artifact_inputs_dir
from worker.content.artifacts import checkpoint_spool_dir
from worker.content.checkpoints import CheckpointUploader
from worker.executors.utils.checkpoint_spool import CheckpointSpool
from worker.runner import Runner


def _runner(results: Path, owner: str) -> Any:
    return SimpleNamespace(
        results_dir=results, _owner=owner, logger=logging.getLogger("test")
    )


def _start(results: Path, owner: str) -> None:
    Runner._clear_working_dirs(_runner(results, owner))


def _task_dir(results: Path, task_id: str, owner: str | None) -> Path:
    if owner is None:
        out = results / task_id
        out.mkdir(parents=True)
    else:
        out = Runner._resolve_output_dir(_runner(results, owner), task_id)
    hydrated = artifact_inputs_dir(out) / ("a" * 64)
    hydrated.mkdir(parents=True)
    (hydrated / "model.safetensors").write_bytes(b"weights")
    return out


def test_a_sibling_starting_leaves_a_running_task_its_inputs_and_spool(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    out = _task_dir(results, "tsk-on-w1", "w1")
    published: list = []
    uploader = CheckpointUploader(
        SharedFilesystemObjectStore(tmp_path / "cas"),
        "org",
        checkpoint_spool_dir(out),
        published.append,
        logging.getLogger("test"),
        poll_sec=0.01,
        backoff_sec=0.0,
    )
    uploader.start()

    _start(results, "w2")

    assert (artifact_inputs_dir(out) / ("a" * 64) / "model.safetensors").exists()
    root = out / "trainer"
    (root / "checkpoint-5").mkdir(parents=True)
    (root / "checkpoint-5" / "model.bin").write_bytes(b"x")
    state = TrainerState(global_step=5)
    state.is_world_process_zero = True
    CheckpointSpool(checkpoint_spool_dir(out)).on_save(
        TrainingArguments(output_dir=root.as_posix(), report_to=[]),
        state,
        TrainerControl(),
    )
    assert [binding.revision for binding in uploader.drain()] == ["step-5"]


def test_a_restarted_worker_clears_what_it_left_behind(tmp_path: Path) -> None:
    results = tmp_path / "results"
    own = _task_dir(results, "tsk-mine", "w1")
    (checkpoint_spool_dir(own) / "step-1").mkdir(parents=True)

    _start(results, "w1")

    assert not artifact_inputs_dir(own).exists()
    assert not checkpoint_spool_dir(own).exists()


def test_a_task_directory_no_worker_marked_is_left_alone(tmp_path: Path) -> None:
    results = tmp_path / "results"
    unmarked = _task_dir(results, "tsk-unmarked", None)

    _start(results, "w1")

    assert artifact_inputs_dir(unmarked).exists()


def test_a_task_a_sibling_took_over_is_left_to_it(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _task_dir(results, "tsk-retried", "w1")
    Runner._resolve_output_dir(_runner(results, "w2"), "tsk-retried")
    taken = results / "tsk-retried"

    _start(results, "w1")

    assert artifact_inputs_dir(taken).exists()


def test_a_planted_owner_link_is_not_followed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = _task_dir(results, "tsk-linked", None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("w1")
    (out / ".working-owner").symlink_to(elsewhere)

    _start(results, "w1")
    assert artifact_inputs_dir(out).exists()

    Runner._resolve_output_dir(_runner(results, "w1"), "tsk-linked")
    assert elsewhere.read_text() == "w1"
    assert not (out / ".working-owner").is_symlink()
