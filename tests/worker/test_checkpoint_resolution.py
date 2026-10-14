"""A checkpoint a spec names resolves to that checkpoint or fails the task; a model
never starts from its base weights in its place."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors import vllm_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.utils import checkpoints
from worker.executors.utils.artifacts import artifact_ref_text
from worker.executors.utils.checkpoints import resolve_checkpoint_load
from worker.executors.vllm_executor import VLLMExecutor


@pytest.mark.parametrize("source", [{"type": "local"}, {"type": "http"}, {}])
def test_a_hydrated_checkpoint_resolves_whatever_type_it_declares(
    tmp_path: Path, source: dict[str, str]
) -> None:
    hydrated = tmp_path / "model"
    hydrated.mkdir()

    assert (
        resolve_checkpoint_load({**source, "url": hydrated.as_posix()}, tmp_path)
        == hydrated
    )


def test_a_task_id_alone_reaching_a_worker_is_not_guessed_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESULTS_DIR", tmp_path.as_posix())
    (tmp_path / "tsk-train" / "artifacts" / "final_model").mkdir(parents=True)

    with pytest.raises(ExecutionError, match="path is required"):
        resolve_checkpoint_load({"task_id": "tsk-train"}, tmp_path)


def test_a_base_url_fetches_the_artifact_from_that_deployment(tmp_path: Path) -> None:
    fetched: list[dict[str, Any]] = []

    def download(load_cfg: dict[str, Any], out_dir: Path) -> Path:
        fetched.append(load_cfg)
        return tmp_path

    with patch.object(checkpoints, "download_and_unpack", side_effect=download):
        resolve_checkpoint_load(
            {"task_id": "tsk-1", "filename": "final_model", "base_url": "https://b/"},
            tmp_path,
        )

    assert fetched[0]["url"] == "https://b/api/v1/results/tsk-1/files/final_model"


def _start(checkpoint_cfg: dict[str, Any]) -> MagicMock:
    llm = MagicMock()
    with patch.object(vllm_executor, "LLM", llm):
        VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._init_vllm_engine(
            ident="org/base",
            vllm_cfg={"tensor_parallel_size": 1},
            checkpoint_cfg=checkpoint_cfg,
            new_inference_spec={"checkpoint": checkpoint_cfg},
            requested_gpu_count=1,
            revision=None,
            extra_llm_kwargs={},
            adjust_tp=lambda size: size,
            task_ids=None,
        )
    return llm


@pytest.mark.parametrize("source", [{"type": "local"}, {"type": "http"}, {}])
def test_vllm_starts_from_a_hydrated_checkpoint(
    tmp_path: Path, source: dict[str, str]
) -> None:
    llm = _start({**source, "url": tmp_path.as_posix()})

    assert llm.call_args.kwargs["model"] == tmp_path.as_posix()


@pytest.mark.parametrize(
    "load",
    [
        {"type": "local", "path": "/nonexistent/checkpoint"},
        {"type": "local", "task_id": "tsk-train"},
        {"type": "http", "url": "flowmesh-artifact:dsp-1.0"},
    ],
)
def test_vllm_fails_rather_than_serve_the_base_model(load: dict[str, Any]) -> None:
    with pytest.raises(ExecutionError):
        _start(load)


def test_a_checkpoints_ref_in_data_reads_as_its_revisions_download_path() -> None:
    upstream = BaseExecutorResult(
        _artifacts=ArtifactContext(task_id="tsk-train", checkpoints="step-30")
    )

    assert (
        artifact_ref_text({"path": "checkpoints"}, {"train": upstream}, "train")
        == "/api/v1/results/tsk-train/files/checkpoints?revision=step-30"
    )
    assert (
        artifact_ref_text({"path": "final_model"}, {"train": upstream}, "train")
        == "/api/v1/results/tsk-train/files/final_model"
    )
