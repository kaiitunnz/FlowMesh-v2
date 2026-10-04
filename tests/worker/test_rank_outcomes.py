"""A torchrun launch returns rank 0's result or the first rank's own failure."""

from pathlib import Path
from typing import Any

import pytest
from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
)

from shared.schemas.result import SFTResult
from shared.tasks import TaskType
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import SFTSpecStrict
from shared.utils.manifest import scratch_dir
from tests.worker.factories import make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.utils import distributed


def _task() -> Any:
    spec = SFTSpecStrict(
        taskType=TaskType.SFT, model=ModelConfig(source=ModelSource(identifier="m"))
    )
    return make_worker_task_message(spec=spec, task_type=TaskType.SFT)


def _launch(out_dir: Path) -> SFTResult:
    return distributed.launch_ranks(
        nproc_per_node=2,
        module="worker.executors.sft_dist_entry",
        out_dir=out_dir,
        task=_task(),
        launcher_env_flag="FLAG",
        result_type=SFTResult,
    )


def _ranks(monkeypatch: pytest.MonkeyPatch, *ranks: tuple[str, Any]) -> None:
    """Stand torchrun in for ranks that each return a result or raise."""

    def torchrun(*, module_args: list[str], **_: Any) -> None:
        out_dir = Path(module_args[1])
        failed: BaseException | None = None
        for rank, outcome in ranks:
            monkeypatch.setenv("RANK", rank)

            def run(outcome: Any = outcome) -> SFTResult:
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            try:
                distributed.run_rank(out_dir, run)
            except Exception as exc:
                failed = failed or exc
        if failed is not None:
            raise RuntimeError("ChildFailedError: rank exited with code 1")

    monkeypatch.setattr(distributed, "run_torchrun", torchrun)


def test_only_rank_zero_hands_back_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ranks(
        monkeypatch,
        ("1", SFTResult(model_name="rank-1")),
        ("0", SFTResult(model_name="rank-0")),
        ("1", SFTResult(model_name="rank-1")),
    )
    result = _launch(tmp_path)
    assert result.model_name == "rank-0"
    assert result.spawned_torchrun
    assert {p.name for p in scratch_dir(tmp_path).iterdir() if p.is_file()} == {
        "distributed_result.json"
    }


def test_a_launch_without_a_rank_zero_result_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    (scratch_dir(tmp_path) / "distributed_result.json").write_text(
        SFTResult(model_name="an-earlier-launch").model_dump_json()
    )
    _ranks(monkeypatch, ("1", SFTResult(model_name="rank-1")))
    with pytest.raises(ExecutionError, match="no result from rank 0"):
        _launch(tmp_path)


def test_a_failed_launch_reports_the_first_ranks_own_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ranks(
        monkeypatch,
        ("1", ExecutionError("CUDA out of memory", retryable=True)),
        ("0", RuntimeError("NCCL watchdog timeout")),
    )
    with pytest.raises(ExecutionError, match="^CUDA out of memory$") as raised:
        _launch(tmp_path)
    assert raised.value.retryable


def test_a_rank_failure_that_is_not_an_execution_error_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ranks(monkeypatch, ("0", ValueError("dataset has no 'text' column")))
    with pytest.raises(ExecutionError, match="no 'text' column") as raised:
        _launch(tmp_path)
    assert not raised.value.retryable


def test_a_launch_that_fails_before_any_rank_reports_names_the_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def torchrun(**_: Any) -> None:
        raise RuntimeError("torchrun could not start")

    monkeypatch.setattr(distributed, "run_torchrun", torchrun)
    with pytest.raises(ExecutionError, match="torchrun could not start") as raised:
        _launch(tmp_path)
    assert not raised.value.retryable


def test_a_rank_killed_before_it_records_a_failure_is_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def torchrun(**_: Any) -> None:
        failure = ProcessFailure(local_rank=1, pid=4242, exitcode=-9, error_file="")
        raise ChildFailedError(
            name="worker.executors.sft_dist_entry", failures={1: failure}
        )

    monkeypatch.setattr(distributed, "run_torchrun", torchrun)
    with pytest.raises(ExecutionError, match="distributed training failed") as raised:
        _launch(tmp_path)
    assert raised.value.retryable
