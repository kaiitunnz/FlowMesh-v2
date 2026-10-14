"""Tests for a task's output directory."""

from pathlib import Path
from unittest.mock import MagicMock

from shared.utils.task_output import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    prepare_output_dir,
)
from worker.runner import Runner


class TestPrepareOutputDir:
    def test_creates_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        assert out.exists()
        assert (out / LOGS_DIR).exists()
        assert (out / ARTIFACTS_DIR).exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        prepare_output_dir(out)  # should not raise
        assert out.exists()


class TestRunnerOutputDir:
    def test_resolve_output_dir_uses_canonical_path(self, tmp_path: Path) -> None:
        """Test that _resolve_output_dir creates and returns the canonical path
        `<results_dir>/<task_id>`."""
        results_dir = tmp_path / "results"
        runner = Runner(
            lifecycle=MagicMock(),
            task_stream=[],
            results_dir=results_dir,
            hardware=MagicMock(),
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

        task_id = "task-123"
        out_dir = runner._resolve_output_dir(task_id)

        assert out_dir == results_dir / task_id
        assert out_dir.exists()
        assert out_dir.is_dir()
