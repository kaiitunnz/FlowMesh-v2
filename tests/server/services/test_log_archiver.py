import logging
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from server.services import log_archiver
from server.services.log_archiver import TaskLogArchiver
from server.task.models import TaskStatus


@pytest.fixture
def runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.get_record.return_value = None
    return runtime


@pytest.fixture
def archiver(tmp_path: Path, runtime: MagicMock) -> TaskLogArchiver:
    redis = MagicMock()
    redis.get.return_value = None
    return TaskLogArchiver(
        redis=redis,
        runtime=runtime,
        results_dir=tmp_path,
        logger=logging.getLogger("test.log_archiver"),
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_probing_an_archived_task_leaves_its_directory_alone(
    archiver: TaskLogArchiver, runtime: MagicMock
) -> None:
    logs_path = archiver._logs_path("tsk-1")
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.touch()
    task_dir = logs_path.parent.parent
    for directory in (task_dir, logs_path.parent):
        directory.chmod(0o755)
    runtime.list_tasks.return_value = [
        SimpleNamespace(task_id="tsk-1", status=TaskStatus.DONE)
    ]

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert sorted(p.name for p in task_dir.iterdir()) == ["logs"]
    assert _mode(task_dir) == 0o755
    assert _mode(logs_path.parent) == 0o755


def test_a_flush_shares_the_logs_directory_it_writes_into(
    archiver: TaskLogArchiver,
) -> None:
    archiver._ensure_task("tsk-1", 0.0)

    archiver._flush_task("tsk-1", [("1-0", {"payload": '{"message": "hi"}'})])

    logs_path = archiver._logs_path("tsk-1")
    assert logs_path.read_text() == '{"message": "hi"}\n'
    for directory in (logs_path.parent.parent, logs_path.parent):
        assert _mode(directory) == 0o777


def test_finalizing_shares_the_logs_directory_it_writes_into(
    archiver: TaskLogArchiver,
) -> None:
    archiver._finalize_manifest("tsk-1")

    logs_path = archiver._logs_path("tsk-1")
    assert logs_path.is_file()
    for directory in (logs_path.parent.parent, logs_path.parent):
        assert _mode(directory) == 0o777
