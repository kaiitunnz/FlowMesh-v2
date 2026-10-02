import asyncio
import logging
import stat
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from server.services import log_archiver
from server.services.log_archiver import TaskLogArchiver
from server.task import runtime as runtime_module
from server.task.models import TaskInfo, TaskStatus
from tests.server.task.test_v2_orchestration import FakeRegistry, _live_runtime


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
    logs_path = archiver._base_dir("tsk-1") / "logs" / "logs.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.touch()
    task_dir = logs_path.parent.parent
    for directory in (task_dir, logs_path.parent):
        directory.chmod(0o755)
    runtime.task_statuses.return_value = {"tsk-1": TaskStatus.DONE}

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

    logs_path = archiver._base_dir("tsk-1") / "logs" / "logs.jsonl"
    assert logs_path.read_text() == '{"message": "hi"}\n'
    for directory in (logs_path.parent.parent, logs_path.parent):
        assert _mode(directory) == 0o777


def test_finalizing_shares_the_logs_directory_it_writes_into(
    archiver: TaskLogArchiver,
) -> None:
    archiver._finalize_manifest("tsk-1")

    logs_path = archiver._base_dir("tsk-1") / "logs" / "logs.jsonl"
    assert logs_path.is_file()
    for directory in (logs_path.parent.parent, logs_path.parent):
        assert _mode(directory) == 0o777


def test_a_tick_builds_no_task_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _live_runtime(FakeRegistry())
    asyncio.run(
        runtime.register(
            "owner",
            "org",
            "apiVersion: flowmesh/v1\nkind: EchoTask\nmetadata: {name: w}\nspec:\n"
            "  taskType: echo\n  stages:\n"
            "    - {name: a, spec: {data: {type: list, items: [x]}}}\n"
            "    - {name: b, spec: {data: {type: list, items: [y]}}}\n",
            format="native",
        )
    )
    built: list[str] = []

    class _Counting(TaskInfo):
        def __init__(self, **data: Any) -> None:
            built.append(data["task_id"])
            super().__init__(**data)

    monkeypatch.setattr(runtime_module, "TaskInfo", _Counting)
    archiver = TaskLogArchiver(
        MagicMock(), runtime, tmp_path, logging.getLogger("test.log_archiver")
    )
    tracked: list[str] = []
    monkeypatch.setattr(
        archiver, "_ensure_task", lambda task_id, now: tracked.append(task_id)
    )

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert built == []
    assert sorted(tracked) == sorted(runtime.tasks)
