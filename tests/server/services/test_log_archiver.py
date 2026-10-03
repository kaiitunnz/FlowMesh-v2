import asyncio
import logging
import os
import socket
import stat
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from server.clients.redis import task_log_stream_key
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


def _flush_within(archiver: TaskLogArchiver, task_id: str, bound: float) -> bool:
    """Whether a flush of one entry for ``task_id`` returns within ``bound`` s."""
    archiver._ensure_task(task_id, 0.0)
    flush = threading.Thread(
        target=archiver._flush_task,
        args=(task_id, [("1-0", {"payload": '{"message": "hi"}'})]),
        daemon=True,
    )
    flush.start()
    flush.join(bound)
    return not flush.is_alive()


def test_a_fifo_at_the_log_file_never_blocks_the_archiver(
    archiver: TaskLogArchiver,
) -> None:
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    os.mkfifo(logs / "logs.jsonl")

    assert _flush_within(archiver, "tsk-1", 2.0)


def test_a_fifo_with_a_reader_never_blocks_the_archiver(
    archiver: TaskLogArchiver,
) -> None:
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    os.mkfifo(logs / "logs.jsonl")
    reader = os.open(logs / "logs.jsonl", os.O_RDONLY | os.O_NONBLOCK)
    try:
        assert _flush_within(archiver, "tsk-1", 2.0)
        assert os.read(reader, 1 << 16) == b""
    finally:
        os.close(reader)


@pytest.mark.parametrize("planted", ["directory", "socket"])
def test_a_task_whose_log_file_is_not_a_file_never_stops_another_tasks_archiving(
    tmp_path: Path, planted: str
) -> None:
    redis = MagicMock()
    redis.get.return_value = None
    redis.xrange_telemetry.return_value = []
    redis.xread_telemetry.return_value = [
        (
            task_log_stream_key(task_id),
            [("1-0", {"payload": f'{{"task": "{task_id}"}}'})],
        )
        for task_id in ("tsk-a", "tsk-b")
    ]
    runtime = MagicMock()
    runtime.get_record.return_value = None
    runtime.task_statuses.return_value = {
        "tsk-a": TaskStatus.DISPATCHED,
        "tsk-b": TaskStatus.DONE,
    }
    archiver = TaskLogArchiver(
        redis, runtime, tmp_path, logging.getLogger("test"), flush_max_entries=1
    )
    planted_path = archiver._base_dir("tsk-a") / "logs" / "logs.jsonl"
    planted_path.parent.mkdir(parents=True)
    held = socket.socket(socket.AF_UNIX)
    if planted == "directory":
        planted_path.mkdir()
    else:
        held.bind(str(planted_path))
    try:
        with patch.object(log_archiver.time, "sleep"):
            archiver._tick()
    finally:
        held.close()

    task_b = archiver._base_dir("tsk-b")
    assert (task_b / "logs" / "logs.jsonl").read_text() == '{"task": "tsk-b"}\n'
    assert (task_b / "manifest.json").is_file()
