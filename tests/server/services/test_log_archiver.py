import asyncio
import errno
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


class _Streams:
    """Redis log streams for the archiver: each ``publish`` lands in the next tick's
    read."""

    def __init__(self) -> None:
        self.redis = MagicMock()
        self.redis.get.return_value = None
        self.redis.xrange_telemetry.return_value = []
        self._pending: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._seq = 0
        self.redis.xread_telemetry.side_effect = self._read

    def publish(self, task_id: str, message: str) -> None:
        self._seq += 1
        entry = (f"{self._seq}-0", {"payload": f'{{"m": "{message}"}}'})
        self._pending.setdefault(task_id, []).append(entry)

    def _read(self, *args: Any, **kwargs: Any) -> list[Any]:
        rows = [(task_log_stream_key(t), batch) for t, batch in self._pending.items()]
        self._pending = {}
        return rows


def _streaming_archiver(
    tmp_path: Path, statuses: dict[str, str], flush_max_entries: int = 1
) -> tuple[TaskLogArchiver, _Streams]:
    streams = _Streams()
    runtime = MagicMock()
    runtime.get_record.return_value = None
    runtime.task_statuses.return_value = statuses
    archiver = TaskLogArchiver(
        streams.redis,
        runtime,
        tmp_path,
        logging.getLogger("test"),
        flush_max_entries=flush_max_entries,
    )
    return archiver, streams


def _lines(archiver: TaskLogArchiver, task_id: str) -> list[str]:
    path = archiver._base_dir(task_id) / "logs" / "logs.jsonl"
    return path.read_text().splitlines() if path.exists() else []


def _failing_for(
    task_dir: Path, failures: list[int]
) -> Any:  # prepare_output_dir failing for one task while failures[0] > 0
    prepare = log_archiver.prepare_output_dir

    def _prepare(base_dir: Path) -> None:
        if base_dir == task_dir and failures[0] != 0:
            failures[0] -= 1
            raise OSError(errno.ENOSPC, "No space left on device")
        prepare(base_dir)

    return _prepare


def test_a_transient_write_error_keeps_every_line_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    monkeypatch.setattr(
        log_archiver,
        "prepare_output_dir",
        _failing_for(archiver._base_dir("tsk-1"), [2]),
    )

    with patch.object(log_archiver.time, "sleep"):
        for line in ("one", "two", "three"):
            streams.publish("tsk-1", line)
            archiver._tick()

    assert _lines(archiver, "tsk-1") == [
        '{"m": "one"}',
        '{"m": "two"}',
        '{"m": "three"}',
    ]
    streams.redis.set_value.assert_called_once()


def test_a_persistent_write_error_drops_after_its_bound_and_never_stalls_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-a": TaskStatus.DISPATCHED, "tsk-b": TaskStatus.DISPATCHED}
    )
    monkeypatch.setattr(
        log_archiver,
        "prepare_output_dir",
        _failing_for(archiver._base_dir("tsk-a"), [-1]),
    )
    ticks = log_archiver._MAX_FLUSH_FAILURES + 2

    with patch.object(log_archiver.time, "sleep"):
        for tick in range(ticks):
            streams.publish("tsk-a", f"a{tick}")
            streams.publish("tsk-b", f"b{tick}")
            archiver._tick()
            assert len(_lines(archiver, "tsk-b")) == tick + 1

    assert len(archiver._buffers["tsk-a"]) < log_archiver._MAX_FLUSH_FAILURES


def test_a_finished_tasks_last_lines_are_archived(tmp_path: Path) -> None:
    # A buffer short of a full flush, read in the tick that finds the task finished.
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, flush_max_entries=100
    )
    streams.publish("tsk-1", "last")
    archiver._runtime.list_tasks.return_value = [  # type: ignore[attr-defined]
        TaskInfo.model_construct(task_id="tsk-1", status=TaskStatus.DONE)
    ]

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert _lines(archiver, "tsk-1") == ['{"m": "last"}']
