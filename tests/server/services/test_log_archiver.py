import asyncio
import errno
import logging
import os
import resource
import signal
import socket
import stat
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from server.clients.redis import (
    task_log_archive_last_id_key,
    task_log_archived_key,
    task_log_stream_key,
)
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
    redis.xrange_telemetry.return_value = []
    return TaskLogArchiver(
        redis=redis,
        runtime=runtime,
        results_dir=tmp_path,
        logger=logging.getLogger("test.log_archiver"),
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize("recorded", [True, False])
def test_probing_an_archived_task_leaves_its_directory_alone(
    archiver: TaskLogArchiver, runtime: MagicMock, recorded: bool
) -> None:
    # Recorded as archived, or archived before that was recorded, with lines.
    logs_path = archiver._base_dir("tsk-1") / "logs" / "logs.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    if recorded:
        logs_path.touch()
        archived_key = task_log_archived_key("tsk-1")
        cast(MagicMock, archiver._redis).get.side_effect = lambda key: (
            "1" if key == archived_key else None
        )
    else:
        logs_path.write_text('{"m": "old"}\n')
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

    archiver._flush_task("tsk-1", [("1-0", {"payload": '{"message": "hi"}'})], 0.0)

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
        args=(task_id, [("1-0", {"payload": '{"message": "hi"}'})], 0.0),
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
    """Redis log streams for the archiver: every published entry stays readable, and
    a read returns the entries past the id each requested stream names."""

    def __init__(self) -> None:
        self.redis = MagicMock()
        self.checkpoints: dict[str, str] = {}
        self.redis.get.side_effect = lambda key: self.checkpoints.get(key)
        self.redis.set_value.side_effect = self.checkpoints.__setitem__
        self.redis.delete.side_effect = lambda key: self.checkpoints.pop(key, None)
        self.redis.xread_telemetry.side_effect = self._read
        self.redis.xrange_telemetry.side_effect = self._range
        self._log: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._seq = 0

    def publish(self, task_id: str, message: str) -> None:
        self._seq += 1
        entry = (f"{self._seq}-0", {"payload": f'{{"m": "{message}"}}'})
        self._log.setdefault(task_log_stream_key(task_id), []).append(entry)

    def _after(self, key: str, last_id: str) -> list[tuple[str, dict[str, Any]]]:
        seq = int(last_id.split("-")[0])
        return [e for e in self._log.get(key, []) if int(e[0].split("-")[0]) > seq]

    def _read(self, streams: dict[str, str], **kwargs: Any) -> list[Any]:
        rows = [(key, self._after(key, last)) for key, last in streams.items()]
        return [(key, batch) for key, batch in rows if batch]

    def _range(self, key: str, min_id: str, count: int) -> list[Any]:
        return self._after(key, min_id.lstrip("("))[:count]


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def time(self) -> float:
        return self.now


def _streaming_archiver(
    tmp_path: Path,
    statuses: dict[str, str],
    flush_max_entries: int = 1,
    streams: _Streams | None = None,
) -> tuple[TaskLogArchiver, _Streams]:
    streams = streams or _Streams()
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


class _Failing:
    """``prepare_output_dir`` failing with ENOSPC for one task while ``left`` is
    nonzero (negative: always), counting the attempts."""

    def __init__(self, task_dir: Path, left: int) -> None:
        self.task_dir = task_dir
        self.left = left
        self.attempts = 0
        self._prepare = log_archiver.prepare_output_dir

    def __call__(self, base_dir: Path) -> None:
        if base_dir == self.task_dir:
            self.attempts += 1
            if self.left != 0:
                self.left -= 1
                raise OSError(errno.ENOSPC, "No space left on device")
        self._prepare(base_dir)


def _ticks(archiver: TaskLogArchiver, clock: _Clock, count: int, step: float) -> None:
    with patch.object(log_archiver.time, "sleep"):
        for _ in range(count):
            archiver._tick()
            clock.now += step


def test_a_transient_write_error_keeps_every_line_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    failing = _Failing(archiver._base_dir("tsk-1"), 2)
    monkeypatch.setattr(log_archiver, "prepare_output_dir", failing)

    for line in ("one", "two", "three"):
        streams.publish("tsk-1", line)
        _ticks(archiver, clock, 1, 0.5)
    _ticks(archiver, clock, 20, 0.5)

    assert _lines(archiver, "tsk-1") == [
        '{"m": "one"}',
        '{"m": "two"}',
        '{"m": "three"}',
    ]


def test_a_persistent_write_error_backs_off_and_drops_only_after_its_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-a": TaskStatus.DISPATCHED, "tsk-b": TaskStatus.DISPATCHED}
    )
    failing = _Failing(archiver._base_dir("tsk-a"), -1)
    monkeypatch.setattr(log_archiver, "prepare_output_dir", failing)
    window = int(log_archiver._GIVE_UP_SEC)

    streams.publish("tsk-a", "lost")
    for tick in range(window * 10 - 10):
        streams.publish("tsk-b", f"b{tick}")
        _ticks(archiver, clock, 1, 0.1)
        assert len(_lines(archiver, "tsk-b")) == tick + 1
    assert archiver._buffers["tsk-a"]

    _ticks(archiver, clock, 200, 0.1)

    assert not archiver._buffers["tsk-a"]
    assert failing.attempts <= window / archiver._flush_interval_sec + 10


def test_a_failed_write_is_truncated_so_a_retry_writes_each_line_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DISPATCHED}, flush_max_entries=2
    )
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    limit = 1 << 20
    earlier = '{"m": "' + "e" * (limit - 20) + '"}\n'
    (logs / "logs.jsonl").write_text(earlier)
    streams.publish("tsk-1", "alpha")
    streams.publish("tsk-1", "beta")
    # The file may grow only a few bytes: the write stops partway with EFBIG.
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    try:
        _ticks(archiver, clock, 1, 1.0)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, handler)

    _ticks(archiver, clock, 10, 1.0)

    assert _lines(archiver, "tsk-1") == [
        earlier.rstrip("\n"),
        '{"m": "alpha"}',
        '{"m": "beta"}',
    ]


def test_a_restart_while_retrying_still_archives_the_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    write = os.write

    def _full(fd: int, data: Any) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(log_archiver.os, "write", _full)
    streams.publish("tsk-1", "kept")
    _ticks(archiver, clock, 1, 1.0)
    assert (archiver._base_dir("tsk-1") / "logs" / "logs.jsonl").exists()

    monkeypatch.setattr(log_archiver.os, "write", write)
    restarted, _ = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, streams=streams
    )
    _ticks(restarted, clock, 3, 1.0)

    assert _lines(restarted, "tsk-1") == ['{"m": "kept"}']
    assert streams.checkpoints[task_log_archived_key("tsk-1")]


def test_finalizing_records_the_task_archived_apart_from_its_stream_checkpoint(
    tmp_path: Path,
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    streams.publish("tsk-1", "one")
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()
    assert streams.checkpoints[task_log_archive_last_id_key("tsk-1")] == "1-0"

    archiver._runtime.task_statuses.return_value = {  # type: ignore[attr-defined]
        "tsk-1": TaskStatus.DONE
    }
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    # Older code reads the stream checkpoint as a stream id.
    assert task_log_archive_last_id_key("tsk-1") not in streams.checkpoints
    assert streams.checkpoints[task_log_archived_key("tsk-1")]
    restarted, _ = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, streams=streams
    )
    assert restarted._archived("tsk-1")


def test_a_tick_with_only_retrying_tasks_waits_for_the_earliest_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(log_archiver.time, "sleep", _sleep)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DISPATCHED, "tsk-old": TaskStatus.DONE}
    )
    streams.checkpoints[task_log_archived_key("tsk-old")] = "1"
    read = streams.redis.xread_telemetry.side_effect

    def _blocking_read(requested: dict[str, str], **kwargs: Any) -> list[Any]:
        if not (rows := read(requested, **kwargs)):
            clock.now += kwargs["block_ms"] / 1000
        return rows

    streams.redis.xread_telemetry.side_effect = _blocking_read
    failing = _Failing(archiver._base_dir("tsk-1"), -1)
    monkeypatch.setattr(log_archiver, "prepare_output_dir", failing)
    streams.publish("tsk-1", "held")

    window = 60.0
    end = clock.now + window
    ticks = 0
    while clock.now < end and ticks < 10_000:
        archiver._tick()
        ticks += 1

    assert sleeps and all(0 < seconds <= 1.0 for seconds in sleeps)
    assert ticks <= 2 * window
    assert streams.redis.get.call_count <= 2 * ticks + 10
    assert failing.attempts <= window / archiver._flush_interval_sec + 5


def test_a_failed_write_never_cuts_another_writers_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    path = logs / "logs.jsonl"
    path.write_text('{"m": "before"}\n')
    write = os.write
    calls = 0

    def _partial_then_full(fd: int, data: Any) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        with path.open("ab") as other:
            other.write(b'{"m": "theirs"}\n')
        return write(fd, bytes(data[:4]))

    monkeypatch.setattr(log_archiver.os, "write", _partial_then_full)
    streams.publish("tsk-1", "ours")
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert path.read_text().splitlines()[:2] == ['{"m": "before"}', '{"m": "theirs"}']
    assert not archiver._buffers["tsk-1"]


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
