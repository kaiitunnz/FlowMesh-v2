"""A task's ``logs/logs.jsonl`` has one writer, the server's log archiver, even where
the worker's results directory is the server's own; the worker keeps its own copy of
the task's logs beside it in ``logs/worker.jsonl``."""

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from server.clients.redis import task_log_stream_key
from server.services import log_archiver
from server.task.models import TaskStatus
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.server.services.test_log_archiver import _streaming_archiver, _Streams
from tests.worker.factories import (
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from worker.executors.base_executor import Executor
from worker.runner import Runner
from worker.utils.logging import TaskLogEmitter

_LOGGER = logging.getLogger("tests.task_code")


class _LoggingExecutor(Executor):
    name = "echo"

    def __init__(self) -> None:
        pass

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        _LOGGER.warning("hello from the task")
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None


class _RecordingStream:
    def __init__(self, sent: list[dict[str, Any]]) -> None:
        self._sent = sent

    def send(self, payload: dict[str, Any]) -> None:
        self._sent.append(payload)

    def close(self) -> None:
        return None


def _run_task(results_dir: Path, task_id: str) -> list[dict[str, Any]]:
    """Run one task whose code logs a line; return the payloads its log emitter
    streamed to the server."""
    sent: list[dict[str, Any]] = []

    def _emitter(**kwargs: Any) -> TaskLogEmitter:
        emitter = TaskLogEmitter(
            stub=MagicMock(),
            metadata=(),
            struct_from_payload=lambda payload: payload,
            logger=logging.getLogger("tests.emitter"),
            worker_id="wrk-test",
            **kwargs,
        )
        emitter._stream.close()
        emitter._stream = _RecordingStream(sent)  # type: ignore[assignment]
        return emitter

    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.side_effect = _emitter
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    executor = _LoggingExecutor()
    msg = make_worker_task_message(
        {"taskType": "echo"}, task_type=TaskType.ECHO, task_id=task_id
    )
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=results_dir,
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=logging.getLogger("tests.runner"),
    ).start()
    assert any(p.get("message") == "hello from the task" for p in sent)
    assert not any(isinstance(h, TaskLogEmitter) for h in logging.getLogger().handlers)
    return sent


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_a_worker_keeps_its_own_copy_apart_from_the_archive(tmp_path: Path) -> None:
    sent = _run_task(tmp_path, "tsk-1")

    logs = tmp_path / "tsk-1" / "logs"
    assert _lines(logs / "worker.jsonl") == [
        json.dumps(payload, ensure_ascii=False) for payload in sent
    ]
    assert not (logs / "logs.jsonl").exists()


def test_a_task_on_the_servers_results_volume_archives_each_line_once(
    tmp_path: Path,
) -> None:
    # The worker shares the server's results directory, as on the root node, and
    # the task finishes before the archiver ever flushes it.
    sent = _run_task(tmp_path, "tsk-1")
    closed = {"type": "LOG_STREAM_CLOSED", "message": "Task log stream closed."}
    streams = _Streams()
    key = task_log_stream_key("tsk-1")
    payloads = [json.dumps(p, ensure_ascii=False) for p in [*sent, closed]]
    streams._log[key] = [
        (f"{seq}-0", {"payload": payload})
        for seq, payload in enumerate(payloads, start=1)
    ]
    archiver, _ = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, flush_max_entries=100, streams=streams
    )

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    logs = tmp_path / "tsk-1" / "logs"
    assert _lines(logs / "logs.jsonl") == payloads
    assert _lines(logs / "worker.jsonl") == payloads[:-1]
