"""A worker reports a failure after its task's external effect may have happened as
ambiguous, and the flag reaches the root on the task's failure event."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.content import (
    OCTET_STREAM,
    ContentReference,
    ContentStoreError,
    SharedFilesystemObjectStore,
)
from shared.schemas.event import TaskEvent, parse_event, serialize_event
from shared.schemas.result import EchoResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    DEFAULT_WORKER_CONFIG,
    FakeContentPlane,
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from tests.worker.test_runner_credential_scrubber import _run
from tests.worker.test_supervisor_client_dispatch_id import _client
from worker.executors.api_executor import APIExecutor
from worker.executors.base_executor import ExecutionError, Executor
from worker.runner import Runner


@pytest.mark.parametrize("ambiguous", [True, False])
def test_the_runner_reports_the_executor_s_ambiguity(
    tmp_path: Path, ambiguous: bool
) -> None:
    error = ExecutionError("read timed out", retryable=True, ambiguous=ambiguous)
    lifecycle = _run(tmp_path, error, [])

    kwargs = lifecycle.set_failed.call_args.kwargs
    assert kwargs["retryable"] is True
    assert kwargs["ambiguous"] is ambiguous


def test_an_uncontrolled_failure_is_not_ambiguous(tmp_path: Path) -> None:
    lifecycle = _run(tmp_path, RuntimeError("boom"), [])

    assert lifecycle.set_failed.call_args.kwargs["ambiguous"] is False


def test_the_failure_event_carries_the_flag_to_the_root() -> None:
    client = _client()
    client.task_failed("tsk-a", "read timed out", ambiguous=True)
    _generation, frame = cast(
        tuple[int, dict[str, Any]], client._event_queue.get_nowait()
    )

    relayed = parse_event(serialize_event(parse_event(frame)))
    assert isinstance(relayed, TaskEvent)
    assert relayed.ambiguous is True


def test_an_older_worker_s_failure_reads_as_not_ambiguous() -> None:
    event = parse_event({"type": "TASK_FAILED", "task_id": "tsk-a", "retryable": True})
    assert isinstance(event, TaskEvent)
    assert event.ambiguous is False


class _UnwritableStore(SharedFilesystemObjectStore):
    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        raise ContentStoreError("store unreachable")


class _Succeeding(Executor):
    name = "echo"

    def __init__(self) -> None:
        self.runs = 0

    def run(self, task: Any, out_dir: Path) -> EchoResult:
        self.runs += 1
        return EchoResult()

    def cancel(self, task_id: str) -> None:
        return None


def _run_with_unwritable_store(
    tmp_path: Path, executor: Executor, spec: dict[str, Any], task_type: TaskType
) -> MagicMock:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    lifecycle.content_plane = FakeContentPlane(_UnwritableStore(tmp_path / "cas"))
    Runner(
        lifecycle=lifecycle,
        task_stream=[
            make_worker_task_message(spec, task_type=task_type, task_id="tsk-1")
        ],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={task_type.value: executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()
    lifecycle.set_succeeded.assert_not_called()
    assert "could not store its result" in lifecycle.set_failed.call_args.args[1]
    return lifecycle


def test_a_failure_after_the_executor_returned_is_ambiguous(tmp_path: Path) -> None:
    executor = _Succeeding()
    lifecycle = _run_with_unwritable_store(
        tmp_path,
        executor,
        {"taskType": "echo", "data": {"mode": "plain"}},
        TaskType.ECHO,
    )

    assert executor.runs == 1
    kwargs = lifecycle.set_failed.call_args.kwargs
    assert kwargs["retryable"] is True
    assert kwargs["ambiguous"] is True


class _Answering(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        cast(_CountingServer, self.server).posts += 1
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


class _CountingServer(ThreadingHTTPServer):
    posts = 0


def test_an_api_call_whose_result_cannot_be_stored_is_sent_once_and_ambiguous(
    tmp_path: Path,
) -> None:
    server = _CountingServer(("127.0.0.1", 0), _Answering)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        lifecycle = _run_with_unwritable_store(
            tmp_path,
            APIExecutor(DEFAULT_WORKER_CONFIG),
            {
                "taskType": "api",
                "api": {
                    "url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "method": "POST",
                    "retries": 3,
                    "response": {"parse_json": False},
                },
            },
            TaskType.API,
        )
    finally:
        APIExecutor.close_all_clients()
        server.shutdown()
        server.server_close()

    assert server.posts == 1
    assert lifecycle.set_failed.call_args.kwargs["ambiguous"] is True
