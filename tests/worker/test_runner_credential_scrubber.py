"""A worker reports and logs nothing of the credentials its dispatch restored."""

import logging
from pathlib import Path
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock

import pytest

from shared.content import SharedFilesystemObjectStore
from shared.harness.adapter import HarnessResult, HarnessResultKind
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    FakeContentPlane,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors.base_executor import ExecutionError, Executor
from worker.executors.episode_support import EpisodeStepResult
from worker.runner import Runner
from worker.utils.logging import TaskLogEmitter

_SECRET = "tok-restored-SECRET"


class _Failing(Executor):
    name = "echo"

    def __init__(self, error: Exception | EpisodeStepResult) -> None:
        self.error = error

    def run(self, task: Any, out_dir: Path) -> Any:
        if isinstance(self.error, EpisodeStepResult):
            return self.error
        raise self.error

    def cancel(self, task_id: str) -> None:
        return None


def _run(
    tmp_path: Path, error: Exception | EpisodeStepResult, pointers: list[str]
) -> MagicMock:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.content_plane = FakeContentPlane(
        SharedFilesystemObjectStore(tmp_path / "cas")
    )
    msg = make_worker_task_message(
        {"taskType": "echo", "data": {"token": _SECRET, "mode": "plain"}},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        credential_pointers={"tsk-1": pointers},
    )
    executor = _Failing(error)
    logger = MagicMock()
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=logger,
    ).start()
    lifecycle.logged = " ".join(str(call) for call in logger.mock_calls)
    return lifecycle


@pytest.mark.parametrize(
    "error",
    [RuntimeError(f"auth failed for {_SECRET}"), ExecutionError(f"bad {_SECRET!r}")],
)
def test_a_failure_reports_and_logs_no_restored_credential(tmp_path, error):
    lifecycle = _run(tmp_path, error, ["/data/token"])

    reported = lifecycle.set_failed.call_args.args[1]
    assert _SECRET not in reported and "[REDACTED]" in reported
    assert _SECRET not in lifecycle.logged


def test_an_agent_step_failure_reports_no_restored_credential(tmp_path):
    step = EpisodeStepResult(
        harness_result=HarnessResult(
            kind=HarnessResultKind.FAILURE, error=f"harness rejected {_SECRET}"
        )
    )
    lifecycle = _run(tmp_path, step, ["/data/token"])

    reported = lifecycle.set_succeeded.call_args.kwargs["metadata"]["agent_episode"]
    assert reported["error"] == "harness rejected [REDACTED]"


def test_a_dispatch_naming_no_credentials_reports_its_error_as_is(tmp_path):
    lifecycle = _run(tmp_path, RuntimeError(f"auth failed for {_SECRET}"), [])

    assert _SECRET in lifecycle.set_failed.call_args.args[1]


def test_the_task_log_emitter_scrubs_every_line() -> None:
    with mock.patch("worker.utils.logging._GrpcLogStream"):
        emitter = TaskLogEmitter(
            stub=mock.Mock(),
            metadata=(),
            struct_from_payload=lambda payload: payload,
            logger=logging.getLogger("test_scrubbed_emitter"),
            task_id="tsk-1",
            workflow_id="wfl-1",
            owner_id="own-1",
            worker_id="wrk-1",
            scrub=lambda text: text.replace(_SECRET, "[REDACTED]"),
        )
    sent: list[dict[str, Any]] = []
    emitter._stream = cast(Any, mock.Mock(send=sent.append))

    emitter.emit(
        logging.LogRecord(
            "task", logging.INFO, __file__, 1, "using %s", (_SECRET,), None
        )
    )

    assert sent[0]["message"] == "using [REDACTED]"
