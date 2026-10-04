"""A worker reports a failure after its task's external effect may have happened as
ambiguous, and the flag reaches the root on the task's failure event."""

from pathlib import Path
from typing import Any, cast

import pytest

from shared.schemas.event import TaskEvent, parse_event, serialize_event
from tests.worker.test_runner_credential_scrubber import _run
from tests.worker.test_supervisor_client_dispatch_id import _client
from worker.executors.base_executor import ExecutionError


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
