"""A worker's report of an upload a success left owed reaches the runtime whatever its
task's state, fenced there to the dispatch owing it."""

from typing import Any, cast
from unittest.mock import MagicMock

from shared.schemas.event import TaskEvent
from shared.tasks.artifact_upload import ARTIFACT_UPLOAD_EVENT, UploadReport
from tests.server.task.test_task_merge import _monitor


def _report(payload: dict[str, Any]) -> TaskEvent:
    return TaskEvent(
        type=ARTIFACT_UPLOAD_EVENT,
        task_id="tsk-1",
        worker_id="wkr-1",
        dispatch_id="dsp-1",
        payload=payload,
    )


def test_an_upload_report_reaches_the_runtime_past_the_running_task_fence() -> None:
    runtime = MagicMock()
    runtime.holds_dispatch.return_value = False

    _monitor(cast(Any, runtime)).handle_task_event(
        _report({"report": "unuploadable", "error": "custody changed"})
    )

    runtime.record_artifact_upload.assert_called_once_with(
        "tsk-1", "wkr-1", "dsp-1", UploadReport.UNUPLOADABLE, "custody changed"
    )


def test_an_unreadable_upload_report_is_ignored() -> None:
    runtime = MagicMock()

    _monitor(cast(Any, runtime)).handle_task_event(_report({"report": "done"}))

    runtime.record_artifact_upload.assert_not_called()
