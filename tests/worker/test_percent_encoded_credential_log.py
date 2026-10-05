"""A vaulted credential an HTTP client logs inside a request URL, percent-encoded,
reaches the task log masked."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

import httpx

from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.api_executor import APIExecutor
from worker.runner import _task_scrubber
from worker.utils.logging import TaskLogEmitter

# A base64 value: '+', '/' and '=' are percent-encoded in a query string.
_VAULTED = "+".join(("abc", "def/ghi", "jkl/mno=="))


def test_an_api_key_in_the_request_url_reaches_the_task_log_masked() -> None:
    msg = make_worker_task_message(
        {
            "taskType": "api",
            "api": {
                "url": "https://api.example/v1",
                "method": "GET",
                "params": {"key": _VAULTED},
                "response": {"parse_json": False},
            },
        },
        task_type=TaskType.API,
        task_id="tsk-1",
        credential_pointers={"tsk-1": ["/api/params/key"]},
    )
    with mock.patch("worker.utils.logging._GrpcLogStream"):
        emitter = TaskLogEmitter(
            stub=mock.Mock(),
            metadata=(),
            struct_from_payload=lambda payload: payload,
            logger=logging.getLogger("test_percent_encoded"),
            task_id="tsk-1",
            workflow_id="wfl-1",
            owner_id="own-1",
            worker_id="wrk-1",
            scrub=_task_scrubber(msg),
        )
    sent: list[dict[str, Any]] = []
    emitter._stream = cast(Any, mock.Mock(send=sent.append))
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addHandler(emitter)
    previous = httpx_logger.level
    httpx_logger.setLevel(logging.INFO)
    try:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        )
        with mock.patch.object(APIExecutor, "_get_client", return_value=client):
            APIExecutor(DEFAULT_WORKER_CONFIG).run(msg, Path(tempfile.gettempdir()))
    finally:
        httpx_logger.removeHandler(emitter)
        httpx_logger.setLevel(previous)

    (line,) = [p["message"] for p in sent if p["message"].startswith("HTTP Request")]
    assert "key=[REDACTED]" in line
    assert "abc%2Bdef" not in line
