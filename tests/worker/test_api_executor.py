"""Tests for the API executor url override and Nebula credential handling."""

import email.utils
import json
import logging
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from shared.schemas.result import APIResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.api_executor import (
    _MAX_RETRIES,
    _RETRY_BACKOFF_MAX_SEC,
    APIExecutor,
)
from worker.executors.base_executor import ExecutionError, TaskCancelledError


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api",
        "workflow_id": "wf-1",
        "owner_id": "owner",
        "assigned_worker": "worker-1",
        "dispatched_at": "2026-03-22T00:00:00Z",
        "task": {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "wf:api"},
            "spec": {
                "taskType": "api",
                "api": {
                    "method": "POST",
                    "body": {"messages": [{"role": "user", "content": "hi"}]},
                    **spec_updates,
                },
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


class _RecordingTransport(httpx.MockTransport):
    """MockTransport that records the request it served."""

    def __init__(self) -> None:
        self.request: httpx.Request | None = None
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"total_tokens": 3},
            },
        )


def _run(
    executor: APIExecutor, task: WorkerTaskMessage, transport: httpx.MockTransport
) -> APIResult:
    with patch.object(
        APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
    ):
        return executor.run(task, Path(tempfile.gettempdir()))


class TestNebulaPath:
    def test_no_url_no_header_uses_nebula_url_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer nebula-token"

    def test_no_url_with_header_preserves_header_and_skips_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers={"Authorization": "Bearer custom"})
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    def test_no_url_with_x_api_key_skips_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers={"X-API-Key": "custom-key"})
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert "Authorization" not in transport.request.headers
        assert transport.request.headers["X-API-Key"] == "custom-key"

    def test_no_url_with_content_type_only_gets_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers={"Content-Type": "application/json"})
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer nebula-token"

    def test_token_request_drops_author_routing_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            headers={
                "Host": "attacker.example.com",
                "forwarded": "host=attacker.example.com",
                "X-Forwarded-Host": "attacker.example.com",
                "x-forwarded-for": "10.0.0.1",
                "X-Host": "attacker.example.com",
                "X-Original-Host": "attacker.example.com",
                "X-Original-URL": "/steal",
                "x-rewrite-url": "/steal",
                "X-Trace": "kept",
            }
        )
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        sent = transport.request.headers
        assert sent["Host"] == "nebula.example.com"
        assert "Forwarded" not in sent
        assert "X-Forwarded-Host" not in sent
        assert "X-Forwarded-For" not in sent
        for name in ("X-Host", "X-Original-Host", "X-Original-URL", "X-Rewrite-URL"):
            assert name not in sent
        assert sent["X-Trace"] == "kept"
        assert sent["Authorization"] == "Bearer nebula-token"

    def test_own_credential_request_keeps_author_routing_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            headers={"Authorization": "Bearer custom", "X-Forwarded-For": "10.0.0.1"}
        )
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.headers["X-Forwarded-For"] == "10.0.0.1"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    @pytest.mark.parametrize(
        ("headers", "verified"),
        [({}, True), ({"Authorization": "Bearer custom"}, False)],
    )
    def test_the_deployment_token_is_sent_only_over_verified_tls(
        self, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], verified: bool
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers=headers, verify_tls=False)
        transport = _RecordingTransport()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ) as get_client:
            APIExecutor(DEFAULT_WORKER_CONFIG).run(task, Path(tempfile.gettempdir()))
        assert get_client.call_args.args[2] is verified

    def test_no_url_no_header_without_token_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.delenv("NEBULA_API_TOKEN", raising=False)
        transport = _RecordingTransport()
        with pytest.raises(ExecutionError, match="no credential configured"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), _task_message(), transport)
        assert transport.request is None

    def test_neither_url_nor_base_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_BASE_URL", raising=False)
        task = _task_message()
        with pytest.raises(ExecutionError, match="spec.api.url or NEBULA_API_BASE_URL"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, _RecordingTransport())


class TestCustomUrl:
    def test_custom_url_without_credential_sends_no_nebula_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The token is set, so its absence proves it is withheld, not unavailable.
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(url="https://custom.example.com/v1/chat/completions")
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert "Authorization" not in transport.request.headers

    def test_custom_url_with_header_preserves_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer custom"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    def test_custom_url_with_x_api_key_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"X-API-Key": "custom-key"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert transport.request.headers["X-API-Key"] == "custom-key"

    def test_custom_url_with_only_innocent_header_stays_unauthenticated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"Content-Type": "application/json"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Content-Type"] == "application/json"
        assert "Authorization" not in transport.request.headers


class _CookieServer(ThreadingHTTPServer):
    cookies_seen: list[str | None]


class _CookieHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = cast(_CookieServer, self.server)
        server.cookies_seen.append(self.headers.get("Cookie"))
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}
        ).encode()
        self.send_response(200)
        self.send_header("Set-Cookie", "session=tenant-a; Path=/")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


def test_a_pooled_client_carries_no_cookie_between_tasks() -> None:
    server = _CookieServer(("127.0.0.1", 0), _CookieHandler)
    server.cookies_seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        for _ in range(2):
            APIExecutor(DEFAULT_WORKER_CONFIG).run(
                _task_message(url=url), Path(tempfile.gettempdir())
            )
    finally:
        APIExecutor.close_all_clients()
        server.shutdown()
        server.server_close()
    assert server.cookies_seen == [None, None]


def test_the_stored_url_carries_no_credential_the_request_sent() -> None:
    task = _task_message(
        url="https://user:pw@api.example/v1/chat",
        params={"api_key": "query-secret", "limit": 3},
        headers={"Authorization": "Bearer custom"},
    )
    transport = _RecordingTransport()
    result = _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)

    assert transport.request is not None
    assert "query-secret" in str(transport.request.url)
    assert result.url == "https://api.example/v1/chat?api_key=[REDACTED]&limit=3"


def test_a_failed_request_quotes_no_url_credential() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    task = _task_message(
        url="https://user:pw-inline-SECRET@api.example/v1?api_key=q-inline-SECRET"
    )
    with patch.object(
        APIExecutor,
        "_get_client",
        return_value=httpx.Client(transport=httpx.MockTransport(refuse)),
    ):
        with pytest.raises(ExecutionError) as raised:
            APIExecutor(DEFAULT_WORKER_CONFIG).run(task, Path(tempfile.gettempdir()))
    assert "inline-SECRET" not in str(raised.value)
    assert "api.example" in str(raised.value)


def test_a_retry_warning_quotes_no_url_credential(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    task = _task_message(
        url="https://user:pw-inline-SECRET@api.example/v1?api_key=q-inline-SECRET",
        retries=1,
    )
    with (
        patch.object(APIExecutor, "_wait_for_backoff"),
        patch.object(
            APIExecutor,
            "_get_client",
            return_value=httpx.Client(transport=httpx.MockTransport(refuse)),
        ),
        caplog.at_level(logging.WARNING, logger="worker.executors.api_executor"),
        pytest.raises(ExecutionError),
    ):
        APIExecutor(DEFAULT_WORKER_CONFIG).run(task, Path(tempfile.gettempdir()))
    assert "attempt 1/1" in caplog.text
    assert "api.example" in caplog.text
    assert "inline-SECRET" not in caplog.text


class _BlockingTransport(httpx.MockTransport):
    """MockTransport that blocks on the first request, then serves a sequence."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            if not self.release.wait(5.0):
                raise AssertionError("blocking transport was not released")
        return self.responses.pop(0)


class _NotifyTransport(httpx.MockTransport):
    """MockTransport that sets an event once it has served a request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.served = threading.Event()
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.served.set()
        return self.responses.pop(0)


class _SequenceTransport(httpx.MockTransport):
    """MockTransport that serves a fixed sequence of responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses.pop(0)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 3},
        },
    )


def _error_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, json={"error": "boom"})


class TestRetries:
    def _task(self, **spec_updates: object) -> WorkerTaskMessage:
        return _task_message(
            url="https://custom.example.com/v1/chat/completions",
            response={"parse_json": False},
            **spec_updates,
        )

    def test_retry_succeeds_after_transient_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 504 followed by a 200 succeeds when retries are configured."""
        monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _ok_response()]
        )
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.calls == 3

    def test_retries_exhausted_still_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistent 5xx failures exhaust retries and raise loudly."""
        monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _error_response(504)]
        )
        with pytest.raises(ExecutionError, match="status 504"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.calls == 3

    def test_no_retry_by_default(self) -> None:
        """Without a retries field, a transient failure fails immediately."""
        task = self._task()
        transport = _SequenceTransport([_error_response(504)])
        with pytest.raises(ExecutionError, match="status 504"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.calls == 1

    def test_non_retryable_status_not_retried(self) -> None:
        """A 4xx (other than 408/429) is never retried."""
        task = self._task(retries=3)
        transport = _SequenceTransport([_error_response(400)])
        with pytest.raises(ExecutionError, match="status 400"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
        assert transport.calls == 1

    def test_cancelled_task_stops_retrying(self) -> None:
        """A cancelled task does not keep retrying."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)
        task = self._task(retries=3)
        executor.cancel(task.task_id)
        transport = _SequenceTransport([_error_response(504)])
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task, Path(tempfile.gettempdir()))
        assert transport.calls == 0

    def test_invalid_retries_rejected(self) -> None:
        """A negative or non-integer retries value is rejected."""
        for bad in (-1, "2", 1.5, True):
            task = self._task(retries=bad)
            with pytest.raises(ExecutionError, match="spec.api.retries"):
                _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, _RecordingTransport())

    def test_cancel_previous_task_does_not_cancel_next(self) -> None:
        """A cancellation left over from a prior task does not cancel the next."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)

        task_a = self._task(retries=0)
        task_a.task_id = "task-a"
        executor.cancel(task_a.task_id)

        task_b = self._task(retries=0)
        task_b.task_id = "task-b"
        transport = _RecordingTransport()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            executor.run(task_b, Path(tempfile.gettempdir()))
        assert transport.request is not None

    def test_late_cancel_of_previous_task_does_not_overwrite_next(self) -> None:
        """A late cancel for A cannot overwrite a recorded cancel for B."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)

        task_b = self._task(retries=0)
        task_b.task_id = "task-b"
        executor.cancel(task_b.task_id)
        executor.cancel("task-a")

        transport = _RecordingTransport()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task_b, Path(tempfile.gettempdir()))
        assert transport.request is None

    def test_delayed_cancel_of_previous_task_does_not_cancel_next(self) -> None:
        """A late cancellation for a prior task does not cancel a running task."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)

        task_b = self._task(retries=1)
        task_b.task_id = "task-b"
        # First request blocks; once released it returns a retryable 503 so the
        # loop re-checks the cancel event, then a 200 succeeds.
        transport = _BlockingTransport([_error_response(503), _ok_response()])

        errors: list[BaseException] = []

        def _run_b() -> None:
            try:
                with patch.object(
                    APIExecutor,
                    "_get_client",
                    return_value=httpx.Client(transport=transport),
                ):
                    executor.run(task_b, Path(tempfile.gettempdir()))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_run_b)
        thread.start()
        try:
            assert transport.started.wait(2.0)
            executor.cancel("task-a")
        finally:
            transport.release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert errors == []
        assert transport.calls == 2

    def test_cancel_of_active_task_still_cancels(self) -> None:
        """A cancellation addressed to the running task still cancels it."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)

        task_b = self._task(retries=3)
        task_b.task_id = "task-b"
        transport = _BlockingTransport([_error_response(503)])

        errors: list[BaseException] = []

        def _run_b() -> None:
            try:
                with patch.object(
                    APIExecutor,
                    "_get_client",
                    return_value=httpx.Client(transport=transport),
                ):
                    executor.run(task_b, Path(tempfile.gettempdir()))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_run_b)
        thread.start()
        try:
            assert transport.started.wait(2.0)
            executor.cancel("task-b")
        finally:
            transport.release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)

    def test_cancel_during_backoff_stops_retrying(self) -> None:
        """A cancellation during the retry backoff aborts well before it ends."""
        executor = APIExecutor(DEFAULT_WORKER_CONFIG)

        task = self._task(retries=3)
        task.task_id = "task-b"
        transport = _NotifyTransport([_error_response(503)])

        def _cancel_on_served() -> None:
            transport.served.wait(2.0)
            executor.cancel(task.task_id)

        canceller = threading.Thread(target=_cancel_on_served)
        canceller.start()
        start = time.monotonic()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task, Path(tempfile.gettempdir()))
        elapsed = time.monotonic() - start
        canceller.join()
        assert elapsed < 0.5
        assert transport.calls == 1

    def _run_recording_delays(
        self, task: WorkerTaskMessage, transport: httpx.MockTransport
    ) -> list[float]:
        """Run a task, recording each backoff delay instead of waiting."""
        delays: list[float] = []

        def _record(delay: float) -> None:
            delays.append(delay)

        executor = APIExecutor(DEFAULT_WORKER_CONFIG)
        with patch.object(APIExecutor, "_wait_for_backoff", side_effect=_record):
            with patch.object(
                APIExecutor,
                "_get_client",
                return_value=httpx.Client(transport=transport),
            ):
                executor.run(task, Path(tempfile.gettempdir()))
        return delays

    def test_retry_after_seconds_is_honoured(self) -> None:
        """A Retry-After in seconds sets the wait for the next attempt."""
        task = self._task(retries=1)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "5"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [5.0]

    def test_retry_after_date_is_honoured(self) -> None:
        """A Retry-After HTTP date sets the wait for the next attempt."""
        task = self._task(retries=1)
        retry_at = datetime.now(UTC) + timedelta(seconds=5)
        transport = _SequenceTransport(
            [
                httpx.Response(
                    429, headers={"Retry-After": email.utils.format_datetime(retry_at)}
                ),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [pytest.approx(5.0, abs=1.0)]

    def test_invalid_retry_after_falls_back_to_exponential(self) -> None:
        """A broken Retry-After falls back to the exponential schedule."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "nan"}),
                httpx.Response(429, headers={"Retry-After": "not-a-date"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [1.0, 2.0]

    def test_exponential_backoff_schedule(self) -> None:
        """Retries back off exponentially from the base delay."""
        task = self._task(retries=3)
        transport = _SequenceTransport(
            [
                _error_response(503),
                _error_response(503),
                _error_response(503),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [1.0, 2.0, 4.0]

    def test_retry_after_is_capped(self) -> None:
        """A hostile Retry-After cannot stall a worker past the cap."""
        task = self._task(retries=1)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "999999"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [_RETRY_BACKOFF_MAX_SEC]

    def test_retries_above_maximum_rejected(self) -> None:
        """A retries value above the maximum is rejected."""
        task = self._task(retries=_MAX_RETRIES + 1)
        with pytest.raises(ExecutionError, match=f"at most {_MAX_RETRIES}"):
            _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, _RecordingTransport())

    def test_one_warning_per_retry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each retry logs one warning naming the attempt and the delay."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _ok_response()]
        )
        with caplog.at_level(logging.WARNING, logger="worker.executors.api_executor"):
            self._run_recording_delays(task, transport)
        warnings = [
            r for r in caplog.records if r.name == "worker.executors.api_executor"
        ]
        assert len(warnings) == 2
        assert "attempt 1/2" in warnings[0].getMessage()
        assert "attempt 2/2" in warnings[1].getMessage()


@pytest.mark.parametrize(
    ("response", "retryable"),
    [
        (httpx.Response(401, text="Unauthorized"), False),
        (httpx.Response(400, json={"error": "bad model"}), False),
        (httpx.Response(502, text="<html>Bad Gateway</html>"), True),
    ],
)
def test_an_error_status_fails_as_its_status_whatever_its_body(
    response: httpx.Response, retryable: bool
) -> None:
    task = _task_message(url="https://api.example/v1/chat/completions")
    with pytest.raises(
        ExecutionError, match=f"status {response.status_code}"
    ) as raised:
        _run(
            APIExecutor(DEFAULT_WORKER_CONFIG),
            task,
            _SequenceTransport([response]),
        )
    assert raised.value.retryable is retryable


def test_a_body_that_is_not_json_fails_without_a_retry() -> None:
    task = _task_message(url="https://api.example/v1/chat/completions")
    with pytest.raises(ExecutionError, match="not a valid JSON") as raised:
        _run(
            APIExecutor(DEFAULT_WORKER_CONFIG),
            task,
            _SequenceTransport([httpx.Response(200, text="plain text")]),
        )
    assert raised.value.retryable is False


class _RaisingTransport(httpx.MockTransport):
    """Raise ``error`` on the first request and answer 200 after it."""

    def __init__(self, error: type[httpx.RequestError]) -> None:
        self.calls = 0
        self._error = error
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            raise self._error("failed", request=request)
        return _ok_response()


@pytest.mark.parametrize(
    "error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout]
)
def test_a_request_that_never_left_is_retried(
    monkeypatch: pytest.MonkeyPatch, error: type[httpx.TransportError]
) -> None:
    monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
    task = _task_message(
        url="https://api.example/v1", retries=1, response={"parse_json": False}
    )
    transport = _RaisingTransport(error)
    _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
    assert transport.calls == 2


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.LocalProtocolError,
        httpx.DecodingError,
        httpx.TooManyRedirects,
        httpx.UnsupportedProtocol,
    ],
)
def test_a_request_that_may_have_left_is_sent_once(
    monkeypatch: pytest.MonkeyPatch, error: type[httpx.RequestError]
) -> None:
    monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
    task = _task_message(url="https://api.example/v1", retries=3)
    transport = _RaisingTransport(error)
    with pytest.raises(ExecutionError, match="API request failed"):
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
    assert transport.calls == 1


def test_an_unrepresentable_retry_after_date_backs_off_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
    task = _task_message(
        url="https://api.example/v1", retries=1, response={"parse_json": False}
    )
    far = httpx.Response(
        503, headers={"Retry-After": "Fri, 31 Dec 99999999999999999999 23:59:59 GMT"}
    )
    transport = _SequenceTransport([far, _ok_response()])
    _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, transport)
    assert transport.calls == 2


def test_a_body_nested_past_the_parser_fails_as_invalid_json() -> None:
    task = _task_message(url="https://api.example/v1/chat/completions")
    deep = httpx.Response(
        200, content=b"[" * 200_000, headers={"content-type": "application/json"}
    )
    with pytest.raises(ExecutionError, match="not a valid JSON") as raised:
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, _SequenceTransport([deep]))
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    ("error", "ambiguous"),
    [
        (httpx.ConnectError, False),
        (httpx.ConnectTimeout, False),
        (httpx.PoolTimeout, False),
        (httpx.UnsupportedProtocol, False),
        (httpx.TooManyRedirects, True),
        (httpx.DecodingError, True),
        (httpx.LocalProtocolError, True),
        (httpx.ReadTimeout, True),
        (httpx.WriteTimeout, True),
        (httpx.ReadError, True),
        (httpx.WriteError, True),
        (httpx.RemoteProtocolError, True),
    ],
)
def test_a_failure_after_the_request_may_have_left_is_ambiguous(
    error: type[httpx.RequestError], ambiguous: bool
) -> None:
    task = _task_message(url="https://api.example/v1")
    with pytest.raises(ExecutionError) as raised:
        _run(APIExecutor(DEFAULT_WORKER_CONFIG), task, _RaisingTransport(error))
    assert raised.value.retryable is True
    assert raised.value.ambiguous is ambiguous


class _RedirectingHandler(BaseHTTPRequestHandler):
    """Record each request, and answer by path: redirect away, loop, or misencode."""

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        server = cast(_RedirectingServer, self.server)
        server.requests.append((self.command, self.path))
        if self.path == "/unreachable":
            self._redirect(303, "http://127.0.0.1:1/gone")
        elif self.path == "/ftp":
            self._redirect(303, "ftp://127.0.0.1/file")
        elif self.path == "/loop":
            self._redirect(307, "/loop")
        elif self.path == "/moved":
            self._redirect(303, "/done")
        elif self.path == "/gzip":
            self._reply(b"not gzip", {"Content-Encoding": "gzip"})
        else:
            self._reply(json.dumps(_ok_response().json()).encode(), {})

    do_GET = do_POST = _serve

    def _redirect(self, status: int, location: str) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _reply(self, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


class _RedirectingServer(ThreadingHTTPServer):
    requests: list[tuple[str, str]]


@pytest.fixture
def redirecting_server() -> Any:
    server = _RedirectingServer(("127.0.0.1", 0), _RedirectingHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    APIExecutor.close_all_clients()
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize(
    ("path", "posts"),
    # A 307 loop re-posts on every hop until the client gives up, once.
    [("/unreachable", 1), ("/ftp", 1), ("/loop", 21), ("/gzip", 1)],
)
def test_a_failure_after_the_server_received_the_request_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
    redirecting_server: _RedirectingServer,
    path: str,
    posts: int,
) -> None:
    monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
    url = f"http://127.0.0.1:{redirecting_server.server_address[1]}{path}"
    with pytest.raises(ExecutionError) as raised:
        APIExecutor(DEFAULT_WORKER_CONFIG).run(
            _task_message(url=url, retries=3), Path(tempfile.gettempdir())
        )
    assert raised.value.ambiguous is True
    assert raised.value.retryable is True
    assert redirecting_server.requests == [("POST", path)] * posts


def test_a_redirect_is_still_followed_to_its_answer(
    redirecting_server: _RedirectingServer,
) -> None:
    url = f"http://127.0.0.1:{redirecting_server.server_address[1]}/moved"
    result = APIExecutor(DEFAULT_WORKER_CONFIG).run(
        _task_message(url=url), Path(tempfile.gettempdir())
    )
    assert result.text == "hello"
    assert result.url.endswith("/done")
    assert redirecting_server.requests == [("POST", "/moved"), ("GET", "/done")]
