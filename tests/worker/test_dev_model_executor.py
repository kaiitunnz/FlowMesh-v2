"""Tests for DevModelExecutor."""

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.schemas.result import DevModelResult
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.dev_model import DevModelSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors import dev_model_executor as mod
from worker.executors.base_executor import TaskCancelledError
from worker.executors.dev_model_executor import (
    _CANNED_TEXT,
    DevModelExecutor,
    _DevModelHandler,
    _DevModelHTTPServer,
)


@contextmanager
def _running_server(
    forward_url: str | None = None,
    model_name: str = "test-model",
    client: httpx.Client | None = None,
) -> Iterator[str]:
    server = _DevModelHTTPServer(
        ("127.0.0.1", 0), _DevModelHandler, forward_url, model_name, client
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


class TestDevModelExecutorInit:
    def test_only_dev_model_task_type(self) -> None:
        assert DevModelExecutor.supported_task_types == frozenset({TaskType.DEV_MODEL})

    def test_is_available_false_when_gate_off(self) -> None:
        assert DevModelExecutor.is_available(make_worker_config()) is False

    def test_is_available_true_when_gate_on(self) -> None:
        cfg = make_worker_config(enable_dev_model=True)
        assert DevModelExecutor.is_available(cfg) is True


class TestDevModelSpec:
    def test_minimal_spec(self) -> None:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL)
        assert spec.model is None
        assert spec.model_name is None
        assert spec.ttlSeconds is None
        assert spec.accessMode is None
        assert spec.port is None

    def test_spec_with_all_fields(self) -> None:
        spec = DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            model=ModelConfig(source=ModelSource(identifier="dev/model")),
            ttlSeconds=60.0,
            accessMode="forward",
            port=8123,
        )
        assert spec.model_name == "dev/model"
        assert spec.ttlSeconds == 60.0
        assert spec.accessMode == "forward"
        assert spec.port == 8123

    def test_invalid_access_mode(self) -> None:
        with pytest.raises(Exception):
            DevModelSpecStrict(taskType=TaskType.DEV_MODEL, accessMode="invalid")  # type: ignore[arg-type]

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            DevModelSpecStrict(taskType=TaskType.DEV_MODEL, ttlSeconds=0.0)

    def test_port_must_be_in_range(self) -> None:
        with pytest.raises(Exception):
            DevModelSpecStrict(taskType=TaskType.DEV_MODEL, port=0)
        with pytest.raises(Exception):
            DevModelSpecStrict(taskType=TaskType.DEV_MODEL, port=65536)


class TestCannedResponses:
    def test_chat_completions_is_deterministic(self) -> None:
        with _running_server() as base:
            first = httpx.post(
                f"{base}/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=5.0,
            ).json()
            second = httpx.post(
                f"{base}/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=5.0,
            ).json()
        assert first == second
        assert first["object"] == "chat.completion"
        assert first["choices"][0]["message"]["content"] == _CANNED_TEXT
        assert first["model"] == "m"

    def test_responses_is_deterministic(self) -> None:
        with _running_server() as base:
            payload = httpx.post(
                f"{base}/v1/responses",
                json={"model": "m", "input": "hi"},
                timeout=5.0,
            ).json()
        assert payload["object"] == "response"
        assert payload["status"] == "completed"
        assert payload["output_text"] == _CANNED_TEXT
        assert payload["output"][0]["content"][0]["text"] == _CANNED_TEXT

    def test_model_falls_back_when_absent(self) -> None:
        with _running_server(model_name="fallback-model") as base:
            payload = httpx.post(
                f"{base}/v1/chat/completions", json={"messages": []}, timeout=5.0
            ).json()
        assert payload["model"] == "fallback-model"

    def test_unknown_route_returns_404(self) -> None:
        with _running_server() as base:
            resp = httpx.post(f"{base}/v1/embeddings", json={}, timeout=5.0)
        assert resp.status_code == 404


class _UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        out = json.dumps(
            {
                "upstream": True,
                "path": self.path,
                "echo_model": body.get("model"),
                "echo_auth": self.headers.get("Authorization"),
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class _AuthUpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        authorized = self.headers.get("Authorization") == "Bearer sk-test"
        out = b'{"ok": true}' if authorized else b'{"error": "unauthorized"}'
        self.send_response(200 if authorized else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@contextmanager
def _upstream_server(
    handler: type[BaseHTTPRequestHandler] = _UpstreamHandler,
) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


class TestForwardMode:
    def test_forwards_request_to_upstream(self) -> None:
        with _upstream_server() as upstream, httpx.Client() as client:
            with _running_server(forward_url=upstream, client=client) as base:
                resp = httpx.post(
                    f"{base}/v1/chat/completions",
                    json={"model": "up-model", "messages": []},
                    timeout=5.0,
                )
        assert resp.json()["upstream"] is True
        assert resp.json()["path"] == "/v1/chat/completions"
        assert resp.json()["echo_model"] == "up-model"
        assert resp.headers["Content-Type"] == "application/json; charset=utf-8"

    def test_forwards_authorization_header(self) -> None:
        with (
            _upstream_server(_AuthUpstreamHandler) as upstream,
            httpx.Client() as client,
        ):
            with _running_server(forward_url=upstream, client=client) as base:
                without = httpx.post(f"{base}/v1/responses", json={}, timeout=5.0)
                withauth = httpx.post(
                    f"{base}/v1/responses",
                    json={},
                    headers={"Authorization": "Bearer sk-test"},
                    timeout=5.0,
                )
        assert without.status_code == 401
        assert withauth.status_code == 200
        assert withauth.json()["ok"] is True

    def test_forward_error_returns_502(self) -> None:
        with httpx.Client() as client:
            unreachable = "http://127.0.0.1:1"
            with _running_server(forward_url=unreachable, client=client) as base:
                resp = httpx.post(
                    f"{base}/v1/responses", json={"input": "x"}, timeout=5.0
                )
        assert resp.status_code == 502


class TestMalformedRequests:
    def _raw_post(self, base: str, headers: str) -> int:
        host, port = base.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=5.0) as sock:
            sock.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                f"{headers}\r\n\r\n".encode()
            )
            status_line = sock.recv(256).decode("latin-1").splitlines()[0]
        return int(status_line.split()[1])

    def test_invalid_content_length_returns_400(self) -> None:
        with _running_server() as base:
            assert self._raw_post(base, "Content-Length: abc") == 400

    def test_oversized_body_returns_413(self) -> None:
        with _running_server() as base:
            assert self._raw_post(base, "Content-Length: 999999999") == 413


class TestRunLifecycle:
    def _make_executor(self) -> DevModelExecutor:
        return DevModelExecutor(
            make_worker_config(enable_dev_model=True), make_worker_hardware()
        )

    def test_run_emits_endpoint_and_returns_result(self, tmp_path: Path) -> None:
        spec = DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            model=ModelConfig(source=ModelSource(identifier="dev/model")),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        emit = MagicMock()
        with (
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_wait_for_serve"),
        ):
            result = ex.run(task, tmp_path)

        serve = emit.call_args.args[1]["serve"]
        assert serve["mode"] == "forward"
        assert serve["host"] == "127.0.0.1"
        assert serve["_relay_target"] == {"host": "127.0.0.1", "port": serve["port"]}
        assert serve["model"] == "dev/model"
        assert isinstance(result, DevModelResult)
        assert result.model == "dev/model"
        assert result.port == serve["port"]
        assert "api_key" not in serve

    def test_direct_mode_binds_all_interfaces_and_advertises_fqdn(
        self, tmp_path: Path
    ) -> None:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL, accessMode="direct")
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        emit = MagicMock()
        bind: dict[str, object] = {}

        def capture_bind(_ttl: float) -> None:
            bind["host"] = ex._server.server_address[0]  # type: ignore[union-attr]

        with (
            patch("socket.getfqdn", return_value="worker-1.cluster.local"),
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_wait_for_serve", side_effect=capture_bind),
        ):
            ex.run(task, tmp_path)

        serve = emit.call_args.args[1]["serve"]
        assert bind["host"] == "0.0.0.0"
        assert serve["mode"] == "direct"
        assert serve["host"] == "worker-1.cluster.local"

    def test_run_serves_canned_endpoint_while_alive(self, tmp_path: Path) -> None:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL)
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        reached: dict[str, object] = {}

        def hit_then_stop(_ttl: float) -> None:
            port = ex._server.server_address[1]  # type: ignore[union-attr]
            reached["payload"] = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=5.0,
            ).json()

        with patch.object(ex, "_wait_for_serve", side_effect=hit_then_stop):
            ex.run(task, tmp_path)

        payload = reached["payload"]
        assert isinstance(payload, dict)
        assert payload["choices"][0]["message"]["content"] == _CANNED_TEXT


class TestCancelStop:
    def _make_executor(self) -> DevModelExecutor:
        return DevModelExecutor(
            make_worker_config(enable_dev_model=True), make_worker_hardware()
        )

    def test_cancel_sets_event_and_shuts_down_server(self) -> None:
        ex = self._make_executor()
        server = MagicMock()
        ex._server = server
        ex.cancel("tsk-test")
        assert ex._cancel_event.is_set()
        server.shutdown.assert_called_once_with()

    def test_stop_sets_event_and_shuts_down_server(self) -> None:
        ex = self._make_executor()
        server = MagicMock()
        ex._server = server
        ex.stop("tsk-test")
        assert ex._stop_event.is_set()
        server.shutdown.assert_called_once_with()

    def test_cancel_no_server_is_safe(self) -> None:
        ex = self._make_executor()
        ex._server = None
        ex.cancel("tsk-test")

    def test_wait_for_serve_unblocks_on_stop(self) -> None:
        ex = self._make_executor()
        ex._stop_event.set()
        orig = mod._POLL_INTERVAL_SEC
        mod._POLL_INTERVAL_SEC = 0.01
        try:
            ex._wait_for_serve(ttl_sec=60.0)
        finally:
            mod._POLL_INTERVAL_SEC = orig

    def test_wait_for_serve_raises_on_cancel(self) -> None:
        ex = self._make_executor()
        ex._cancel_event.set()
        with pytest.raises(TaskCancelledError):
            ex._wait_for_serve(ttl_sec=60.0)
