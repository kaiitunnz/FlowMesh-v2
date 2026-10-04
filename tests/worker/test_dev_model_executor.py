"""Tests for DevModelExecutor."""

import json
import shutil
import socket
import tempfile
import threading
import time
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
    make_dev_model_executor,
    make_worker_config,
    make_worker_task_message,
)
from worker.executors import dev_model_executor as mod
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.dev_model_executor import (
    _CANNED_TEXT,
    DevModelExecutor,
    _DevModelHandler,
    _DevModelHTTPServer,
)
from worker.resident import LocalEngine


@contextmanager
def _running_server(
    forward_url: str | None = None,
    model_name: str = "test-model",
    client: httpx.Client | None = None,
    max_loras: int | None = None,
) -> Iterator[httpx.Client]:
    """Serve the stand-in on a socket; yield a client that reaches it there."""
    directory = Path(tempfile.mkdtemp())
    path = (directory / "engine.sock").as_posix()
    server = _DevModelHTTPServer(
        path,
        _DevModelHandler,
        forward_url,
        model_name,
        client,
        max_loras=max_loras,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with _socket_client(path) as reach:
            yield reach
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        shutil.rmtree(directory)


def _socket_client(path: str) -> httpx.Client:
    return httpx.Client(
        base_url="http://localhost", transport=httpx.HTTPTransport(uds=path)
    )


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
        assert spec.port is None

    def test_spec_with_all_fields(self) -> None:
        spec = DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            model=ModelConfig(source=ModelSource(identifier="dev/model")),
            ttlSeconds=60.0,
            port=8123,
        )
        assert spec.model_name == "dev/model"
        assert spec.ttlSeconds == 60.0
        assert spec.port == 8123

    def test_accepts_both_gated_access_modes(self) -> None:
        for mode in ("proxy", "forward"):
            assert (
                DevModelSpecStrict(
                    taskType=TaskType.DEV_MODEL, accessMode=mode
                ).accessMode
                == mode
            )

    def test_rejects_the_removed_direct_access_mode(self) -> None:
        # ``direct`` named a raw ungated listener, which is no longer a mode at all.
        with pytest.raises(Exception):
            DevModelSpecStrict(taskType=TaskType.DEV_MODEL, accessMode="direct")  # type: ignore[arg-type]

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
            first = base.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=5.0,
            ).json()
            second = base.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=5.0,
            ).json()
        assert first == second
        assert first["object"] == "chat.completion"
        assert _CANNED_TEXT in first["choices"][0]["message"]["content"]
        assert first["model"] == "m"

    def test_responses_is_deterministic(self) -> None:
        with _running_server() as base:
            payload = base.post(
                "/v1/responses",
                json={"model": "m", "input": "hi"},
                timeout=5.0,
            ).json()
        assert payload["object"] == "response"
        assert payload["status"] == "completed"
        assert _CANNED_TEXT in payload["output_text"]
        assert _CANNED_TEXT in payload["output"][0]["content"][0]["text"]

    def test_model_falls_back_when_absent(self) -> None:
        with _running_server(model_name="fallback-model") as base:
            payload = base.post(
                "/v1/chat/completions", json={"messages": []}, timeout=5.0
            ).json()
        assert payload["model"] == "fallback-model"

    def test_embeddings_returns_one_vector_per_input(self) -> None:
        with _running_server() as base:
            payload = base.post(
                "/v1/embeddings",
                json={"model": "m", "input": ["a", "b", "c"]},
                timeout=5.0,
            ).json()
        assert payload["object"] == "list"
        assert payload["model"] == "m"
        assert [d["index"] for d in payload["data"]] == [0, 1, 2]
        assert all(isinstance(d["embedding"], list) for d in payload["data"])

    def test_unknown_route_returns_404(self) -> None:
        with _running_server() as base:
            resp = base.post("/v1/unknown", json={}, timeout=5.0)
        assert resp.status_code == 404

    def test_load_lora_adapter_records_and_succeeds(self) -> None:
        with _running_server() as base:
            resp = base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "my-lora", "lora_path": "hf/my-lora"},
                timeout=5.0,
            )
        assert resp.status_code == 200
        assert resp.json()["lora_name"] == "my-lora"

    def test_loaded_adapter_is_selectable_after_load(self) -> None:
        with _running_server() as base:
            base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "my-lora", "lora_path": "hf/my-lora"},
                timeout=5.0,
            )
            resp = base.post(
                "/v1/chat/completions",
                json={"model": "my-lora", "messages": []},
                timeout=5.0,
            )
        assert resp.status_code == 200
        assert resp.json()["model"] == "my-lora"

    def test_selecting_an_unloaded_adapter_after_a_load_is_404(self) -> None:
        with _running_server() as base:
            base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "my-lora", "lora_path": "hf/my-lora"},
                timeout=5.0,
            )
            resp = base.post(
                "/v1/chat/completions",
                json={"model": "other-lora", "messages": []},
                timeout=5.0,
            )
        assert resp.status_code == 404

    def test_a_full_adapter_registry_refuses_a_new_distinct_load(self) -> None:
        with _running_server(max_loras=1) as base:
            first = base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "lora-a", "lora_path": "hf/lora-a"},
                timeout=5.0,
            )
            second = base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "lora-b", "lora_path": "hf/lora-b"},
                timeout=5.0,
            )
        assert first.status_code == 200
        assert second.status_code == 400  # no free slot until one is unloaded

    def test_unload_frees_a_slot_for_a_later_distinct_load(self) -> None:
        with _running_server(max_loras=1) as base:
            base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "lora-a", "lora_path": "hf/lora-a"},
                timeout=5.0,
            )
            unloaded = base.post(
                "/v1/unload_lora_adapter",
                json={"lora_name": "lora-a"},
                timeout=5.0,
            )
            reused = base.post(
                "/v1/load_lora_adapter",
                json={"lora_name": "lora-b", "lora_path": "hf/lora-b"},
                timeout=5.0,
            )
        assert unloaded.status_code == 200
        assert reused.status_code == 200  # the freed slot admits the new adapter

    def test_unload_of_an_absent_adapter_is_idempotent(self) -> None:
        with _running_server(max_loras=1) as base:
            resp = base.post(
                "/v1/unload_lora_adapter",
                json={"lora_name": "never-loaded"},
                timeout=5.0,
            )
        assert resp.status_code == 200


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
                resp = base.post(
                    "/v1/chat/completions",
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
                without = base.post("/v1/responses", json={}, timeout=5.0)
                withauth = base.post(
                    "/v1/responses",
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
                resp = base.post("/v1/responses", json={"input": "x"}, timeout=5.0)
        assert resp.status_code == 502


class TestMalformedRequests:
    def _raw_post(self, base: httpx.Client, headers: str) -> int:
        path = base._transport._pool._uds  # type: ignore[attr-defined]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(path)
            sock.sendall(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
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
        return make_dev_model_executor()

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
        assert serve["model"] == "dev/model"
        # Only the worker-private ("_"-prefixed) socket plus the model name and served
        # interface; no listener or credential.
        assert set(serve) == {"model", "interface", "_socket"}
        assert serve["interface"] == "chat"
        assert isinstance(result, DevModelResult)
        assert result.model == "dev/model"
        assert result.port is None

    def test_a_socket_path_past_the_unix_limit_fails_the_task_clearly(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / ("d" * 100)
        parent.mkdir()
        spec = DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            model=ModelConfig(source=ModelSource(identifier="dev/model")),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = make_dev_model_executor(engine_parent=parent)
        with (
            patch.object(ex, "emit_update"),
            patch.object(ex, "_wait_for_serve"),
            pytest.raises(ExecutionError, match="107-byte Unix socket limit") as raised,
        ):
            ex.run(task, tmp_path / "out")
        assert raised.value.retryable
        assert list(parent.iterdir()) == []

    def test_pooling_runner_serves_the_embedding_interface(
        self, tmp_path: Path
    ) -> None:
        # A pooling-runner serve task serves embeddings, so it advertises the embedding
        # interface and is adopted as an embedding allocation rather than always chat.
        spec = DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            model=ModelConfig(
                source=ModelSource(identifier="dev/embed"), vllm={"runner": "pooling"}
            ),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        emit = MagicMock()
        with (
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_wait_for_serve"),
        ):
            ex.run(task, tmp_path)
        assert emit.call_args.args[1]["serve"]["interface"] == "embedding"

    def test_listens_only_on_its_published_socket(self, tmp_path: Path) -> None:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL, port=8123)
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        emit = MagicMock()
        seen: dict[str, object] = {}

        def capture(_deadline: float) -> None:
            seen["address"] = ex._server.server_address  # type: ignore[union-attr]
            seen["engine"] = ex._local_engines().lookup(task.task_id)

        with (
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_wait_for_serve", side_effect=capture),
        ):
            ex.run(task, tmp_path)

        serve = emit.call_args.args[1]["serve"]
        assert seen["address"] == serve["_socket"]
        assert seen["engine"] == LocalEngine(serve["_socket"])
        assert ex._local_engines().lookup(task.task_id) is None
        assert not Path(serve["_socket"]).parent.exists()

    def test_run_serves_canned_endpoint_while_alive(self, tmp_path: Path) -> None:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL)
        task = make_worker_task_message(spec=spec, task_type=TaskType.DEV_MODEL)
        ex = self._make_executor()
        reached: dict[str, object] = {}

        def hit_then_stop(_deadline: float) -> None:
            path = ex._server.server_address  # type: ignore[union-attr]
            with _socket_client(str(path)) as reach:
                reached["payload"] = reach.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": []},
                    timeout=5.0,
                ).json()

        with patch.object(ex, "_wait_for_serve", side_effect=hit_then_stop):
            ex.run(task, tmp_path)

        payload = reached["payload"]
        assert isinstance(payload, dict)
        assert _CANNED_TEXT in payload["choices"][0]["message"]["content"]


class TestCancelStop:
    def _make_executor(self) -> DevModelExecutor:
        return make_dev_model_executor()

    def test_cancel_signals_the_running_task_and_shuts_down_server(self) -> None:
        ex = self._make_executor()
        server = MagicMock()
        ex._server = server
        with ex._signals.running("tsk-test"):
            ex.cancel("tsk-test")
            assert ex._signals.cancelled
        server.shutdown.assert_called_once_with()

    def test_stop_signals_the_running_task_and_shuts_down_server(self) -> None:
        ex = self._make_executor()
        server = MagicMock()
        ex._server = server
        with ex._signals.running("tsk-test"):
            ex.stop("tsk-test")
            assert ex._signals.stopped
        server.shutdown.assert_called_once_with()

    def test_a_signal_for_a_task_it_is_not_running_is_a_no_op(self) -> None:
        ex = self._make_executor()
        server = MagicMock()
        ex._server = server
        ex.cancel("tsk-done")
        ex.stop("tsk-done")
        with ex._signals.running("tsk-next"):
            assert not ex._signals.cancelled
            assert not ex._signals.stopped
        server.shutdown.assert_not_called()

    def test_cancel_no_server_is_safe(self) -> None:
        ex = self._make_executor()
        ex._server = None
        ex.cancel("tsk-test")

    def test_wait_for_serve_unblocks_on_stop(self) -> None:
        ex = self._make_executor()
        orig = mod._POLL_INTERVAL_SEC
        mod._POLL_INTERVAL_SEC = 0.01
        try:
            with ex._signals.running("tsk-test"):
                ex.stop("tsk-test")
                ex._wait_for_serve(deadline=time.time() + 60.0)
        finally:
            mod._POLL_INTERVAL_SEC = orig

    def test_wait_for_serve_raises_on_cancel(self) -> None:
        ex = self._make_executor()
        with ex._signals.running("tsk-test"), pytest.raises(TaskCancelledError):
            ex.cancel("tsk-test")
            ex._wait_for_serve(deadline=time.time() + 60.0)


class TestServeTtlAcrossReruns:
    def _run(
        self, tmp_path: Path, ttl: float, elapsed: float | None
    ) -> tuple[MagicMock, list[float]]:
        spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL, ttlSeconds=ttl)
        task = make_worker_task_message(
            spec=spec, task_type=TaskType.DEV_MODEL, serve_elapsed_sec=elapsed
        )
        ex = make_dev_model_executor()
        emit = MagicMock()
        deadlines: list[float] = []
        with (
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_wait_for_serve", side_effect=deadlines.append),
        ):
            ex.run(task, tmp_path)
        return emit, deadlines

    def test_a_re_run_serves_what_remains_of_the_ttl(self, tmp_path: Path) -> None:
        before = time.time()
        _emit, deadlines = self._run(tmp_path, ttl=180.0, elapsed=100.0)
        assert before + 80.0 - 1.0 <= deadlines[0] <= time.time() + 80.0

    def test_an_elapsed_ttl_starts_no_server(self, tmp_path: Path) -> None:
        emit, deadlines = self._run(tmp_path, ttl=180.0, elapsed=200.0)
        emit.assert_not_called()
        assert deadlines == []
