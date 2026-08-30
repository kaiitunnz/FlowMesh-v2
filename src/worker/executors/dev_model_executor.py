"""GPU-free model-serving executor for agent / model-gateway development.

Stands up an OpenAI-compatible HTTP endpoint (Chat Completions + Responses)
without a GPU, emits a TASK_UPDATE with the endpoint details, and blocks until
the TTL expires or a stop command arrives. Requests either forward to a live
upstream model endpoint (``dev_model_forward_url``) or return deterministic
canned responses when no upstream is configured.
"""

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from shared.schemas.result import DevModelResult
from shared.tasks.specs.dev_model import DevModelSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import parse_float_env
from worker.config import WorkerConfig

from .base_executor import Executor, ExecutorTask, TaskCancelledError
from .vllm_serve_executor import _resolve_port

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SEC = 3600.0
_MAX_TTL_SEC = 86400.0
_POLL_INTERVAL_SEC = 5.0
_FORWARD_TIMEOUT_SEC = 120.0
_ROUTES = frozenset({"/v1/chat/completions", "/v1/responses"})
_CANNED_TEXT = "This is a deterministic dev_model response."


def _request_model(body: bytes, fallback: str) -> str:
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, ValueError):
        return fallback
    model = payload.get("model") if isinstance(payload, dict) else None
    return model if isinstance(model, str) and model else fallback


def _canned_response(path: str, model: str) -> dict[str, Any]:
    if path == "/v1/responses":
        return {
            "id": "dev-model-resp",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "dev-model-msg",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": _CANNED_TEXT, "annotations": []}
                    ],
                }
            ],
            "output_text": _CANNED_TEXT,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    return {
        "id": "dev-model-chatcmpl",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _CANNED_TEXT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class _DevModelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        forward_url: str | None,
        model_name: str,
        client: httpx.Client | None,
    ) -> None:
        super().__init__(address, handler)
        self.forward_url = forward_url
        self.model_name = model_name
        self.client = client


class _DevModelHandler(BaseHTTPRequestHandler):
    server: _DevModelHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("[dev_model] " + format, *args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path not in _ROUTES:
            self._write_json(404, {"error": f"unknown route {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        server = self.server
        if server.forward_url is not None and server.client is not None:
            self._forward(server.client, server.forward_url, path, body)
        else:
            model = _request_model(body, server.model_name)
            self._write_json(200, _canned_response(path, model))

    def _forward(
        self, client: httpx.Client, forward_url: str, path: str, body: bytes
    ) -> None:
        try:
            resp = client.post(
                forward_url.rstrip("/") + path,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=_FORWARD_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            self._write_json(502, {"error": f"dev_model forward failed: {exc}"})
            return
        self.send_response(resp.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp.content)))
        self.end_headers()
        self.wfile.write(resp.content)


class DevModelExecutor(Executor):
    name = "dev_model"
    supported_task_types = frozenset({TaskType.DEV_MODEL})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._server: _DevModelHTTPServer | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return config.enable_dev_model

    def run(self, task: ExecutorTask, out_dir: Path) -> DevModelResult:
        spec = self.require_spec(task, DevModelSpecStrict)

        model_id = spec.model_name or "dev-model"
        ttl_sec = min(
            spec.ttlSeconds
            or parse_float_env("SERVE_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC),
            parse_float_env("SERVE_MAX_TTL_SEC", _MAX_TTL_SEC),
        )
        access_mode = spec.accessMode or "forward"
        bind_host = (
            "0.0.0.0" if access_mode == "direct" else "127.0.0.1"
        )  # nosec B104 - direct mode is an explicit opt-in to a client-reachable endpoint
        port = _resolve_port(spec.port, bind_host)
        forward_url = self._config.dev_model_forward_url

        out_dir.mkdir(parents=True, exist_ok=True)
        if self._stop_event.is_set():
            raise TaskCancelledError(
                f"dev_model task {task.task_id} stopped before launch"
            )

        client = httpx.Client() if forward_url is not None else None
        server = _DevModelHTTPServer(
            (bind_host, port), _DevModelHandler, forward_url, model_id, client
        )
        self._server = server
        serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        serve_thread.start()

        logger.info(
            "dev_model server ready for model %s on port %d "
            "(task=%s mode=%s ttl=%.0fs forward=%s)",
            model_id,
            port,
            task.task_id,
            access_mode,
            ttl_sec,
            forward_url or "canned",
        )

        try:
            advertised_host = (
                socket.getfqdn() if access_mode == "direct" else "127.0.0.1"
            )
            self.emit_update(
                task.task_id,
                {
                    "serve": {
                        "mode": access_mode,
                        "_relay_target": {"host": "127.0.0.1", "port": port},
                        "host": advertised_host,
                        "port": port,
                        "model": model_id,
                    }
                },
            )
            self._wait_for_serve(ttl_sec)
        finally:
            self._server = None
            self._cancel_event.clear()
            self._stop_event.clear()
            server.shutdown()
            server.server_close()
            if client is not None:
                client.close()
            serve_thread.join(timeout=5.0)

        return DevModelResult(model=model_id, port=port)

    def _wait_for_serve(self, ttl_sec: float) -> None:
        deadline = time.time() + ttl_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("dev_model task cancelled")
            if self._stop_event.is_set():
                logger.info("dev_model task stop requested; terminating server")
                return
            time.sleep(_POLL_INTERVAL_SEC)
        logger.info("dev_model task TTL reached; terminating server")

    def cancel(self, task_id: str) -> None:
        self._cancel_event.set()
        if (server := self._server) is not None:
            server.shutdown()

    def stop(self, task_id: str) -> None:
        self._stop_event.set()
        if (server := self._server) is not None:
            server.shutdown()
