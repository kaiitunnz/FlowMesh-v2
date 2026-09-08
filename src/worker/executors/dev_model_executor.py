"""GPU-free model-serving executor.

Stands up an OpenAI-compatible HTTP endpoint (Chat Completions, Responses, and
Embeddings) without a GPU, emits a TASK_UPDATE with the endpoint details, and
blocks until the TTL expires or a stop command arrives. Requests either forward to a
live upstream model endpoint (``dev_model_forward_url``) or return deterministic
canned responses when no upstream is configured.
"""

import contextlib
import json
import logging
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
from .utils.net import resolve_bind_port

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SEC = 3600.0
_MAX_TTL_SEC = 86400.0
_POLL_INTERVAL_SEC = 5.0
_FORWARD_TIMEOUT_SEC = 120.0
_MAX_BODY_BYTES = 10 * 1024 * 1024
_ROUTES = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/embeddings"})
_LOAD_ADAPTER_ROUTE = "/v1/load_lora_adapter"
_UNLOAD_ADAPTER_ROUTE = "/v1/unload_lora_adapter"
_CANNED_TEXT = "This is a deterministic dev_model response."
_CANNED_EMBEDDING = [0.0, 0.0, 0.0, 0.0]


def _request_model(body: bytes, fallback: str) -> str:
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        return fallback
    model = payload.get("model") if isinstance(payload, dict) else None
    return model if isinstance(model, str) and model else fallback


def _canned_embeddings(body: bytes, model: str) -> dict[str, Any]:
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        payload = {}
    raw = payload.get("input") if isinstance(payload, dict) else None
    inputs = raw if isinstance(raw, list) else [raw]
    return {
        "object": "list",
        "model": model,
        "data": [
            {"object": "embedding", "index": i, "embedding": list(_CANNED_EMBEDDING)}
            for i, _ in enumerate(inputs)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _canned_text(model: str) -> str:
    # The served model rides the response text so a caller can tell an adapter's output
    # from the base model's on the GPU-free stand-in.
    return f"{_CANNED_TEXT} [model={model}]"


def _canned_response(path: str, model: str) -> dict[str, Any]:
    text = _canned_text(model)
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
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "output_text": text,
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
                "message": {"role": "assistant", "content": text},
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
        response_delay_sec: float = 0.0,
        max_loras: int | None = None,
    ) -> None:
        super().__init__(address, handler)
        self.forward_url = forward_url
        self.model_name = model_name
        self.client = client
        self.response_delay_sec = max(0.0, response_delay_sec)
        self.max_loras = max_loras
        self.loaded_adapters: set[str] = set()


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

    def _adapter_name(self) -> tuple[str, bytes] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"error": "invalid Content-Length"})
            return None
        body = self.rfile.read(length) if 0 < length <= _MAX_BODY_BYTES else b""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            payload = {}
        name = payload.get("lora_name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name:
            self._write_json(400, {"error": "lora_name is required"})
            return None
        return name, body

    def _forward_adapter(self, route: str, body: bytes) -> None:
        server = self.server
        if server.forward_url is not None and server.client is not None:
            # Mirror the load/unload to the upstream so a real serve tracks the same
            # registry; a non-fatal upstream response (already loaded, or gone) is
            # tolerated.
            with contextlib.suppress(httpx.RequestError):
                server.client.post(
                    server.forward_url.rstrip("/") + route,
                    content=body,
                    headers={"Content-Type": "application/json"},
                    timeout=_FORWARD_TIMEOUT_SEC,
                )

    def _load_adapter(self) -> None:
        parsed = self._adapter_name()
        if parsed is None:
            return
        name, body = parsed
        server = self.server
        if (
            server.max_loras is not None
            and name not in server.loaded_adapters
            and len(server.loaded_adapters) >= server.max_loras
        ):
            # A finite adapter registry, like a real engine's slot budget: a new
            # distinct adapter cannot load until an occupied slot is unloaded. This is
            # what the last-holder unload frees, so a lifetime-distinct sequence serves.
            self._write_json(400, {"error": f"no free LoRA slot for {name!r}"})
            return
        server.loaded_adapters.add(name)
        self._forward_adapter(_LOAD_ADAPTER_ROUTE, body)
        self._write_json(200, {"status": "success", "lora_name": name})

    def _unload_adapter(self) -> None:
        parsed = self._adapter_name()
        if parsed is None:
            return
        name, body = parsed
        # Idempotent: an adapter already gone is not an error; the slot frees anyway.
        self.server.loaded_adapters.discard(name)
        self._forward_adapter(_UNLOAD_ADAPTER_ROUTE, body)
        self._write_json(200, {"status": "success", "lora_name": name})

    def do_GET(self) -> None:
        # A read-only endpoint (e.g. GET /v1/models): a real engine serves these, so the
        # stand-in forwards them upstream too, or answers a canned model list when no
        # upstream is configured, so the transparent serve surface reaches any path.
        server = self.server
        if server.forward_url is not None and server.client is not None:
            self._forward_get(
                server.client,
                server.forward_url,
                self.path,
                self.headers.get("Authorization"),
            )
        elif self.path.rstrip("/") == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": server.model_name,
                            "object": "model",
                            "owned_by": "flowmesh-dev",
                        }
                    ],
                },
            )
        else:
            self._write_json(404, {"error": f"unknown route {self.path}"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path == _LOAD_ADAPTER_ROUTE:
            self._load_adapter()
            return
        if path == _UNLOAD_ADAPTER_ROUTE:
            self._unload_adapter()
            return
        if path not in _ROUTES:
            self._write_json(404, {"error": f"unknown route {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"error": "invalid Content-Length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._write_json(413, {"error": "request body too large"})
            return
        body = self.rfile.read(length) if length else b""
        server = self.server
        if server.response_delay_sec > 0:
            # A test seam (DEV_MODEL_RESPONSE_DELAY_SEC): hold each response so an
            # in-flight invocation keeps its admission slot occupied, making a
            # slot-exhaustion race deterministic without a body marker.
            time.sleep(server.response_delay_sec)
        requested = _request_model(body, server.model_name)
        if (
            server.loaded_adapters
            and requested != server.model_name
            and requested not in server.loaded_adapters
        ):
            # Load-before-select: once the replica holds adapters, a request selecting a
            # non-base model must name a loaded adapter, so it never silently serves the
            # base in place of an unloaded adapter.
            self._write_json(404, {"error": f"adapter {requested!r} is not loaded"})
            return
        if server.forward_url is not None and server.client is not None:
            self._forward(
                server.client,
                server.forward_url,
                path,
                body,
                self.headers.get("Authorization"),
            )
        else:
            model = _request_model(body, server.model_name)
            if path == "/v1/embeddings":
                self._write_json(200, _canned_embeddings(body, model))
            else:
                self._write_json(200, _canned_response(path, model))

    def _forward(
        self,
        client: httpx.Client,
        forward_url: str,
        path: str,
        body: bytes,
        authorization: str | None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        try:
            resp = client.post(
                forward_url.rstrip("/") + path,
                content=body,
                headers=headers,
                timeout=_FORWARD_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            self._write_json(502, {"error": f"dev_model forward failed: {exc}"})
            return
        self.send_response(resp.status_code)
        self.send_header(
            "Content-Type", resp.headers.get("Content-Type", "application/json")
        )
        self.send_header("Content-Length", str(len(resp.content)))
        self.end_headers()
        self.wfile.write(resp.content)

    def _forward_get(
        self,
        client: httpx.Client,
        forward_url: str,
        path: str,
        authorization: str | None,
    ) -> None:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization
        try:
            resp = client.get(
                forward_url.rstrip("/") + path,
                headers=headers,
                timeout=_FORWARD_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            self._write_json(502, {"error": f"dev_model forward failed: {exc}"})
            return
        self.send_response(resp.status_code)
        self.send_header(
            "Content-Type", resp.headers.get("Content-Type", "application/json")
        )
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
        vllm = (spec.model.vllm if spec.model is not None else None) or {}
        raw_max_loras = vllm.get("max_loras")
        max_loras = raw_max_loras if isinstance(raw_max_loras, int) else None
        # Loopback only: the endpoint is reached solely by its co-located claim-gated
        # sidecar and, externally, only through the gated task-ID serve route.
        bind_host = "127.0.0.1"
        port = resolve_bind_port(spec.port, bind_host)
        forward_url = self._config.dev_model_forward_url

        out_dir.mkdir(parents=True, exist_ok=True)
        if self._stop_event.is_set():
            raise TaskCancelledError(
                f"dev_model task {task.task_id} stopped before launch"
            )

        client = httpx.Client() if forward_url is not None else None
        try:
            server = _DevModelHTTPServer(
                (bind_host, port),
                _DevModelHandler,
                forward_url,
                model_id,
                client,
                self._config.dev_model_response_delay_sec,
                max_loras,
            )
        except BaseException:
            if client is not None:
                client.close()
            raise
        self._server = server
        serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        serve_thread.start()

        logger.info(
            "dev_model server ready for model %s on port %d "
            "(task=%s ttl=%.0fs forward=%s)",
            model_id,
            port,
            task.task_id,
            ttl_sec,
            forward_url or "canned",
        )

        try:
            # Worker-private endpoint facts ("_"-prefixed so task metadata never
            # discloses the raw loopback listener); the resident endpoint probe reads
            # them to bind the claim-gated sidecar in front of the endpoint.
            interface = "embedding" if vllm.get("runner") == "pooling" else "chat"
            self.emit_update(
                task.task_id,
                {
                    "serve": {
                        "model": model_id,
                        "interface": interface,
                        "_host": "127.0.0.1",
                        "_port": port,
                        "_api_key": None,
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
