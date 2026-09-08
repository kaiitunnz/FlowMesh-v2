"""The replica engine delivery routes by interface and carries opaque content.

A chat endpoint POSTs ``/chat/completions`` and streams the assistant message text; an
embedding endpoint POSTs ``/embeddings`` and streams the ``data`` vectors serialized as
JSON, so both ride the same content path.
"""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import httpx
import pytest

from shared.resident.contracts import ReplicaEndpoint
from worker.resident.engine import (
    HttpEngineDelivery,
    RawHttpEngineDelivery,
    unload_adapter,
)


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.paths.append(self.path)
        self.server.bodies.append(body)
        if self.path == "/v1/load_lora_adapter":
            self.server.loaded.append(body.get("lora_name"))
            status, message = self.server.load_response
            loaded = json.dumps({"message": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(loaded)))
            self.end_headers()
            self.wfile.write(loaded)
            return
        if self.path == "/v1/unload_lora_adapter":
            self.server.unloaded.append(body.get("lora_name"))
            status, message = self.server.unload_response
            unloaded = json.dumps({"message": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(unloaded)))
            self.end_headers()
            self.wfile.write(unloaded)
            return
        if self.path == "/v1/embeddings":
            n = len(body.get("input") or [])
            payload: dict[str, Any] = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i)]}
                    for i in range(n)
                ],
            }
        else:
            if self.server.chat_override is not None:
                status, ctype, raw = self.server.chat_override
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "hi there"}}]
            }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.paths: list[str] = []
        self.bodies: list[dict[str, Any]] = []
        self.loaded: list[str | None] = []
        self.load_response: tuple[int, str] = (200, "success")
        self.unloaded: list[str | None] = []
        self.unload_response: tuple[int, str] = (200, "success")
        self.chat_override: tuple[int, str, bytes] | None = None


@contextmanager
def _running() -> Iterator[_Server]:
    server = _Server()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


async def _drain(
    endpoint: ReplicaEndpoint,
    payload: str | None,
    adapter_name: str | None = None,
    adapter_source: str | None = None,
) -> str:
    opened = await HttpEngineDelivery()(endpoint, payload, adapter_name, adapter_source)
    parts = [chunk async for chunk in opened.chunks]
    await opened.aclose()
    return "".join(parts)


def test_chat_interface_posts_chat_completions_and_streams_text() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        content = asyncio.run(_drain(endpoint, "hello"))
    assert server.paths == ["/v1/chat/completions"]
    assert content == "hi there"


def test_adapter_bound_invocation_loads_then_selects_the_adapter() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="base-model", interface="chat")
        content = asyncio.run(
            _drain(endpoint, "hi", adapter_name="my-lora", adapter_source="hf/my-lora")
        )
    assert server.loaded == ["my-lora"]
    assert server.paths == ["/v1/load_lora_adapter", "/v1/chat/completions"]
    # The request selects the adapter as its model, not the base.
    assert server.bodies[1]["model"] == "my-lora"
    assert content == "hi there"


def test_already_loaded_adapter_is_idempotent_and_still_selects() -> None:
    with _running() as server:
        server.load_response = (400, "LoRA adapter 'my-lora' has already been loaded")
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="base-model", interface="chat")
        content = asyncio.run(
            _drain(endpoint, "hi", adapter_name="my-lora", adapter_source="hf/my-lora")
        )
    # An already-loaded response is tolerated; the request still selects the adapter.
    assert server.paths == ["/v1/load_lora_adapter", "/v1/chat/completions"]
    assert content == "hi there"


def test_a_precise_load_error_fails_the_invocation() -> None:
    with _running() as server:
        server.load_response = (400, "invalid lora_path: no adapter_config.json found")
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="base-model", interface="chat")
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(
                _drain(endpoint, "hi", adapter_name="my-lora", adapter_source="bad")
            )
    # The engine request is never sent when the load fails loudly.
    assert server.paths == ["/v1/load_lora_adapter"]


def test_unload_adapter_posts_unload_lora_adapter() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        asyncio.run(unload_adapter(endpoint, "my-lora"))
    assert server.paths == ["/v1/unload_lora_adapter"]
    assert server.bodies[0] == {"lora_name": "my-lora"}
    assert server.unloaded == ["my-lora"]


def test_unload_of_a_missing_adapter_is_idempotent() -> None:
    with _running() as server:
        server.unload_response = (404, "adapter 'my-lora' not found")
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        asyncio.run(unload_adapter(endpoint, "my-lora"))  # no raise
    assert server.paths == ["/v1/unload_lora_adapter"]


def test_a_precise_unload_error_raises() -> None:
    with _running() as server:
        server.unload_response = (500, "internal engine error")
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(unload_adapter(endpoint, "my-lora"))


async def _drain_raw(
    endpoint: ReplicaEndpoint, payload: str | None
) -> tuple[int, str, str]:
    opened = await RawHttpEngineDelivery()(endpoint, payload)
    parts = [chunk async for chunk in opened.chunks]
    await opened.aclose()
    return opened.status, opened.content_type, "".join(parts)


def test_raw_delivery_relays_status_content_type_and_body_verbatim() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, content_type, body = asyncio.run(_drain_raw(endpoint, "hello"))
    assert server.paths == ["/v1/chat/completions"]
    assert status == 200
    assert content_type.startswith("application/json")
    # The raw path relays the engine envelope verbatim, not the extracted content.
    assert json.loads(body)["choices"][0]["message"]["content"] == "hi there"


def test_raw_delivery_streams_an_sse_body_and_content_type() -> None:
    sse = 'data: {"delta":"hi"}\n\ndata: [DONE]\n\n'
    with _running() as server:
        server.chat_override = (200, "text/event-stream", sse.encode())
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, content_type, body = asyncio.run(
            _drain_raw(endpoint, '{"messages":[],"stream":true}')
        )
    assert status == 200
    assert content_type.startswith("text/event-stream")
    assert body == sse


def test_raw_delivery_relays_an_engine_error_status_without_raising() -> None:
    with _running() as server:
        server.chat_override = (400, "application/json", b'{"error":"bad request"}')
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, _content_type, body = asyncio.run(_drain_raw(endpoint, "hi"))
    assert status == 400
    assert json.loads(body)["error"] == "bad request"


def test_embedding_interface_posts_embeddings_and_streams_vectors() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="embedding")
        content = asyncio.run(_drain(endpoint, json.dumps({"input": ["a", "b"]})))
    assert server.paths == ["/v1/embeddings"]
    assert server.bodies[0] == {"input": ["a", "b"], "model": "m"}
    vectors = json.loads(content)
    assert [v["index"] for v in vectors] == [0, 1]
    assert vectors[0]["embedding"] == [0.0]
