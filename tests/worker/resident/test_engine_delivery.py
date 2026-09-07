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

from shared.resident.contracts import ReplicaEndpoint
from worker.resident.engine import HttpEngineDelivery


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
            loaded = json.dumps({"status": "success"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(loaded)))
            self.end_headers()
            self.wfile.write(loaded)
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
