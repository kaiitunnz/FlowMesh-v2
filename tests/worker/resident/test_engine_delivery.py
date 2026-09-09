"""The replica engine deliveries: parsed content for workflows, verbatim serve proxy.

A workflow consumer routes by interface — a chat endpoint POSTs ``/chat/completions``
and streams the assistant message text, an embedding endpoint POSTs ``/embeddings`` and
streams the ``data`` vectors as JSON — so both ride the same content path. A
task-addressed serve request is instead replayed against the engine exactly as the
client sent it and its response relayed back unchanged.
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
from shared.resident.envelope import ServeRequestEnvelope, freeze_request_envelope
from worker.resident.engine import (
    HttpEngineDelivery,
    RawHttpEngineDelivery,
    unload_adapter,
)


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, *args: Any) -> None:
        pass

    def _record(self) -> bytes:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": list(self.headers.items()),
                "body": raw,
            }
        )
        return raw

    def _reply(self, status: int, ctype: str, raw: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        for name, value in self.server.extra_response_headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._record()
        if self.server.chat_override is not None:
            self._reply(*self.server.chat_override)
            return
        self._reply(
            200, "application/json", json.dumps({"data": [{"id": "m"}]}).encode()
        )

    def do_PUT(self) -> None:
        self._record()
        self._reply(200, "application/json", b"{}")

    def do_DELETE(self) -> None:
        self._record()
        self._reply(204, "application/json", b"")

    def do_POST(self) -> None:
        raw = self._record()
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
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
                self._reply(*self.server.chat_override)
                return
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "hi there"}}]
            }
        self._reply(200, "application/json", json.dumps(payload).encode())


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
        self.seen: list[dict[str, Any]] = []
        self.extra_response_headers: list[tuple[str, str]] = []


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
    endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
) -> tuple[int, tuple[tuple[str, str], ...], bytes]:
    opened = await RawHttpEngineDelivery()(endpoint, envelope)
    parts = [chunk async for chunk in opened.chunks]
    await opened.aclose()
    return opened.status, opened.headers, b"".join(parts)


def _serve_envelope(
    method: str = "POST",
    path: str = "v1/chat/completions",
    body: bytes = b"{}",
    query: str = "",
    headers: list[tuple[str, str]] | None = None,
) -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method=method,
        upstream_path=path,
        query=query,
        headers=[("content-type", "application/json")] if headers is None else headers,
        body=body,
    )


def _content_type(headers: tuple[tuple[str, str], ...]) -> str:
    return next(v for k, v in headers if k.lower() == "content-type")


def test_raw_delivery_relays_status_headers_and_body_verbatim() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, headers, body = asyncio.run(
            _drain_raw(endpoint, _serve_envelope(body=b'{"messages":[]}'))
        )
    assert server.paths == ["/v1/chat/completions"]
    assert status == 200
    assert _content_type(headers).startswith("application/json")
    # The raw path relays the engine envelope verbatim, not the extracted content.
    assert json.loads(body)["choices"][0]["message"]["content"] == "hi there"


def test_raw_delivery_forwards_the_client_method_path_query_and_body_unchanged() -> (
    None
):
    # The client's own request reaches the engine: an endpoint outside the interface's
    # classification, a non-POST method, its query string, and its exact body bytes.
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        asyncio.run(
            _drain_raw(
                endpoint,
                _serve_envelope(method="GET", path="v1/models", query="limit=2"),
            )
        )
        asyncio.run(
            _drain_raw(
                endpoint,
                _serve_envelope(path="v1/responses", body=b'{"model":"other","x":1}'),
            )
        )
    assert [(s["method"], s["path"]) for s in server.seen] == [
        ("GET", "/v1/models?limit=2"),
        ("POST", "/v1/responses"),
    ]
    # No body rebuild and no model override: the client's bytes arrive untouched.
    assert server.seen[1]["body"] == b'{"model":"other","x":1}'


def test_raw_delivery_swaps_the_client_credential_for_the_engine_one() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(
            base_url=base, model="m", interface="chat", api_key="engine-key"
        )
        # A client Authorization never survives the freeze, so it cannot reach upstream.
        envelope = _serve_envelope(
            headers=[("authorization", "Bearer client-token"), ("x-trace", "t1")]
        )
        asyncio.run(_drain_raw(endpoint, envelope))
    sent = {name.lower(): value for name, value in server.seen[0]["headers"]}
    assert sent["authorization"] == "Bearer engine-key"
    assert sent["x-trace"] == "t1"


def test_raw_delivery_does_not_invent_headers_the_client_never_sent() -> None:
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        asyncio.run(_drain_raw(endpoint, _serve_envelope(headers=[("x-only", "1")])))
    sent = {name.lower() for name, _ in server.seen[0]["headers"]}
    assert "x-only" in sent
    assert not sent & {"accept", "accept-encoding", "user-agent"}


def test_raw_delivery_preserves_every_non_hop_by_hop_response_header() -> None:
    with _running() as server:
        server.extra_response_headers = [("X-Request-Id", "r1"), ("Keep-Alive", "t=5")]
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        _status, headers, _body = asyncio.run(_drain_raw(endpoint, _serve_envelope()))
    names = {name.lower() for name, _ in headers}
    assert ("X-Request-Id", "r1") in headers
    # Hop-by-hop fields belong to the engine hop and are not relayed onward.
    assert "keep-alive" not in names


def test_raw_delivery_streams_an_sse_body_and_content_type() -> None:
    sse = 'data: {"delta":"hi"}\n\ndata: [DONE]\n\n'
    with _running() as server:
        server.chat_override = (200, "text/event-stream", sse.encode())
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, headers, body = asyncio.run(
            _drain_raw(endpoint, _serve_envelope(body=b'{"messages":[],"stream":true}'))
        )
    assert status == 200
    assert _content_type(headers).startswith("text/event-stream")
    assert body == sse.encode()


def test_raw_delivery_relays_an_engine_error_status_without_raising() -> None:
    with _running() as server:
        server.chat_override = (400, "application/json", b'{"error":"bad request"}')
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        status, _headers, body = asyncio.run(_drain_raw(endpoint, _serve_envelope()))
    assert status == 400
    assert json.loads(body)["error"] == "bad request"


def test_raw_delivery_carries_a_binary_body_unchanged() -> None:
    blob = bytes(range(256))
    with _running() as server:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = ReplicaEndpoint(base_url=base, model="m", interface="chat")
        asyncio.run(
            _drain_raw(
                endpoint,
                _serve_envelope(
                    path="v1/audio/transcriptions",
                    body=blob,
                    headers=[("content-type", "application/octet-stream")],
                ),
            )
        )
    assert server.seen[0]["body"] == blob


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
