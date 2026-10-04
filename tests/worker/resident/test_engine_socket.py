"""The replica sidecar reaches its engine over the engine's worker-private socket."""

import asyncio
import json
import shutil
import socketserver
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from shared.resident.contracts import LOCAL_ENGINE_BASE_URL, ReplicaEndpoint
from shared.resident.envelope import ServeRequestEnvelope
from worker.resident.engine import (
    HttpEngineDelivery,
    RawHttpEngineDelivery,
    unload_adapter,
)


class _Engine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[tuple[str, str, str | None]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        _Engine.seen.append(("POST", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/chat/completions":
            self._reply({"choices": [{"message": {"content": "over the socket"}}]})
        else:
            self._reply({})

    def do_GET(self) -> None:
        _Engine.seen.append(("GET", self.path, self.headers.get("Authorization")))
        chunks = [b"data: 1\n\n", b"data: [DONE]\n\n"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")

    def _reply(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


@contextmanager
def _engine_on_socket() -> Iterator[ReplicaEndpoint]:
    directory = Path(tempfile.mkdtemp())
    path = (directory / "engine.sock").as_posix()
    _Engine.seen = []
    server = socketserver.ThreadingUnixStreamServer(path, _Engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ReplicaEndpoint(
            base_url=LOCAL_ENGINE_BASE_URL,
            model="m",
            api_key="engine-key",
            socket_path=path,
        )
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(directory)


def test_a_completion_and_its_adapter_calls_reach_the_engine_socket() -> None:
    async def run(endpoint: ReplicaEndpoint) -> str:
        delivery = HttpEngineDelivery()
        try:
            response = await delivery(
                endpoint, '{"prompt": "hi"}', adapter_name="a", adapter_source="/a"
            )
            text = "".join([chunk async for chunk in response.chunks])
            await unload_adapter(endpoint, "a")
            return text
        finally:
            await delivery.aclose()

    with _engine_on_socket() as endpoint:
        assert asyncio.run(run(endpoint)) == "over the socket"
    assert _Engine.seen == [
        ("POST", "/v1/load_lora_adapter", "Bearer engine-key"),
        ("POST", "/v1/chat/completions", "Bearer engine-key"),
        ("POST", "/v1/unload_lora_adapter", "Bearer engine-key"),
    ]


def test_a_task_addressed_request_streams_from_the_engine_socket() -> None:
    async def run(endpoint: ReplicaEndpoint) -> bytes:
        response = await RawHttpEngineDelivery()(
            endpoint, ServeRequestEnvelope(method="GET", path="/v1/stream")
        )
        try:
            return b"".join([chunk async for chunk in response.chunks])
        finally:
            await response.aclose()

    with _engine_on_socket() as endpoint:
        assert asyncio.run(run(endpoint)) == b"data: 1\n\ndata: [DONE]\n\n"
    assert _Engine.seen == [("GET", "/v1/stream", "Bearer engine-key")]
