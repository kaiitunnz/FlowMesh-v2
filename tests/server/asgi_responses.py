"""Drive a Starlette response through ASGI and capture what it sends."""

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import anyio
from starlette.responses import Response


@dataclass
class Served:
    status: int
    headers: dict[str, str]
    body: bytes


class ClientGone(OSError):
    """The client disconnected while the body was being sent."""


async def serve(
    response: Response,
    headers: dict[str, str] | None = None,
    abort_after_chunks: int | None = None,
) -> Served:
    """Send ``response`` for a GET carrying ``headers``; with ``abort_after_chunks``
    the client disconnects once that many body chunks have arrived."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or {}).items()
        ],
    }
    sent: dict[str, Any] = {"chunks": []}
    requested = False
    gone = anyio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
            sent["headers"] = {
                name.decode("latin-1"): value.decode("latin-1")
                for name, value in message["headers"]
            }
        elif message["type"] == "http.response.body":
            if (
                abort_after_chunks is not None
                and len(sent["chunks"]) >= abort_after_chunks
            ):
                gone.set()
                raise ClientGone("client disconnected")
            sent["chunks"].append(message.get("body", b""))

    try:
        await response(scope, receive, send)
    finally:
        gone.set()
    return Served(sent["status"], sent["headers"], b"".join(sent["chunks"]))
