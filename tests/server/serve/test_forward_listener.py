"""The root forward listener reads a request, admits it, and streams the response.

It binds one plain-HTTP listener per reserved port, resolves the serve task from the
arrival port, freezes the request, hands it to the admission callback, and writes the
engine's head and body back to the client socket. A denial maps to that HTTP status; a
request on a port bound to no task is refused; a request on a released port is closed.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from server.serve.forward_listener import RootForwardIngress, ServeForwardDenied


@dataclass
class _Event:
    kind: str
    payload: bytes = b""
    detail: str | None = None
    status: int = 200
    headers: tuple[tuple[str, str], ...] = ()

    @property
    def terminal(self) -> bool:
        return self.kind in ("done", "error")


@dataclass
class _FakeResult:
    events_list: list[_Event]
    closed: bool = field(default=False)

    async def events(self) -> AsyncIterator[_Event]:
        for ev in self.events_list:
            yield ev

    def close_client(self) -> None:
        self.closed = True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _request(port: int, method: str = "GET", target: str = "/v1/models") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"{method} {target} HTTP/1.1\r\nHost: x\r\n"
        "Authorization: Bearer k\r\nContent-Length: 0\r\n\r\n".encode()
    )
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data


def test_listener_admits_and_streams_the_engine_head_and_body() -> None:
    async def run() -> None:
        result = _FakeResult(
            [
                _Event(
                    "head", status=200, headers=(("content-type", "application/json"),)
                ),
                _Event("chunk", payload=b"hello"),
                _Event("done"),
            ]
        )

        async def _admit(cred, task_id, envelope):
            assert cred == "Bearer k"
            assert task_id == "tsk-1"
            assert envelope.path == "/v1/models"
            return result

        listener = RootForwardIngress(
            bind_host="127.0.0.1", authority="local", admit=_admit, on_bound=_noop
        )
        listener.start(asyncio.get_running_loop())
        port = _free_port()
        await listener._bind_and_report("tsk-1", 0, port)

        raw = await _request(port)
        assert b"200 OK" in raw
        assert b"content-type: application/json" in raw.lower()
        assert b"Transfer-Encoding: chunked" in raw
        assert b"hello" in raw
        assert raw.endswith(b"0\r\n\r\n")
        await listener.stop()

    asyncio.run(run())


def test_listener_maps_a_denial_to_its_http_status() -> None:
    async def run() -> None:
        async def _admit(cred, task_id, envelope):
            raise ServeForwardDenied(403, "denied")

        listener = RootForwardIngress(
            bind_host="127.0.0.1", authority="local", admit=_admit, on_bound=_noop
        )
        listener.start(asyncio.get_running_loop())
        port = _free_port()
        await listener._bind_and_report("tsk-1", 0, port)

        raw = await _request(port)
        assert b"403" in raw.split(b"\r\n")[0]
        await listener.stop()

    asyncio.run(run())


def test_listener_refuses_a_body_before_a_head_without_synthesizing_200() -> None:
    async def run() -> None:
        # A terminal (or body) with no engine head is a bad-gateway failure, never a
        # synthetic 200 over a response that never produced a head.
        result = _FakeResult([_Event("error", detail="no head")])

        async def _admit(cred, task_id, envelope):
            return result

        listener = RootForwardIngress(
            bind_host="127.0.0.1", authority="local", admit=_admit, on_bound=_noop
        )
        listener.start(asyncio.get_running_loop())
        port = _free_port()
        await listener._bind_and_report("tsk-1", 0, port)

        raw = await _request(port)
        assert b"502" in raw.split(b"\r\n")[0]
        await listener.stop()

    asyncio.run(run())


def test_listener_refuses_a_request_on_a_released_port() -> None:
    async def run() -> None:
        async def _admit(cred, task_id, envelope):
            raise AssertionError("admit must not run for a released port")

        listener = RootForwardIngress(
            bind_host="127.0.0.1", authority="local", admit=_admit, on_bound=_noop
        )
        listener.start(asyncio.get_running_loop())
        port = _free_port()
        await listener._bind_and_report("tsk-1", 0, port)
        # Drop the port→task mapping but keep the socket bound, as a drain does before
        # the listener closes it.
        listener._port_to_task.pop(port, None)

        raw = await _request(port)
        assert b"404" in raw.split(b"\r\n")[0]
        await listener.stop()

    asyncio.run(run())


def _noop(_task_id: str, _exposure_generation: int, _listener_generation: int) -> None:
    pass
