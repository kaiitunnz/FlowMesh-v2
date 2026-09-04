"""Unit coverage for the supervisor's worker tool-route deputy.

The deputy is a pure opaque router: it forwards each operation frame to the target
worker verbatim (never decoding the envelope, request, or outcome), correlates the reply
only by a transport session id, writes the opaque reply back, and pops the session
on every terminal. It constructs no provider and reads no credential.
"""

import ast
import asyncio
import base64
import inspect
from typing import Any

import pytest

from server.network import wire as netwire
from server.supervisor.services import tool_egress_deputy as deputy_mod
from server.supervisor.services.tool_egress_deputy import WorkerToolRouteDeputy
from shared.tools.wire import FRAME_CANCEL, FRAME_OPERATION, FRAME_REPLY


class _FakeTaskListener:
    def __init__(self, attached: bool = True) -> None:
        self.calls: list[tuple[str, str, str, bytes]] = []
        self.attached = attached
        self.ev = asyncio.Event()

    def enqueue_egress(
        self, worker_id: str, session_id: str, egress_kind: str, payload: bytes
    ) -> bool:
        self.calls.append((worker_id, session_id, egress_kind, payload))
        self.ev.set()
        return self.attached


def _deputy(tl: _FakeTaskListener, **kw: Any) -> WorkerToolRouteDeputy:
    return WorkerToolRouteDeputy(task_listener=tl, **kw)  # type: ignore[arg-type]


async def _bind(deputy: WorkerToolRouteDeputy) -> tuple[str, int]:
    data = await deputy.bind_sidecar(
        {"target_id": "wrk-1", "worker_id": "wrk-1", "route": "127.0.0.1:0"}
    )
    return str(data["host"]), int(data["port"])


def test_forwards_the_frame_verbatim_and_writes_the_reply_back() -> None:
    async def run() -> None:
        tl = _FakeTaskListener()
        deputy = _deputy(tl)
        host, port = await _bind(deputy)
        reader, writer = await asyncio.open_connection(host, port)
        op = b"OPAQUE-OPERATION-FRAME"
        await netwire.write_frame(writer, op)
        await asyncio.wait_for(tl.ev.wait(), 2.0)
        worker_id, session_id, kind, payload = tl.calls[0]
        # Forwarded to the worker's attachment as opaque bytes, verbatim.
        assert (worker_id, kind, payload) == ("wrk-1", FRAME_OPERATION, op)
        reply = b"OPAQUE-REPLY-FRAME"
        deputy.deliver_up(session_id, FRAME_REPLY, base64.b64encode(reply).decode())
        assert await asyncio.wait_for(netwire.read_frame(reader), 2.0) == reply
        # The session is popped on the clean terminal, not only on cancel.
        assert deputy._sessions == {}
        writer.close()
        await deputy.stop()

    asyncio.run(run())


def test_deliver_up_drops_an_unknown_session() -> None:
    async def run() -> None:
        deputy = _deputy(_FakeTaskListener())
        await _bind(deputy)
        deputy.deliver_up("xtr-nope", FRAME_REPLY, base64.b64encode(b"x").decode())
        await asyncio.sleep(0.05)
        assert deputy._sessions == {}
        await deputy.stop()

    asyncio.run(run())


def test_a_lost_reply_forwards_a_cancel_and_retains_then_reaps() -> None:
    async def run() -> None:
        tl = _FakeTaskListener()
        deputy = _deputy(tl, recv_timeout_sec=0.05, reap_ttl_sec=0.05)
        host, port = await _bind(deputy)
        reader, writer = await asyncio.open_connection(host, port)
        await netwire.write_frame(writer, b"op")
        await asyncio.wait_for(tl.ev.wait(), 2.0)
        # The reply never comes: the deputy forwards a cancel to the worker, keeps the
        # record until the bounded reaper fires, then pops it.
        await asyncio.sleep(0.2)
        assert tl.calls[-1][2] == FRAME_CANCEL
        assert deputy._sessions == {}
        writer.close()
        await deputy.stop()

    asyncio.run(run())


def test_a_failed_write_back_to_the_origin_pops_the_session() -> None:
    async def run() -> None:
        tl = _FakeTaskListener()
        deputy = _deputy(tl)
        deputy._loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        op = b"OPAQUE-OPERATION-FRAME"
        reader.feed_data(len(op).to_bytes(4, "big") + op)

        class _BoomWriter:
            def write(self, data: bytes) -> None:
                raise ConnectionResetError("origin gone")

            async def drain(self) -> None:
                pass

        task = asyncio.create_task(
            deputy.serve_operation("wrk-1", reader, _BoomWriter())  # type: ignore[arg-type]
        )
        await asyncio.wait_for(tl.ev.wait(), 2.0)
        session_id = tl.calls[0][1]
        # The worker's reply arrives, but writing it back to the origin fails.
        deputy.deliver_up(session_id, FRAME_REPLY, base64.b64encode(b"reply").decode())
        with pytest.raises(ConnectionResetError):
            await asyncio.wait_for(task, 2.0)
        # The failed write-back did not leak the session record.
        assert deputy._sessions == {}
        await deputy.stop()

    asyncio.run(run())


def test_the_supervisor_deputy_constructs_no_provider() -> None:
    # The invariant: the supervisor routes opaque frames and never egresses. From the
    # shared tool package it may import only the opaque wire codec (the FRAME_* frame
    # kinds) — never a provider, egress surface, envelope decoder, or credential, under
    # either the old or the relocated shared.tools.* paths.
    src = inspect.getsource(deputy_mod)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "shared.tools"
        ):
            assert node.module == "shared.tools.wire", (
                f"supervisor deputy may import from shared.tools only the wire codec, "
                f"not {node.module}"
            )
            for alias in node.names:
                assert alias.name.startswith("FRAME_"), (
                    f"supervisor deputy may import only FRAME_* from the wire codec, "
                    f"not {alias.name}"
                )
    for forbidden in (
        "ExternalToolSidecar",
        "search_providers",
        "shared.tools.search",
        "WebSearchConfig",
        "LazySearchProvider",
        "build_search_provider",
        "SearchProvider",
        "decode_msg",
        "encode_msg",
        "requests",
        "httpx",
    ):
        assert forbidden not in src, f"supervisor deputy must not reference {forbidden}"
    for attr in (
        "ExternalToolSidecar",
        "LazySearchProvider",
        "build_search_provider",
        "SearchProvider",
        "requests",
    ):
        assert not hasattr(deputy_mod, attr)
