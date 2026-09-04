"""The remote external-tool sidecar validates its operation fence before any egress.

Drives a loopback ``ExternalToolSidecarListener`` with one operation frame and asserts a
valid fence egresses once, while each tampered field is rejected with no provider call —
the fence failures the acceptance gate requires, proven at the enforcing surface.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from server.services import tool_sidecar_wire as wire
from server.services.external_tool_sidecar import (
    ExternalToolSidecarListener,
    ExternalToolSidecarServer,
)
from server.services.tool_egress import (
    ExternalToolSidecar,
    RemoteToolOperationEnvelope,
    ToolRequest,
    tool_request_digest,
)
from shared.tools.providers import SearchResult, SearchUnavailable
from shared.utils.ids import new_tool_delivery_nonce

TARGET_ID = "stg-1"
TARGET_GEN = 4
PROVIDER = "fake"
QUERY = "what is flowmesh"


class _FakeProvider:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]:
        self.calls.append(query)
        if self._fail:
            raise SearchUnavailable("down")
        return [SearchResult(title=f"r:{query}", url="http://x", snippet="s")]


def _envelope(**overrides: object) -> RemoteToolOperationEnvelope:
    base: dict[str, object] = dict(
        interface="search/v1",
        provider=PROVIDER,
        idempotency_key="idm-abc",
        request_digest=tool_request_digest("search/v1", QUERY, 3),
        target_id=TARGET_ID,
        target_generation=TARGET_GEN,
        delivery_nonce=new_tool_delivery_nonce(),
        deadline_epoch=time.time() + 30,
        max_results=3,
        timeout_sec=5.0,
        result_char_cap=6000,
    )
    base.update(overrides)
    return RemoteToolOperationEnvelope(**base)  # type: ignore[arg-type]


async def _exchange(
    route: str, envelope: RemoteToolOperationEnvelope, request: ToolRequest
) -> dict[str, object]:
    host, port = wire.split_host_port(route)
    reader, writer = await asyncio.open_connection(host, port)
    try:
        await wire.write_msg(
            writer,
            wire.KIND_OPERATION,
            envelope=envelope.model_dump(mode="json"),
            request=request.model_dump(mode="json"),
        )
        return await wire.read_msg(reader)
    finally:
        writer.close()
        await writer.wait_closed()


def _run(
    body: Callable[[str, _FakeProvider], Awaitable[None]], *, fail: bool = False
) -> None:
    async def wrapped() -> None:
        provider = _FakeProvider(fail=fail)
        listener = ExternalToolSidecarListener(
            ExternalToolSidecarServer(
                sidecar=ExternalToolSidecar(provider),
                target_id=TARGET_ID,
                target_generation=TARGET_GEN,
                provider=PROVIDER,
                interfaces=frozenset({"search/v1"}),
            ),
            route="127.0.0.1:0",
        )
        host, port = await listener.start()
        try:
            await body(f"{host}:{port}", provider)
        finally:
            await listener.stop()

    asyncio.run(wrapped())


def test_valid_fence_egresses_once() -> None:
    async def body(route: str, provider: _FakeProvider) -> None:
        req = ToolRequest(interface="search/v1", query=QUERY, max_results=3)
        msg = await _exchange(route, _envelope(), req)
        assert msg["kind"] == wire.KIND_RESULT
        assert msg["outcome"]["status"] == "success"  # type: ignore[index]
        assert provider.calls == [QUERY]

    _run(body)


def test_each_tampered_fence_is_rejected_without_egress() -> None:
    tampers = {
        "provider": _envelope(provider="other"),
        "audience": _envelope(target_id="stg-other"),
        "generation": _envelope(target_generation=TARGET_GEN + 1),
        "policy": _envelope(policy_class="other"),
        "expired": _envelope(deadline_epoch=time.time() - 1),
        "digest": _envelope(request_digest="deadbeef"),
        "interface": _envelope(interface="unknown/v1"),
    }

    async def body(route: str, provider: _FakeProvider) -> None:
        req = ToolRequest(interface="search/v1", query=QUERY, max_results=3)
        for name, env in tampers.items():
            msg = await _exchange(route, env, req)
            assert msg["kind"] == wire.KIND_REJECT, name
        # An over-budget request within a valid envelope is also rejected pre-egress.
        big = ToolRequest(interface="search/v1", query=QUERY, max_results=99)
        over = _envelope(request_digest=tool_request_digest("search/v1", QUERY, 99))
        msg = await _exchange(route, over, big)
        assert msg["kind"] == wire.KIND_REJECT
        assert provider.calls == []

    _run(body)


def test_provider_fault_maps_to_typed_outcome() -> None:
    async def body(route: str, provider: _FakeProvider) -> None:
        req = ToolRequest(interface="search/v1", query=QUERY, max_results=3)
        msg = await _exchange(route, _envelope(), req)
        assert msg["kind"] == wire.KIND_RESULT
        assert msg["outcome"]["status"] == "unavailable"  # type: ignore[index]
        assert provider.calls == [QUERY]

    _run(body, fail=True)


def test_exact_replay_of_one_authorization_is_rejected_before_egress() -> None:
    async def body(route: str, provider: _FakeProvider) -> None:
        req = ToolRequest(interface="search/v1", query=QUERY, max_results=3)
        # One authorized delivery egresses; replaying that exact nonce is refused with
        # no second provider call, while a fresh nonce under the same idm egresses.
        fixed = _envelope()
        first = await _exchange(route, fixed, req)
        assert first["kind"] == wire.KIND_RESULT
        replay = await _exchange(route, fixed, req)
        assert replay["kind"] == wire.KIND_REJECT
        assert replay["reason"] == "replay"
        fresh = await _exchange(route, _envelope(), req)
        assert fresh["kind"] == wire.KIND_RESULT
        assert provider.calls == [QUERY, QUERY]

    _run(body)
