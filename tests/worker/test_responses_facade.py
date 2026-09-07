"""The worker-local Responses facade's held-turn handling and capture."""

import json
from typing import Any, cast

import httpx
import pytest

from shared.harness import BoundaryEventKind
from shared.tools.facade import FacadeDescriptor
from shared.tools.model.schema import ModelCompletion, ModelToolCall
from shared.tools.search.schema import SEARCH_INTERFACE
from worker.egress import HeldEgressReject
from worker.lifecycle import PendingEgressRequestStore
from worker.model_turn import FacadeTurnError, ResponsesFacade

_TASK = "tsk-agent"
_SEARCH = FacadeDescriptor(
    name="web_search",
    kind=BoundaryEventKind.INVOCATION,
    interface=SEARCH_INTERFACE,
    tool_schema=json.dumps(
        {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
    ),
)


class _StubEgress:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: list[tuple[str, str, Any]] = []

    def run(self, task_id: str, correlation: str, request: Any) -> Any:
        self.seen.append((task_id, correlation, request))
        return self._result


def _facade(
    result: Any,
) -> tuple[ResponsesFacade, _StubEgress, PendingEgressRequestStore]:
    egress = _StubEgress(result)
    pending = PendingEgressRequestStore()
    facade = ResponsesFacade(held_egress=cast(Any, egress), pending=pending)
    return facade, egress, pending


def test_plain_turn_returns_the_reply_and_captures_no_group() -> None:
    facade, egress, _ = _facade(ModelCompletion(content="just thinking"))
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    output = facade.handle_turn(_TASK, token, {"input": "hello"})
    assert [item["type"] for item in output] == ["message"]
    assert output[0]["content"][0]["text"] == "just thinking"
    assert facade.take_captured_group(_TASK) is None
    # The egress ran on the translated chat body with the injected facade tool.
    _, correlation, request = egress.seen[0]
    assert correlation == "model:0"
    assert request.body["messages"] == [{"role": "user", "content": "hello"}]
    assert request.body["tools"][0]["function"]["name"] == "web_search"


def test_facade_call_is_captured_and_the_turn_is_cleaned() -> None:
    completion = ModelCompletion(
        content="searching",
        tool_calls=(
            ModelToolCall(
                call_id="c1", name="web_search", arguments='{"query": "weather"}'
            ),
        ),
    )
    facade, _, pending = _facade(completion)
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    output = facade.handle_turn(_TASK, token, {"input": "find the weather"})

    # The group is captured for the completion to carry; the search request is kept in
    # worker custody.
    group = facade.take_captured_group(_TASK)
    assert group is not None and group.group_id == f"{_TASK}:0"
    assert pending.peek(_TASK, f"{_TASK}:0:0") is not None
    # The turn is cleaned: the raw facade call never returns to codex, only a summary.
    texts = [item["content"][0]["text"] for item in output if item["type"] == "message"]
    assert texts[0] == "searching"
    assert "web search" in texts[-1]


def test_a_native_call_co_emitted_with_a_facade_call_is_preserved() -> None:
    completion = ModelCompletion(
        content="",
        tool_calls=(
            ModelToolCall(call_id="s1", name="web_search", arguments='{"query": "x"}'),
            ModelToolCall(call_id="n1", name="native_fn", arguments="{}"),
        ),
    )
    facade, _, _ = _facade(completion)
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    output = facade.handle_turn(_TASK, token, {"input": "hi"})
    # The facade call is captured; the co-emitted native call stays in the turn so the
    # harness runs it, and a dispatch summary follows.
    assert facade.take_captured_group(_TASK) is not None
    calls = [item for item in output if item["type"] == "function_call"]
    assert [c["call_id"] for c in calls] == ["n1"]
    assert any(item["type"] == "message" for item in output)


def test_unknown_episode_or_bad_token_is_a_turn_error() -> None:
    facade, _, _ = _facade(ModelCompletion(content="x"))
    with pytest.raises(FacadeTurnError):
        facade.handle_turn("nope", "t", {"input": "hi"})
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    with pytest.raises(FacadeTurnError):
        facade.handle_turn(_TASK, token + "x", {"input": "hi"})


def test_a_rejected_egress_fails_the_turn() -> None:
    facade, _, _ = _facade(HeldEgressReject(reason="permit fence rejected: digest"))
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    with pytest.raises(FacadeTurnError):
        facade.handle_turn(_TASK, token, {"input": "hi"})


def test_http_server_serves_a_turn_over_loopback() -> None:
    facade, _, _ = _facade(ModelCompletion(content="hi there"))
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    facade.start()
    try:
        base = facade.base_url()
        assert base.startswith("http://127.0.0.1:")  # loopback only
        url = f"{base}/agent/{_TASK}/v1/responses"
        ok = httpx.post(
            url, json={"input": "hello"}, headers={"Authorization": f"Bearer {token}"}
        )
        assert ok.status_code == 200 and "response.completed" in ok.text
        # A wrong token cannot drive the episode's egress.
        bad = httpx.post(
            url, json={"input": "hi"}, headers={"Authorization": "Bearer wrong"}
        )
        assert bad.status_code == 502
    finally:
        facade.stop()


def test_a_native_tool_call_passes_through_uncaptured() -> None:
    completion = ModelCompletion(
        content="",
        tool_calls=(ModelToolCall(call_id="c1", name="native_fn", arguments="{}"),),
    )
    facade, _, _ = _facade(completion)
    token = facade.register_episode(_TASK, "http://up/v1", "m", [_SEARCH])
    output = facade.handle_turn(_TASK, token, {"input": "hi"})
    assert (
        facade.take_captured_group(_TASK) is None
    )  # a non-facade call is not captured
    assert [item["type"] for item in output] == ["function_call"]
