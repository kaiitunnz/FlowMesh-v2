"""The worker-local external-model egress surface."""

from typing import Any

import pytest
import requests

from shared.tools.contract import ToolOperationEnvelope, ToolOutcomeStatus
from shared.tools.model.egress import ExternalModelSidecar, ModelEgressError
from shared.tools.model.schema import MODEL_INTERFACE, ModelRequest

_REQUEST = ModelRequest(
    interface=MODEL_INTERFACE,
    url="http://up/v1",
    body={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
)


def _envelope() -> ToolOperationEnvelope:
    return ToolOperationEnvelope(
        interface=MODEL_INTERFACE,
        idempotency_key="idm-1",
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=6,
    )


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_success_returns_the_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return _Response({"choices": [{"message": {"content": "a completion body"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    out = ExternalModelSidecar().execute(_envelope(), _REQUEST, "sk-worker")
    assert out.status is ToolOutcomeStatus.SUCCESS
    # The response is capped at the envelope's char budget.
    assert out.value == "a comp"
    assert captured["url"] == "http://up/v1/chat/completions"
    assert captured["json"]["model"] == "m"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["headers"]["Authorization"] == "Bearer sk-worker"


def test_missing_key_sends_no_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        seen["headers"] = kwargs["headers"]
        return _Response({"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    ExternalModelSidecar().execute(_envelope(), _REQUEST, None)
    assert "Authorization" not in seen["headers"]


def test_timeout_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _Response:
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", fake_post)
    out = ExternalModelSidecar().execute(_envelope(), _REQUEST, "k")
    assert out.status is ToolOutcomeStatus.TIMEOUT


def test_unreachable_provider_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _Response:
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "post", fake_post)
    out = ExternalModelSidecar().execute(_envelope(), _REQUEST, "k")
    assert out.status is ToolOutcomeStatus.UNAVAILABLE


def test_interface_outside_envelope_is_unavailable() -> None:
    envelope = _envelope().model_copy(update={"interface": "search/v1"})
    out = ExternalModelSidecar().execute(envelope, _REQUEST, "k")
    assert out.status is ToolOutcomeStatus.UNAVAILABLE


def test_complete_returns_the_whole_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _Response:
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "thinking",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "x"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    out = ExternalModelSidecar().complete(_envelope(), _REQUEST, "k")
    # complete returns the whole message, tool calls included and uncapped.
    assert out.content == "thinking"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].call_id == "call_1"
    assert out.tool_calls[0].name == "web_search"
    assert out.tool_calls[0].arguments == '{"query": "x"}'


def test_complete_tolerates_a_tool_only_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _Response:
        return _Response({"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    out = ExternalModelSidecar().complete(_envelope(), _REQUEST, "k")
    assert out.content == "" and out.tool_calls == ()


def test_complete_raises_terminal_on_a_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _Response:
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(ModelEgressError):
        ExternalModelSidecar().complete(_envelope(), _REQUEST, "k")


def test_complete_interface_mismatch_raises() -> None:
    envelope = _envelope().model_copy(update={"interface": "search/v1"})
    with pytest.raises(ModelEgressError):
        ExternalModelSidecar().complete(envelope, _REQUEST, "k")
