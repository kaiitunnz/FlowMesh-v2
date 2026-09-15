"""Building an engine request from a boundary payload.

A payload carries either one conversation or a batch of them, and the two are built by
different callers. A shape read as the wrong one would issue a request the leaf never
declared, so the single-conversation builder refuses a batch rather than flattening it.
"""

import json

import pytest

from shared.resident.engine_request import (
    batch_chat_bodies,
    chat_body,
    is_batch_request,
)

_BATCH = json.dumps(
    [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    ]
)


class TestBatchShape:
    def test_a_list_of_chat_requests_is_a_batch(self) -> None:
        assert is_batch_request(_BATCH)
        bodies = batch_chat_bodies(_BATCH, "m")
        assert bodies is not None
        assert [b["model"] for b in bodies] == ["m", "m"]

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "",
            "hello",
            json.dumps({"messages": [{"role": "user", "content": "a"}]}),
            json.dumps([]),
            json.dumps(["a", "b"]),
            json.dumps([{"prompt": "a"}]),
        ],
    )
    def test_anything_else_carries_one_conversation(self, payload: str | None) -> None:
        assert not is_batch_request(payload)
        assert batch_chat_bodies(payload, "m") is None


class TestSingleConversation:
    def test_a_batch_payload_is_refused_rather_than_flattened(self) -> None:
        # Stuffing the array's text into one user message would quietly generate one
        # completion for a leaf that declared several.
        with pytest.raises(ValueError, match="batch_chat_bodies"):
            chat_body(_BATCH, "m")

    def test_a_chat_request_keeps_its_messages_and_pins_the_model(self) -> None:
        body = chat_body(
            json.dumps({"messages": [{"role": "user", "content": "a"}]}), "m"
        )
        assert body == {"messages": [{"role": "user", "content": "a"}], "model": "m"}

    def test_a_bare_prompt_becomes_one_user_message(self) -> None:
        assert chat_body("hello", "m") == {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
