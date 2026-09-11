"""Responses <-> Chat translation the held facade drives a Codex turn through."""

import json

from shared.tools.model.schema import ModelCompletion, ModelToolCall
from worker.model_turn.translation import (
    chat_tools,
    completion_to_responses_output,
    responses_input_to_messages,
    responses_sse,
)

_SEARCH_TOOL = json.dumps(
    {
        "type": "function",
        "name": "web_search",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
)


def test_string_input_is_one_user_message() -> None:
    assert responses_input_to_messages("hello") == [
        {"role": "user", "content": "hello"}
    ]


def test_conversation_items_replay_faithfully() -> None:
    value = [
        {"type": "message", "role": "user", "content": "find the weather"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": '{"query": "weather"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "it is sunny"}],
        },
    ]
    messages = responses_input_to_messages(value)
    assert messages[0] == {"role": "user", "content": "find the weather"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "sunny",
    }
    assert messages[3] == {"role": "assistant", "content": "it is sunny"}


def test_chat_tools_nest_and_inject() -> None:
    allowed = {
        "type": "function",
        "name": "update_plan",
        "parameters": {"type": "object"},
    }
    other = {"type": "web_search"}  # a non-function harness tool chat providers reject
    tools = chat_tools([allowed, other], [_SEARCH_TOOL])
    assert len(tools) == 2  # the allowed harness tool and the injected facade
    assert all(t["type"] == "function" and "function" in t for t in tools)
    names = {t["function"]["name"] for t in tools}
    assert names == {"update_plan", "web_search"}


def test_a_native_code_execution_tool_never_reaches_the_model() -> None:
    """Code runs through the fabric's fenced runtime or not at all."""
    native_exec = [
        {"type": "function", "name": name, "parameters": {"type": "object"}}
        for name in ("exec_command", "write_stdin", "shell_command")
    ]

    tools = chat_tools(native_exec, [_SEARCH_TOOL])

    assert {t["function"]["name"] for t in tools} == {"web_search"}


def test_a_native_delegation_tool_never_reaches_the_model() -> None:
    """Child agents come from the fabric's spawn facade, never a native subagent."""
    tools = chat_tools(
        [{"type": "function", "name": "multi_agent_v1", "parameters": {}}], []
    )

    assert tools == []


def test_an_unrecognized_harness_tool_fails_closed() -> None:
    """A renamed or new harness tool drops rather than reaching the model unmediated."""
    tools = chat_tools(
        [{"type": "function", "name": "exec_command_v2", "parameters": {}}], []
    )

    assert tools == []


def test_completion_maps_to_message_and_function_calls() -> None:
    completion = ModelCompletion(
        content="thinking",
        tool_calls=(
            ModelToolCall(call_id="c1", name="web_search", arguments='{"query": "x"}'),
        ),
    )
    output = completion_to_responses_output(completion)
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "thinking"
    assert output[1] == {
        "type": "function_call",
        "name": "web_search",
        "arguments": '{"query": "x"}',
        "call_id": "c1",
    }


def test_tool_only_completion_emits_no_message_item() -> None:
    completion = ModelCompletion(
        content="",
        tool_calls=(ModelToolCall(call_id="c1", name="spawn_agent", arguments="{}"),),
    )
    output = completion_to_responses_output(completion)
    assert [item["type"] for item in output] == ["function_call"]


def test_sse_frames_the_turn() -> None:
    sse = responses_sse([{"type": "message", "id": "m"}]).decode()
    assert "response.created" in sse
    assert "response.output_item.done" in sse
    assert "response.completed" in sse
