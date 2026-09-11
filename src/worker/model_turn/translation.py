"""Translation between the OpenAI Responses wire a harness speaks and Chat Completions.

A held facade drives a Codex turn over the Responses API but egresses to a Chat
Completions provider, so it maps a Responses request's input and tools into a chat
request and maps the chat reply back into Responses output items. The functions here are
pure: the facade owns the egress, the permit, and the capture; this only reshapes wire
formats. Tool schemas are authored in the Responses-flat shape and nest for chat.
"""

import json
from typing import Any

from shared.tools.model.schema import ModelCompletion, ModelToolCall

_ASSISTANT_MESSAGE_ID = "msg_fm"


def responses_input_to_messages(value: Any) -> list[dict[str, Any]]:
    """Map a Responses request ``input`` into chat ``messages``.

    A bare string is one user message. A list carries message items, the assistant's
    prior tool calls (``function_call``), and their results (``function_call_output``),
    each mapped to its chat-role equivalent so the conversation replays faithfully.
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            messages.extend(_item_to_messages(item))
    return messages


def _item_to_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    kind = item.get("type")
    if kind == "function_call":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(item.get("call_id", "")),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name", "")),
                            "arguments": str(item.get("arguments", "")),
                        },
                    }
                ],
            }
        ]
    if kind == "function_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": str(item.get("call_id", "")),
                "content": _text_of(item.get("output")),
            }
        ]
    if kind in (None, "message"):
        role = str(item.get("role", "user"))
        return [{"role": role, "content": _text_of(item.get("content"))}]
    return []


def _text_of(content: Any) -> str:
    """The plain text of a Responses content value, list of parts, or bare string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("input_text", "output_text", "text")
        ]
        return "".join(parts)
    return str(content)


# The harness-native tools the fabric forwards to the model. Everything else a harness
# advertises is dropped, including anything this set does not name: a tool that runs a
# command, delegates to a child, or blocks on a person must reach the model only as the
# fabric's own mediated facade, and a harness version that renames or adds one fails
# closed here rather than reaching the model unmediated. These forward because their
# effects stay inside the model's reasoning or the harness's own sealed session state.
HARNESS_TOOL_ALLOWLIST = frozenset(
    {"update_plan", "view_image", "get_goal", "create_goal", "update_goal"}
)


def chat_tools(responses_tools: Any, facade_schemas: list[str]) -> list[dict[str, Any]]:
    """The chat ``tools`` for a turn: the allowed harness tools plus the fabric facades.

    A Responses-flat function tool nests under a ``function`` key for chat; a
    non-function tool a harness advertises is dropped, a chat provider not accepting it.
    A harness tool outside :data:`HARNESS_TOOL_ALLOWLIST` is dropped too, so native code
    execution and native delegation never reach the model — the fabric's own facades are
    the only path to either. This holds for every agent, whether or not it declares a
    sandbox: an agent with no sandbox gets no executable tool at all.
    """
    tools: list[dict[str, Any]] = []
    if isinstance(responses_tools, list):
        for tool in responses_tools:
            if (nested := _nest_function_tool(tool)) is None:
                continue
            if nested["function"].get("name") not in HARNESS_TOOL_ALLOWLIST:
                continue
            tools.append(nested)
    for schema in facade_schemas:
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            continue
        if (nested := _nest_function_tool(parsed)) is not None:
            tools.append(nested)
    return tools


def _nest_function_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    if isinstance(tool.get("function"), dict):
        return {"type": "function", "function": tool["function"]}
    function = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
    return {"type": "function", "function": function}


def completion_to_responses_output(completion: ModelCompletion) -> list[dict[str, Any]]:
    """Map a chat completion message back into Responses output items.

    The assistant text becomes a message item; each tool call becomes a Responses
    ``function_call`` item at its own call id, so the harness sees the calls it emitted.
    """
    output: list[dict[str, Any]] = []
    if completion.content:
        output.append(message_output_item(completion.content))
    output.extend(function_call_item(call) for call in completion.tool_calls)
    return output


def function_call_item(call: ModelToolCall) -> dict[str, Any]:
    """One Responses ``function_call`` output item for a model's tool call."""
    return {
        "type": "function_call",
        "name": call.name,
        "arguments": call.arguments,
        "call_id": call.call_id,
    }


def message_output_item(text: str) -> dict[str, Any]:
    """One assistant message output item carrying ``text``."""
    return {
        "type": "message",
        "id": _ASSISTANT_MESSAGE_ID,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def responses_sse(output: list[dict[str, Any]]) -> bytes:
    """A minimal Responses SSE stream re-emitting a buffered turn's output items."""
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_fm"}}
    ]
    for index, item in enumerate(output):
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    events.append(
        {
            "type": "response.completed",
            "response": {"id": "resp_fm", "status": "completed", "output": output},
        }
    )
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )
