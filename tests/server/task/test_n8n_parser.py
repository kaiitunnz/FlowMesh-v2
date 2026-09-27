"""Tests for n8n workflow translation."""

import pytest

from server.task.n8n_parser import _decode_secret_part, translate_n8n_workflow
from server.task.parser import parse_workflow


class TestTranslateN8nWorkflow:
    def test_simple_openai_node(self) -> None:
        """A single OpenAI chat node should produce an API task with correct fields."""
        nodes = [
            {
                "name": "Chat",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {
                        "values": [{"content": "Hello, world!"}],
                    },
                },
            }
        ]
        result = translate_n8n_workflow({"nodes": nodes, "connections": {}})

        # Top-level shape
        assert result["kind"] == "APITask"
        assert result["apiVersion"] == "flowmesh/v1"
        assert "spec" in result

        # Task type and API spec
        spec = result["spec"]
        assert spec["taskType"] == "api"
        assert "api" in spec
        api = spec["api"]
        assert api["method"] == "POST"
        assert api["body"]["model"] == "gpt-4"

        # Prompt content preserved in messages
        messages = api["body"]["messages"]
        assert any("Hello, world!" in m.get("content", "") for m in messages)

    def test_no_task_nodes_raises_value_error(self) -> None:
        """Workflow with no recognized task nodes should raise ValueError."""
        with pytest.raises(ValueError, match="No task nodes found"):
            translate_n8n_workflow({"nodes": [], "connections": {}})

    def test_invalid_json_via_parse_workflow(self) -> None:
        """Non-JSON input to n8n format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_workflow("not json at all {{{", format="n8n")


class TestDecodeSecretPart:
    def test_hex_decode(self) -> None:
        data = b"hello"
        encoded = data.hex()
        assert _decode_secret_part(encoded) == data

    def test_base64_decode(self) -> None:
        import base64

        data = b"hello world"
        encoded = base64.b64encode(data).decode()
        assert _decode_secret_part(encoded) == data

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(Exception):
            _decode_secret_part("!!!not-valid-hex-or-base64!!!")


def test_a_credentialed_node_carries_its_key_only_in_the_header() -> None:
    nodes = [
        {
            "name": "Chat",
            "type": "@n8n/n8n-nodes-langchain.openAi",
            "parameters": {
                "modelId": {"value": "gpt-4"},
                "responses": {"values": [{"content": "hi"}]},
            },
            "credentials": {"openAiApi": {"data": {"apiKey": "sk-n8n"}}},
        }
    ]
    api = translate_n8n_workflow({"nodes": nodes, "connections": {}})["spec"]["api"]
    assert api["headers"]["Authorization"] == "Bearer sk-n8n"
    assert "key" not in api
