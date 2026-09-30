"""Tests for n8n workflow translation."""

import json
import logging
import os
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from server.config import N8nConfig, OrchestrationConfig
from server.task.n8n_parser import _decode_secret_part, translate_n8n_workflow
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.result_store import make_result_reader
from tests.server.task.test_runtime_rehydrate import (
    FakeWorkflowRegistry,
    _WorkerRegistryStub,
)


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

    def test_a_credentialed_node_carries_its_key_only_in_the_header(self) -> None:
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


_AES_PASSWORD = "p" * 32


def _encrypted(plaintext: str) -> str:
    nonce = os.urandom(12)
    sealed = AESGCM(_AES_PASSWORD.encode()).encrypt(nonce, plaintext.encode(), None)
    ciphertext, tag = sealed[:-16], sealed[-16:]
    return ":".join(part.hex() for part in (nonce, tag, ciphertext))


def _encrypted_openai_payload() -> str:
    node = {
        "name": "Chat",
        "type": "@n8n/n8n-nodes-langchain.openAi",
        "parameters": {
            "modelId": {"value": "gpt-4"},
            "responses": {"values": [{"content": "hi"}]},
        },
        "credentials": {
            "openAiApi": {
                "data": {"apiKey": _encrypted("sk-n8n"), "apiKey_encrypted": True}
            }
        },
    }
    return json.dumps({"nodes": [node], "connections": {}})


@pytest.mark.anyio
async def test_the_runtime_decrypts_n8n_credentials_with_its_configured_password():
    vault = InMemoryCredentialVault()
    runtime = TaskRuntime(
        cast(Any, FakeWorkflowRegistry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("n8n-test"),
        credential_vault=vault,
        n8n=N8nConfig(credential_password=_AES_PASSWORD),
    )
    workflow_id, results = await runtime.register(
        "owner", "org", _encrypted_openai_payload(), format="n8n"
    )
    record = runtime.get_record(results[0].task_id)
    assert record is not None and record.credential_refs is not None
    ref = record.credential_refs["/api/headers/Authorization"]
    assert vault.resolve_values(workflow_id, [ref]) == {ref: "Bearer sk-n8n"}


def test_n8n_config_reads_its_password_at_the_config_edge(monkeypatch):
    monkeypatch.setenv("N8N_CREDENTIAL_AES_PASSWORD", f" {_AES_PASSWORD} ")
    assert N8nConfig.from_env().credential_password == _AES_PASSWORD
