import json

import yaml
from pydantic import SecretStr

from server.task.parser import parse_workflow
from server.task.v2.credentials import pop_inline_model_secrets, redact_source_text
from shared.utils.redact import REDACTED

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: agent
          harness: {backend: scripted, params: {script: []}}
          model_binding:
            mode: openai
            url: "https://h/v1"
            model: m
            api_key: "sk-secret"
"""


def test_pop_inline_model_secrets_strips_and_returns_the_key():
    parsed = parse_workflow(_WF, "native")
    secrets = pop_inline_model_secrets(parsed)
    assert len(secrets) == 1
    secret = next(iter(secrets.values()))
    assert isinstance(secret, SecretStr)
    assert secret.get_secret_value() == "sk-secret"
    # The key is stripped from the parsed spec in place.
    agent = next(t for t in parsed.tasks if t.task.spec.taskType == "agent")
    assert agent.task.spec.model_binding.api_key is None


def test_redact_source_text_masks_the_inline_key():
    redacted = redact_source_text(_WF, "native")
    assert "sk-secret" not in redacted
    assert REDACTED in redacted


def test_redaction_survives_alternate_quoting_and_escaping():
    # Single-quoted here; masking is structural, not a literal string replace.
    wf = _WF.replace('"sk-secret"', "'sk-secret'")
    assert "sk-secret" not in redact_source_text(wf, "native")


def test_a_source_without_a_credential_keeps_its_content():
    wf = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
"""
    assert yaml.safe_load(redact_source_text(wf, "native")) == yaml.safe_load(wf)


def test_a_duplicate_or_commented_out_key_leaves_no_credential():
    wf = """
spec:
  # api_key: sk-commented
  api_key: sk-duplicate
  api_key: null
"""
    redacted = redact_source_text(wf, "native")
    assert "sk-commented" not in redacted and "sk-duplicate" not in redacted


def test_pop_is_empty_without_an_agent_credential():
    wf = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
"""
    assert pop_inline_model_secrets(parse_workflow(wf, "native")) == {}


_HEADER_WF = """
apiVersion: flowmesh/v1
kind: Task
metadata: {name: t}
spec:
  taskType: api
  api:
    url: "https://h/v1"
    headers: {Authorization: "Bearer sk-header", X-API-Key: sk-xkey}
    json: {model: m, max_tokens: 8, cache_key: c}
"""


def test_redaction_masks_every_credential_key_and_keeps_ordinary_ones():
    redacted = redact_source_text(_HEADER_WF, "native")
    assert "sk-header" not in redacted and "sk-xkey" not in redacted
    doc = yaml.safe_load(redacted)
    assert doc["spec"]["api"]["json"] == {
        "model": "m",
        "max_tokens": 8,
        "cache_key": "c",
    }


_N8N = {
    "nodes": [
        {
            "name": "Chat",
            "type": "@n8n/n8n-nodes-langchain.openAi",
            "parameters": {
                "modelId": {"value": "gpt-4"},
                "responses": {"values": [{"content": "hi"}]},
            },
            "credentials": {"openAiApi": {"data": {"apiKey": "sk-n8n"}}},
        }
    ],
    "connections": {},
}


def test_an_n8n_source_is_redacted_as_json_even_when_tab_indented():
    payload = json.dumps(_N8N, indent="\t")
    parse_workflow(payload, "n8n")
    redacted = redact_source_text(payload, "n8n")
    assert "sk-n8n" not in redacted
    assert (
        json.loads(redacted)["nodes"][0]["parameters"] == _N8N["nodes"][0]["parameters"]
    )


def test_an_unparseable_source_persists_no_text():
    assert redact_source_text("{not: [valid", "native") == REDACTED
    assert redact_source_text("not json sk-raw", "n8n") == REDACTED


def test_an_n8n_header_array_is_masked():
    payload = json.dumps(
        {
            "nodes": [
                {
                    "name": "Fetch",
                    "type": "n8n-nodes-base.httpRequest",
                    "parameters": {
                        "headerParameters": {
                            "parameters": [
                                {"name": "Authorization", "value": "Bearer sk-hdr"}
                            ]
                        },
                        "queryParameters": {
                            "parameters": [{"name": "api_key", "value": "sk-query"}]
                        },
                    },
                }
            ],
            "connections": {},
        }
    )
    redacted = redact_source_text(payload, "n8n")
    assert "sk-hdr" not in redacted and "sk-query" not in redacted
