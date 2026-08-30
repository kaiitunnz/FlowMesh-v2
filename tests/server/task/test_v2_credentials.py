from pydantic import SecretStr

from server.task.parser import parse_workflow
from server.task.v2.credentials import pop_inline_model_secrets, redact_source_text

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
    assert "***redacted***" in redacted


def test_redaction_survives_alternate_quoting_and_escaping():
    # Single-quoted here; masking is structural, not a literal string replace.
    wf = _WF.replace('"sk-secret"', "'sk-secret'")
    assert "sk-secret" not in redact_source_text(wf, "native")


def test_redaction_is_a_no_op_without_an_inline_key():
    wf = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
"""
    assert redact_source_text(wf, "native") == wf


def test_pop_is_empty_without_an_agent_credential():
    wf = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
"""
    assert pop_inline_model_secrets(parse_workflow(wf, "native")) == {}
