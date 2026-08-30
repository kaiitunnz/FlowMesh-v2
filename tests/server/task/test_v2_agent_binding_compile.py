import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import (
    AgentBindingDefaults,
    service_family_for_ref,
)
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.operators import AgentOperator, BindingProvenance
from server.task.v2.representations.source import FrontendWorkflowSource
from shared.tasks import TaskType
from shared.tasks.specs import ModelBindingMode


def _indent(body: str) -> str:
    return "\n".join(
        ("          " + line.strip()) if line.strip() else ""
        for line in body.splitlines()
    )


def _agent_workflow(spec_body: str) -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: agent
          configName: default
          task: do
{_indent(spec_body)}
"""


def _compile(text, defaults, secret_refs=None):
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_workflow(
        "wfl-t", parsed, source, bindings=defaults, secret_refs=secret_refs
    )


def _agent(template) -> AgentOperator:
    return next(op for op in template.operators if isinstance(op, AgentOperator))


def test_source_harness_and_model_binding_pin_with_source_provenance():
    text = _agent_workflow(
        """        harness: {backend: scripted, version: v2, params: {script: []}}
        model_binding: {mode: openai, url: "https://api.src/v1", model: src-model}"""
    )
    template, _ = _compile(text, AgentBindingDefaults(default_backend="codex"))
    agent = _agent(template)
    assert agent.harness_binding.backend == "scripted"
    assert agent.harness_binding.provenance.backend is BindingProvenance.SOURCE
    assert agent.model_binding.url == "https://api.src/v1"
    assert agent.model_binding.provenance.url is BindingProvenance.SOURCE


def test_deployment_default_beats_fallback_but_loses_to_source():
    defaults = AgentBindingDefaults(
        default_backend="codex",
        default_mode=ModelBindingMode.OPENAI,
        default_url="https://api.default/v1",
    )
    template, _ = _compile(_agent_workflow(""), defaults)
    agent = _agent(template)
    assert agent.harness_binding.backend == "codex"
    assert agent.harness_binding.provenance.backend is BindingProvenance.DEFAULT
    assert agent.model_binding.mode is ModelBindingMode.OPENAI
    assert agent.model_binding.url == "https://api.default/v1"
    assert agent.model_binding.provenance.url is BindingProvenance.DEFAULT
    # gpt-4o-mini fallback only when openai + url are otherwise complete.
    assert agent.model_binding.model == "gpt-4o-mini"
    assert agent.model_binding.provenance.model is BindingProvenance.FALLBACK


def test_bare_agent_without_default_backend_fails_validation():
    with pytest.raises(CompileError, match="agent.harness.unresolved"):
        _compile(_agent_workflow(""), AgentBindingDefaults())


def test_canned_fallback_when_no_model_binding_or_default():
    template, _ = _compile(
        _agent_workflow(""), AgentBindingDefaults(default_backend="codex")
    )
    binding = _agent(template).model_binding
    assert binding.mode is ModelBindingMode.CANNED
    assert binding.url is None and binding.secret_ref is None


def test_openai_binding_without_url_is_rejected():
    text = _agent_workflow("        model_binding: {mode: openai, model: m}")
    with pytest.raises(CompileError, match="agent.model_binding.missing_url"):
        _compile(text, AgentBindingDefaults(default_backend="codex"))


def test_any_external_url_compiles_without_an_allowlist():
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://anything.example/v1", model: m}'
    )
    template, _ = _compile(text, AgentBindingDefaults(default_backend="codex"))
    assert _agent(template).model_binding.url == "https://anything.example/v1"


def test_vaulted_secret_ref_pins_on_the_model_binding():
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://a/v1", model: m}'
    )
    parsed = parse_workflow(text, "native")
    task_id = next(
        t.task_id for t in parsed.tasks if t.task.spec.taskType == TaskType.AGENT
    )
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow(
        "wfl-t",
        parsed,
        source,
        bindings=AgentBindingDefaults(default_backend="codex"),
        secret_refs={task_id: "msk-abc123"},
    )
    assert _agent(template).model_binding.secret_ref == "msk-abc123"


def test_any_resident_reference_pins_a_canonical_service_family():
    text = _agent_workflow(
        "        model_binding: {mode: resident, service_model_ref: Qwen/Qwen3-4B}"
    )
    template, plan = _compile(text, AgentBindingDefaults(default_backend="codex"))
    binding = _agent(template).model_binding
    assert binding.mode is ModelBindingMode.RESIDENT
    assert binding.url is None
    resident = [n for n in plan.nodes if n.service_family_requirement is not None]
    assert len(resident) == 1
    assert resident[0].service_family_requirement.family == "Qwen/Qwen3-4B"
    assert resident[0].residency_intent.required is True


def test_resident_binding_without_a_reference_is_rejected():
    text = _agent_workflow("        model_binding: {mode: resident}")
    with pytest.raises(CompileError, match="missing_resident_ref"):
        _compile(text, AgentBindingDefaults(default_backend="codex"))


def test_identical_resident_references_derive_the_same_family():
    assert service_family_for_ref("Qwen/Qwen3-4B ") == service_family_for_ref(
        "Qwen/Qwen3-4B"
    )


def test_compat_sugar_normalizes_harness_params_to_openai_binding():
    text = _agent_workflow(
        "harness: {backend: codex, params: {base_url: 'https://a/v1', model: m}}"
    )
    template, _ = _compile(text, AgentBindingDefaults())
    binding = _agent(template).model_binding
    assert binding.mode is ModelBindingMode.OPENAI
    assert binding.url == "https://a/v1" and binding.model == "m"
