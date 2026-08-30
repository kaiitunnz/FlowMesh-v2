import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import (
    AgentBindingDefaults,
    ResidentModelEntry,
)
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.operators import (
    AgentOperator,
    BindingProvenance,
)
from server.task.v2.representations.source import FrontendWorkflowSource
from shared.tasks.specs import ModelBindingMode

_RESIDENT = AgentBindingDefaults(
    default_backend="codex",
    resident_catalog={"catalog/qwen": ResidentModelEntry(family="inference:qwen")},
)


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


def _compile(text: str, defaults: AgentBindingDefaults):
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_workflow("wfl-t", parsed, source, bindings=defaults)


def _agent(template) -> AgentOperator:
    return next(op for op in template.operators if isinstance(op, AgentOperator))


def test_source_harness_and_model_binding_pin_with_source_provenance():
    text = _agent_workflow(
        """        harness: {backend: scripted, version: v2, params: {script: []}}
        model_binding: {mode: openai, url: "https://api.src/v1", model: src-model}"""
    )
    template, _ = _compile(
        text,
        AgentBindingDefaults(
            default_backend="codex", allowed_upstream_hosts=frozenset({"api.src"})
        ),
    )
    agent = _agent(template)
    assert agent.harness_binding.backend == "scripted"
    assert agent.harness_binding.provenance.backend is BindingProvenance.SOURCE
    assert agent.model_binding.url == "https://api.src/v1"
    assert agent.model_binding.provenance.url is BindingProvenance.SOURCE


def test_deployment_default_beats_fallback_but_loses_to_source():
    text = _agent_workflow("")
    defaults = AgentBindingDefaults(
        default_backend="codex",
        default_mode=ModelBindingMode.OPENAI,
        default_url="https://api.default/v1",
        allowed_upstream_hosts=frozenset({"api.default"}),
    )
    template, _ = _compile(text, defaults)
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


def test_resident_binding_pins_service_family_and_required_intent():
    text = _agent_workflow(
        "        model_binding: {mode: resident, service_model_ref: catalog/qwen}"
    )
    template, plan = _compile(text, _RESIDENT)
    binding = _agent(template).model_binding
    assert binding.mode is ModelBindingMode.RESIDENT
    assert binding.url is None
    resident = [n for n in plan.nodes if n.service_family_requirement is not None]
    assert len(resident) == 1
    assert resident[0].service_family_requirement.family == "inference:qwen"
    assert resident[0].residency_intent.required is True


def test_unauthorized_resident_reference_is_rejected():
    text = _agent_workflow(
        "        model_binding: {mode: resident, service_model_ref: catalog/unknown}"
    )
    with pytest.raises(CompileError, match="unauthorized_resident"):
        _compile(text, _RESIDENT)


def test_unauthorized_secret_ref_is_rejected():
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://a/v1", model: m, secret_ref: bad}'
    )
    with pytest.raises(CompileError, match="unauthorized_secret"):
        _compile(
            text,
            AgentBindingDefaults(
                default_backend="codex", allowed_upstream_hosts=frozenset({"a"})
            ),
        )


def test_authorized_secret_ref_compiles():
    defaults = AgentBindingDefaults(
        default_backend="codex",
        secret_allowlist=frozenset({"team"}),
        allowed_upstream_hosts=frozenset({"a"}),
    )
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://a/v1", model: m, secret_ref: team}'
    )
    template, _ = _compile(text, defaults)
    assert _agent(template).model_binding.secret_ref == "team"


def test_external_url_off_the_allowlist_is_rejected():
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://evil/v1", model: m}'
    )
    with pytest.raises(CompileError, match="upstream_not_allowed"):
        _compile(text, AgentBindingDefaults(default_backend="codex"))


def test_external_url_on_the_allowlist_compiles():
    text = _agent_workflow(
        'model_binding: {mode: openai, url: "https://trusted/v1", model: m}'
    )
    defaults = AgentBindingDefaults(
        default_backend="codex", allowed_upstream_hosts=frozenset({"trusted"})
    )
    template, _ = _compile(text, defaults)
    assert _agent(template).model_binding.url == "https://trusted/v1"


def test_compat_sugar_normalizes_harness_params_to_openai_binding():
    text = _agent_workflow(
        "harness: {backend: codex, params: {base_url: 'https://a/v1', model: m}}"
    )
    template, _ = _compile(
        text, AgentBindingDefaults(allowed_upstream_hosts=frozenset({"a"}))
    )
    binding = _agent(template).model_binding
    assert binding.mode is ModelBindingMode.OPENAI
    assert binding.url == "https://a/v1" and binding.model == "m"
