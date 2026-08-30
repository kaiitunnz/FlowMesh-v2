from server.task.v2.representations.operators import (
    AgentHarnessBinding,
    AgentModelGatewayBinding,
    AgentOperator,
    BindingKey,
    BindingProvenance,
    HarnessBindingProvenance,
    ModelBindingProvenance,
)
from server.task.v2.representations.plan import ResidencyIntent
from shared.tasks import TaskType
from shared.tasks.specs import ModelBindingMode


def _operator() -> AgentOperator:
    return AgentOperator(
        operator_id="agent-1",
        source_ref="agent-1",
        binding=BindingKey(task_type=TaskType.AGENT),
        harness_binding=AgentHarnessBinding(
            backend="codex",
            version="v1",
            params={"base_url": "http://gw/agent/x/v1", "model": "m"},
            provenance=HarnessBindingProvenance(
                backend=BindingProvenance.SOURCE, version=BindingProvenance.DEFAULT
            ),
        ),
        model_binding=AgentModelGatewayBinding(
            mode=ModelBindingMode.OPENAI,
            url="https://api.example/v1",
            model="gpt-4o-mini",
            secret_ref="team-openai",
            provenance=ModelBindingProvenance(
                mode=BindingProvenance.SOURCE,
                url=BindingProvenance.DEFAULT,
                model=BindingProvenance.FALLBACK,
            ),
        ),
    )


def test_operator_binding_round_trips_through_json():
    op = _operator()
    back = AgentOperator.model_validate_json(op.model_dump_json())
    assert back.harness_binding is not None and back.model_binding is not None
    assert back.harness_binding.backend == "codex"
    assert back.harness_binding.provenance.version is BindingProvenance.DEFAULT
    assert back.model_binding.mode is ModelBindingMode.OPENAI
    assert back.model_binding.secret_ref == "team-openai"
    assert back.model_binding.provenance.model is BindingProvenance.FALLBACK


def test_serialized_binding_carries_only_the_secret_reference():
    dumped = _operator().model_dump_json()
    assert "team-openai" in dumped
    assert "secret_ref" in dumped
    assert "api_key" not in dumped and "password" not in dumped


def test_required_residency_intent_is_an_explicit_marker():
    intent = ResidencyIntent(service_family="inference:qwen", required=True)
    assert intent.required is True
    assert intent.warmth is None
    assert ResidencyIntent().required is False
