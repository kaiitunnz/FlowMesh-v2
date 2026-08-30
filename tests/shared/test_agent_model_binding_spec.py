import pytest
from pydantic import ValidationError

from shared.tasks.specs import (
    AgentHarnessSpec,
    AgentModelBindingSpec,
    AgentSpecStrict,
    ModelBindingMode,
)


def test_external_openai_binding_accepts_url_model_secret_ref():
    binding = AgentModelBindingSpec(
        mode=ModelBindingMode.OPENAI,
        url="https://api.example/v1",
        model="gpt-4o-mini",
        secret_ref="team-openai",
    )
    assert binding.mode is ModelBindingMode.OPENAI
    assert binding.secret_ref == "team-openai"


def test_resident_binding_needs_no_url_or_credential():
    binding = AgentModelBindingSpec(
        mode=ModelBindingMode.RESIDENT, service_model_ref="catalog/qwen"
    )
    assert binding.service_model_ref == "catalog/qwen"
    assert binding.url is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "openai", "url": "https://user:pass@host/v1"},
        {"mode": "resident", "url": "https://host/v1"},
        {"mode": "resident", "secret_ref": "x"},
        {"mode": "openai", "service_model_ref": "catalog/x"},
        {"mode": "canned", "model": "m"},
        {"mode": "echo", "url": "https://host"},
        {"service_model_ref": "catalog/x", "url": "https://host"},
    ],
)
def test_incoherent_or_credential_bindings_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        AgentModelBindingSpec(**kwargs)


def test_raw_api_key_in_model_binding_is_rejected():
    with pytest.raises(ValidationError):
        AgentModelBindingSpec(mode="openai", url="https://host/v1", api_key="sk-secret")


@pytest.mark.parametrize(
    "key", ["api_key", "apiKey", "openai_token", "SECRET", "password"]
)
def test_credential_harness_params_are_rejected(key):
    with pytest.raises(ValidationError):
        AgentHarnessSpec(backend="codex", params={key: "sk-secret"})


def test_non_secret_harness_params_are_allowed():
    spec = AgentHarnessSpec(backend="codex", params={"base_url": "x", "model": "m"})
    assert spec.params == {"base_url": "x", "model": "m"}


def test_agent_spec_carries_model_binding_beside_harness():
    spec = AgentSpecStrict(
        taskType="agent",
        harness=AgentHarnessSpec(backend="scripted", params={"script": []}),
        model_binding=AgentModelBindingSpec(mode="canned"),
    )
    assert spec.model_binding is not None
    assert spec.model_binding.mode is ModelBindingMode.CANNED
