import pytest

from server.config import AgentBindingConfig
from shared.tasks.specs import ModelBindingMode


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        "AGENT_HARNESS_DEFAULT_BACKEND",
        "AGENT_HARNESS_DEFAULT_VERSION",
        "AGENT_MODEL_GATEWAY_MODE",
        "AGENT_MODEL_GATEWAY_URL",
        "AGENT_MODEL_GATEWAY_MODEL",
        "AGENT_MODEL_GATEWAY_SECRETS",
        "AGENT_MODEL_GATEWAY_RESIDENT_MODELS",
        "UTU_LLM_BASE_URL",
        "UTU_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_gateway_env_values_become_binding_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_HARNESS_DEFAULT_BACKEND", "codex")
    monkeypatch.setenv("AGENT_HARNESS_DEFAULT_VERSION", "v2")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY_MODE", "openai")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY_URL", "https://api.example/v1")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY_MODEL", "gpt-4o")
    cfg = AgentBindingConfig.from_env()
    assert cfg.default_backend == "codex" and cfg.default_version == "v2"
    assert cfg.default_mode is ModelBindingMode.OPENAI
    assert cfg.default_url == "https://api.example/v1"
    assert cfg.default_model == "gpt-4o"


def test_model_defaults_do_not_inherit_utu_fallback(monkeypatch):
    monkeypatch.setenv("UTU_LLM_BASE_URL", "https://utu/v1")
    monkeypatch.setenv("UTU_LLM_MODEL", "utu-model")
    cfg = AgentBindingConfig.from_env()
    assert cfg.default_url is None
    assert cfg.default_model is None


def test_proxy_mode_defaults_a_binding_to_openai(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_GATEWAY_MODE", "proxy")
    assert AgentBindingConfig.from_env().default_mode is ModelBindingMode.OPENAI


def test_secret_refs_resolve_to_server_side_values_at_the_edge(monkeypatch):
    monkeypatch.setenv("TEAM_ENV", "resolved-team-value")
    monkeypatch.setenv(
        "AGENT_MODEL_GATEWAY_SECRETS", "team=TEAM_ENV,missing=ABSENT_ENV"
    )
    cfg = AgentBindingConfig.from_env()
    assert set(cfg.secrets) == {"team"}
    assert cfg.secrets["team"].get_secret_value() == "resolved-team-value"


def test_resident_catalog_parses_family_and_qualifiers(monkeypatch):
    monkeypatch.setenv(
        "AGENT_MODEL_GATEWAY_RESIDENT_MODELS",
        "qwen=inference:vllm:isolated, plain=embed",
    )
    catalog = AgentBindingConfig.from_env().resident_models
    assert catalog["qwen"].family == "inference"
    assert catalog["qwen"].engine_batch_key == "vllm"
    assert catalog["qwen"].isolation == "isolated"
    assert catalog["plain"].family == "embed"
    assert catalog["plain"].engine_batch_key is None
