from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from server.config import AgentModelGatewayConfig, GatewayMode
from server.services.agent_model_gateway import (
    AgentModelGateway,
    ResidentBindingNotServable,
    ResolvedGatewayBinding,
    to_gateway_binding,
)
from server.task.v2.representations.operators import (
    AgentModelGatewayBinding,
    BindingProvenance,
    ModelBindingProvenance,
)
from shared.tasks.specs import ModelBindingMode

_PROV = ModelBindingProvenance(
    mode=BindingProvenance.SOURCE,
    url=BindingProvenance.SOURCE,
    model=BindingProvenance.SOURCE,
)


class _FakeVault:
    """A workflow-scoped credential store, keyed by (workflow_id, ref)."""

    def __init__(self, store: dict[tuple[str, str], SecretStr]) -> None:
        self._store = store

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        return self._store.get((workflow_id, ref)) if ref else None


def _gateway() -> AgentModelGateway:
    settler = SimpleNamespace(settle_episode_invocation=lambda *a, **k: True)
    return AgentModelGateway(settler, AgentModelGatewayConfig(mode=GatewayMode.CANNED))


def test_two_tasks_resolve_different_upstreams_without_cross_talk():
    gateway = _gateway()
    bindings = {
        "tsk-a": ResolvedGatewayBinding(mode=GatewayMode.ECHO),
        "tsk-b": ResolvedGatewayBinding(mode=GatewayMode.CANNED),
    }
    gateway.set_binding_resolver(lambda tid: bindings.get(tid))
    assert gateway.invoke("hello", "tsk-a") == "hello"
    assert gateway.invoke("hello", "tsk-b") == "canned-response:hello"


def test_canned_and_echo_settle_without_an_external_binding():
    gateway = _gateway()
    gateway.set_binding_resolver(
        lambda tid: ResolvedGatewayBinding(mode=GatewayMode.ECHO)
    )
    assert gateway.invoke("x", "tsk") == "x"


def test_no_resolver_falls_back_to_deployment_default():
    settler = SimpleNamespace(settle_episode_invocation=lambda *a, **k: True)
    gateway = AgentModelGateway(settler, AgentModelGatewayConfig(mode=GatewayMode.ECHO))
    assert gateway.invoke("echoed", "tsk") == "echoed"


def test_an_external_binding_never_settles_on_the_server():
    gateway = _gateway()
    gateway.set_binding_resolver(
        lambda tid: ResolvedGatewayBinding(
            mode=GatewayMode.OPENAI, url="https://pinned/v1", model="m"
        )
    )
    with pytest.raises(RuntimeError, match="egresses on the worker"):
        gateway.invoke("hi", "tsk-a")


def _openai_binding(secret_ref: str | None = None) -> AgentModelGatewayBinding:
    return AgentModelGatewayBinding(
        mode=ModelBindingMode.OPENAI,
        url="https://h/v1",
        model="m",
        secret_ref=secret_ref,
        provenance=_PROV,
    )


def test_to_gateway_binding_resolves_vaulted_key_within_its_workflow():
    vault = _FakeVault({("wfl-1", "msk-a"): SecretStr("sk-user")})
    resolved = to_gateway_binding(_openai_binding("msk-a"), vault, "wfl-1")
    assert resolved.api_key == "sk-user"
    assert resolved.url == "https://h/v1" and resolved.model == "m"


def test_vaulted_ref_does_not_resolve_across_workflows():
    vault = _FakeVault({("wfl-1", "msk-a"): SecretStr("sk-user")})
    # Another workflow presenting the same ref gets no credential.
    assert to_gateway_binding(_openai_binding("msk-a"), vault, "wfl-2").api_key is None


def test_missing_ref_is_unauthenticated():
    resolved = to_gateway_binding(_openai_binding(None), _FakeVault({}), "wfl-1")
    assert resolved.api_key is None


def test_to_gateway_binding_rejects_resident_for_external_gateway():
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.RESIDENT, service_model_ref="cat/x", provenance=_PROV
    )
    with pytest.raises(ResidentBindingNotServable, match="not served externally"):
        to_gateway_binding(pinned, _FakeVault({}), "wfl-1")
