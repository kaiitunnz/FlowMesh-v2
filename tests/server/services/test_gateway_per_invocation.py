from types import SimpleNamespace

import pytest

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
        lambda tid: ResolvedGatewayBinding(mode=GatewayMode.OPENAI)
    )
    with pytest.raises(RuntimeError, match="egresses on the worker"):
        gateway.invoke("hi", "tsk-a")


def test_to_gateway_binding_resolves_the_control_plane_mode():
    pinned = AgentModelGatewayBinding(mode=ModelBindingMode.ECHO, provenance=_PROV)
    assert to_gateway_binding(pinned).mode is GatewayMode.ECHO


def test_to_gateway_binding_rejects_resident_for_external_gateway():
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.RESIDENT, service_model_ref="cat/x", provenance=_PROV
    )
    with pytest.raises(ResidentBindingNotServable, match="not served externally"):
        to_gateway_binding(pinned)
