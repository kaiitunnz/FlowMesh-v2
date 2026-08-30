import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

import server.services.agent_model_gateway as gw
from server.config import AgentModelGatewayConfig, GatewayMode
from server.services.agent_model_gateway import (
    AgentModelGateway,
    ResidentBindingNotServable,
    ResolvedGatewayBinding,
    to_gateway_binding,
)
from server.services.secret_ref import SecretRefResolver
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


class _Resp:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}


def test_two_tasks_resolve_different_upstreams_without_cross_talk():
    gateway = _gateway()
    bindings = {
        "tsk-a": ResolvedGatewayBinding(mode=GatewayMode.ECHO),
        "tsk-b": ResolvedGatewayBinding(mode=GatewayMode.CANNED),
    }
    gateway.set_binding_resolver(lambda tid: bindings.get(tid))
    assert gateway.invoke("hello", "tsk-a") == "hello"
    assert gateway.invoke("hello", "tsk-b") == "canned-response:hello"


def test_body_cannot_override_the_pinned_upstream(monkeypatch):
    captured: list[tuple] = []

    def _post(url, json, headers, timeout):  # noqa: A002 - mirror requests.post
        captured.append((url, json, headers))
        return _Resp()

    monkeypatch.setattr(gw.requests, "post", _post)
    gateway = _gateway()
    gateway.set_binding_resolver(
        lambda tid: ResolvedGatewayBinding(
            mode=GatewayMode.OPENAI,
            url="https://pinned/v1",
            model="pinned-model",
            api_key="sk-pinned",
        )
    )
    gateway.invoke('{"prompt": "hi", "model": "evil", "url": "https://evil"}', "tsk-a")
    url, body, headers = captured[0]
    assert url == "https://pinned/v1/chat/completions"
    assert body["model"] == "pinned-model"
    assert headers["Authorization"] == "Bearer sk-pinned"


def test_canned_and_echo_make_no_upstream_call(monkeypatch):
    called: list[int] = []
    monkeypatch.setattr(gw.requests, "post", lambda *a, **k: called.append(1))
    gateway = _gateway()
    gateway.set_binding_resolver(
        lambda tid: ResolvedGatewayBinding(mode=GatewayMode.ECHO)
    )
    assert gateway.invoke("x", "tsk") == "x"
    assert not called


def test_no_resolver_falls_back_to_deployment_default():
    settler = SimpleNamespace(settle_episode_invocation=lambda *a, **k: True)
    gateway = AgentModelGateway(settler, AgentModelGatewayConfig(mode=GatewayMode.ECHO))
    assert gateway.invoke("echoed", "tsk") == "echoed"


def test_to_gateway_binding_resolves_secret_ref_server_side():
    resolver = SecretRefResolver({"team": SecretStr("sk-team")})
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.OPENAI,
        url="https://h/v1",
        model="m",
        secret_ref="team",
        provenance=_PROV,
    )
    resolved = to_gateway_binding(pinned, resolver, default_api_key=None)
    assert resolved.api_key == "sk-team"
    assert resolved.url == "https://h/v1" and resolved.model == "m"


def test_deployment_key_pairs_only_with_the_deployment_default_url():
    prov = ModelBindingProvenance(
        mode=BindingProvenance.DEFAULT,
        url=BindingProvenance.DEFAULT,
        model=BindingProvenance.DEFAULT,
    )
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.OPENAI, url="https://gw/v1", model="m", provenance=prov
    )
    resolved = to_gateway_binding(pinned, SecretRefResolver({}), "sk-default")
    assert resolved.api_key == "sk-default"


def test_deployment_key_is_never_sent_to_a_workflow_provided_url():
    # A workflow-provided (source) url with no secret_ref must not inherit the
    # deployment key, or the shared key would leak to a workflow-chosen host.
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.OPENAI, url="https://byo/v1", model="m", provenance=_PROV
    )
    resolved = to_gateway_binding(pinned, SecretRefResolver({}), "sk-default")
    assert resolved.api_key is None


def test_to_gateway_binding_rejects_resident_for_external_gateway():
    pinned = AgentModelGatewayBinding(
        mode=ModelBindingMode.RESIDENT, service_model_ref="cat/x", provenance=_PROV
    )
    with pytest.raises(ResidentBindingNotServable, match="not served externally"):
        to_gateway_binding(pinned, SecretRefResolver({}), None)


def test_secret_resolver_denies_unregistered_ref():
    assert SecretRefResolver({"team": SecretStr("s")}).resolve("rogue") is None
    assert SecretRefResolver({}).resolve(None) is None


def test_originate_reports_a_resident_binding_as_not_implemented():
    # A resident binding is detection-only in this transition: the episode surface must
    # fail with a typed 501, not an unhandled exception rendered as a 500.
    gateway = _gateway()

    def _resident(_task_id):
        raise ResidentBindingNotServable("resident is not served externally")

    gateway.set_binding_resolver(_resident)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(gateway.originate_or_forward("tsk", {"input": "hi"}))
    assert excinfo.value.status_code == 501


def test_originate_surfaces_a_resolution_failure_as_a_clean_502():
    gateway = _gateway()

    def _broken(_task_id):
        raise RuntimeError("resolver blew up")

    gateway.set_binding_resolver(_broken)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(gateway.originate_or_forward("tsk", {"input": "hi"}))
    assert excinfo.value.status_code == 502
