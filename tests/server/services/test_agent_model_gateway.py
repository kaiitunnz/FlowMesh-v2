"""The agent-model gateway settles a mediated model boundary and resumes the episode.

A model request an agent defers becomes a durable invocation the gateway settles
through its upstream, off the agent's lane; the outcome injects at the originating call
so the episode resumes with the model result. An upstream failure fails the boundary
rather than resuming as a phantom empty success. The Responses API surface runs the same
conversion for a harness whose provider targets the gateway directly.
"""

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import AgentModelGatewayConfig, GatewayMode
from server.orchestration import WorkItemStatus
from server.services.agent_model_gateway import (
    AgentModelGateway,
    ResponsesRequest,
    build_agent_model_router,
)
from server.task.v2.representations.operators import FacadeDescriptor
from shared.harness import BoundaryEventKind, HarnessCapsule
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_TS = "2026-08-29T00:00:00Z"

_MODEL_AGENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: model-agent}
spec:
  graph:
    nodes:
      - name: solver
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _canned_gateway(runtime) -> AgentModelGateway:
    gateway = AgentModelGateway(
        runtime, AgentModelGatewayConfig(mode=GatewayMode.CANNED)
    )

    # A synchronous settle keeps the test deterministic; production submits off-lane.
    def _settle(env) -> None:
        runtime.settle_episode_invocation(
            env.task_id, env.call_correlation, gateway.invoke(env.request_payload)
        )

    runtime.set_model_settler(_settle)
    return gateway


def _model_boundary_adapter() -> ScriptedHarnessAdapter:
    return ScriptedHarnessAdapter(
        [
            ScriptedStep(
                op="boundary",
                kind=BoundaryEventKind.INVOCATION,
                call="c0",
                interface="model",
                payload="solve 2+2",
            ),
            ScriptedStep(op="complete", value_from="c0"),
        ],
        "v1",
    )


def test_model_boundary_settles_and_resumes_with_the_result() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        _canned_gateway(runtime)
        workflow_id, ids = await _register(runtime, _MODEL_AGENT_WF)
        solver = ids["solver"]
        adapter = _model_boundary_adapter()
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        # Step 1: the model boundary suspends the lane; the canned settle (synchronous
        # here) re-readies it with the injected result.
        engine.on_dispatched(solver, "wkr-1")
        first = adapter.start(solver, capsule=None, outcomes=[])
        runtime.mark_succeeded(
            solver, "wkr-1", {"agent_episode": first.model_dump(mode="json")}, _TS
        )
        wi = engine.work_item(solver)
        assert wi is not None and wi.status is WorkItemStatus.READY

        # Step 2: the re-dispatch carries the injected model result; it completes.
        dispatch = runtime.agent_episode_dispatch(solver)
        assert dispatch is not None and len(dispatch.delivered_outcomes) == 1
        capsule = HarnessCapsule(
            backend=dispatch.backend, blob=dispatch.capsule_blob or ""
        )
        engine.on_dispatched(solver, "wkr-1")
        done = adapter.start(
            solver, capsule=capsule, outcomes=dispatch.delivered_outcomes
        )
        assert done.value == "canned-response:solve 2+2"
        runtime.mark_succeeded(
            solver, "wkr-1", {"agent_episode": done.model_dump(mode="json")}, _TS
        )
        settled = engine.work_item(solver)
        assert settled is not None and settled.status is WorkItemStatus.SETTLED
        pub = engine.resolve_output(f"legacy:{solver}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def test_upstream_failure_fails_the_boundary_not_an_empty_success() -> None:
    # A gateway/upstream error must not resume the agent with an empty RESULT; it fails
    # the boundary so the workflow surfaces the failure.
    async def run() -> None:
        runtime = _runtime(FakeRegistry())

        def _settle(env) -> None:
            runtime.settle_episode_invocation(
                env.task_id, env.call_correlation, None, error="upstream 503"
            )

        runtime.set_model_settler(_settle)
        workflow_id, ids = await _register(runtime, _MODEL_AGENT_WF)
        solver = ids["solver"]
        adapter = _model_boundary_adapter()
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        engine.on_dispatched(solver, "wkr-1")
        first = adapter.start(solver, capsule=None, outcomes=[])
        runtime.mark_succeeded(
            solver, "wkr-1", {"agent_episode": first.model_dump(mode="json")}, _TS
        )
        wi = engine.work_item(solver)
        assert wi is not None and wi.status is not WorkItemStatus.READY
        failed = runtime.get_record(solver)
        assert failed is not None and str(failed.status) == "FAILED"

    asyncio.run(run())


def test_gateway_modes_and_responses_conversion() -> None:
    canned = AgentModelGateway(None, AgentModelGatewayConfig(mode=GatewayMode.CANNED))  # type: ignore[arg-type]
    echo = AgentModelGateway(None, AgentModelGatewayConfig(mode=GatewayMode.ECHO))  # type: ignore[arg-type]
    assert canned.invoke("hi") == "canned-response:hi"
    assert echo.invoke("hi") == "hi"
    # A JSON payload's prompt is extracted before the upstream runs.
    assert echo.invoke('{"prompt": "deep"}') == "deep"
    body = canned.responses(ResponsesRequest(model="m", input="q"))
    assert body["output"][0]["content"][0]["text"] == "canned-response:q"


def test_responses_router_serves_the_openai_surface() -> None:
    gateway = AgentModelGateway(None, AgentModelGatewayConfig(mode=GatewayMode.ECHO))  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    client = TestClient(app)
    resp = client.post("/v1/responses", json={"model": "m", "input": "ping"})
    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "ping"


def _responses_object(text: str) -> dict[str, Any]:
    return {
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": [{"type": "text", "text": "..."}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }


def test_response_text_reads_the_message_ignoring_reasoning() -> None:
    from server.services.agent_model_gateway import _response_text

    assert _response_text(_responses_object("answer")) == "answer"
    assert _response_text({"output": []}) == ""


def test_proxy_mode_settles_a_facade_single_shot(monkeypatch: Any) -> None:
    import server.services.agent_model_gateway as mod

    cfg = AgentModelGatewayConfig(
        mode=GatewayMode.PROXY, url="http://up/v1", model="qwen"
    )
    gateway = AgentModelGateway(None, cfg)  # type: ignore[arg-type]

    class _Resp:
        def raise_for_status(self) -> None: ...

        @staticmethod
        def json() -> dict[str, Any]:
            return _responses_object("settled")

    def _post(url: str, json: dict[str, Any], **_: Any) -> _Resp:
        assert url == "http://up/v1/responses"
        assert json["model"] == "qwen" and json["stream"] is False
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _post)
    assert gateway.invoke("solve 2+2") == "settled"


def test_proxy_mode_streams_the_upstream_turn_verbatim(monkeypatch: Any) -> None:
    import server.services.agent_model_gateway as mod

    cfg = AgentModelGatewayConfig(
        mode=GatewayMode.PROXY, url="http://up/v1", model="qwen"
    )
    gateway = AgentModelGateway(None, cfg)  # type: ignore[arg-type]
    sse = b"event: response.completed\ndata: {}\n\n"

    class _Upstream:
        async def __aenter__(self) -> "_Upstream":
            return self

        async def __aexit__(self, *exc: object) -> None: ...

        def raise_for_status(self) -> None: ...

        async def aiter_raw(self) -> Any:
            yield sse

    class _Client:
        def __init__(self, **_: Any) -> None: ...

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> None: ...

        def stream(self, method: str, url: str, **kwargs: Any) -> _Upstream:
            body = kwargs["json"]
            assert url == "http://up/v1/responses" and body["stream"] is True
            assert body["model"] == "qwen"
            # The namespace tool is dropped; function tools pass through.
            assert [t["name"] for t in body["tools"]] == ["exec_command"]
            return _Upstream()

    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {
        "model": "m",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "exec_command"},
            {"type": "namespace", "name": "multi_agent_v1", "tools": []},
        ],
    }
    resp = TestClient(app).post("/v1/responses", json=request)
    assert resp.status_code == 200 and resp.content == sse


def _facade_index() -> tuple[dict[str, Any], int]:
    from server.services.agent_model_gateway import _facade_calls, _forward_index

    output = [
        {"type": "reasoning", "content": []},
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "call_1",
            "arguments": '{"region": "reviewer", "args": {"focus": "security"}}',
        },
    ]
    facades = _facade_calls(output, {"spawn_agent": _spawn_facade()})
    assert len(facades) == 1
    history = [
        {"type": "message", "role": "user"},
        {"type": "function_call_output", "call_id": "fab-x:0", "output": "r"},
    ]
    return facades[0], _forward_index(history)


def test_facade_capture_maps_the_native_call_to_a_boundary() -> None:
    from server.services.agent_model_gateway import _facade_boundary

    call, index = _facade_index()
    # One fab- output already resolved in history advances the next facade's index.
    assert index == 1
    boundary = _facade_boundary(call, _spawn_facade(), "tsk-1:1")
    assert boundary.kind is BoundaryEventKind.SPAWN
    assert boundary.child_region_ref == "reviewer"
    assert boundary.call_correlation == "tsk-1:1"
    assert (
        boundary.request_payload is not None and "security" in boundary.request_payload
    )


def _upstream_client(monkeypatch: Any, output: list[dict[str, Any]]) -> None:
    import server.services.agent_model_gateway as mod

    class _Resp:
        def raise_for_status(self) -> None: ...

        @staticmethod
        def json() -> dict[str, Any]:
            return {"object": "response", "status": "completed", "output": output}

    class _Client:
        def __init__(self, **_: Any) -> None: ...

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> None: ...

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            assert url == "http://up/v1/responses"
            body = kwargs["json"]
            assert body["stream"] is False and body["model"] == "qwen"
            # The facade tool is injected alongside the harness's function tools.
            assert "spawn_agent" in [t.get("name") for t in body["tools"]]
            return _Resp()

    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)


def _spawn_facade() -> FacadeDescriptor:
    return FacadeDescriptor(
        name="spawn_agent",
        kind=BoundaryEventKind.SPAWN,
        tool_schema=json.dumps(
            {
                "type": "function",
                "name": "spawn_agent",
                "parameters": {
                    "type": "object",
                    "properties": {"region": {"type": "string"}},
                    "required": ["region"],
                },
            }
        ),
    )


def _agent_gateway() -> tuple[AgentModelGateway, list[Any]]:
    cfg = AgentModelGatewayConfig(
        mode=GatewayMode.PROXY, url="http://up/v1", model="qwen"
    )
    gateway = AgentModelGateway(None, cfg)  # type: ignore[arg-type]
    originated: list[Any] = []
    gateway.set_boundary_originator(lambda task, req: originated.append((task, req)))
    gateway.set_facade_resolver(lambda task: [_spawn_facade()])
    return gateway, originated


def test_agent_turn_captures_a_facade_and_clean_completes(monkeypatch: Any) -> None:
    output = [
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "call_1",
            "arguments": '{"region": "reviewer"}',
        }
    ]
    _upstream_client(monkeypatch, output)
    gateway, originated = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200
    # The boundary is originated server-side, keyed to the episode's task.
    assert len(originated) == 1
    task_id, boundary = originated[0]
    assert task_id == "tsk-1"
    assert boundary.kind is BoundaryEventKind.SPAWN
    assert boundary.call_correlation == "tsk-1:0"
    # Codex sees a clean turn-completing message, never the raw tool call.
    body = resp.content.decode()
    assert "Dispatched spawn_agent to the reviewer region" in body
    assert "function_call" not in body


def test_agent_turn_without_a_facade_passes_through(monkeypatch: Any) -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "just reasoning"}],
        }
    ]
    _upstream_client(monkeypatch, output)
    gateway, originated = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200 and not originated
    assert "just reasoning" in resp.content.decode()


def test_facade_turn_preserves_co_emitted_reasoning(monkeypatch: Any) -> None:
    # A reasoning item co-emitted with the facade is kept for continuity; only the
    # facade call becomes the dispatch message.
    output: list[dict[str, Any]] = [
        {"type": "reasoning", "id": "rs_1", "content": []},
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "call_1",
            "arguments": '{"region": "reviewer"}',
        },
    ]
    _upstream_client(monkeypatch, output)
    gateway, originated = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    body = resp.content.decode()
    assert len(originated) == 1
    assert '"reasoning"' in body and "rs_1" in body  # reasoning preserved
    assert "Dispatched spawn_agent" in body
    assert '"function_call"' not in body  # the facade call itself is not surfaced


def test_a_multi_facade_or_mixed_turn_is_denied(monkeypatch: Any) -> None:
    # Two facade calls, or a facade co-emitted with a real tool call, must not silently
    # collapse to one boundary — the turn is denied.
    two_facades: list[dict[str, Any]] = [
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "c1",
            "arguments": '{"region": "a"}',
        },
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "c2",
            "arguments": '{"region": "b"}',
        },
    ]
    facade_plus_real: list[dict[str, Any]] = [
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "c1",
            "arguments": '{"region": "a"}',
        },
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c2",
            "arguments": "{}",
        },
    ]
    for output in (two_facades, facade_plus_real):
        _upstream_client(monkeypatch, output)
        gateway, originated = _agent_gateway()
        app = FastAPI()
        app.include_router(build_agent_model_router(gateway))
        request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
        with pytest.raises(RuntimeError, match="exactly one facade call"):
            TestClient(app, raise_server_exceptions=True).post(
                "/agent/tsk-1/v1/responses", json=request
            )
        assert not originated
