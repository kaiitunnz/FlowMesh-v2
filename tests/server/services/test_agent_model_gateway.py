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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import AgentModelGatewayConfig, GatewayMode
from server.orchestration import WorkItemStatus
from server.orchestration.tool_dispatch import FacadeCompletionMode
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


def test_facade_capture_maps_the_native_call_to_a_group_member(
    monkeypatch: Any,
) -> None:
    _, index = _facade_index()
    # One fab- output already resolved in history advances the next group's base.
    assert index == 1
    output: list[dict[str, Any]] = [
        {"type": "reasoning", "content": []},
        {
            "type": "function_call",
            "name": "spawn_agent",
            "call_id": "call_1",
            "arguments": '{"region": "reviewer", "args": {"focus": "security"}}',
        },
    ]
    _upstream_client(monkeypatch, output)
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    history = [
        {"type": "message", "role": "user"},
        {"type": "function_call_output", "call_id": "fab-x:0", "output": "r"},
    ]
    request = {"model": "m", "input": history, "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200
    _, group = groups[0]
    assert group.group_id == "tsk-1:1"  # the base tracks the resolved-outcome count
    (member,) = group.members
    assert member.kind is BoundaryEventKind.SPAWN
    assert member.interface_or_region == "reviewer"
    assert member.call_correlation == "tsk-1:1:0"
    assert member.request_payload is not None and "security" in member.request_payload


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


def _search_facade() -> FacadeDescriptor:
    return FacadeDescriptor(
        name="web_search",
        kind=BoundaryEventKind.INVOCATION,
        interface="search/v1",
        tool_schema=json.dumps({"type": "function", "name": "web_search"}),
    )


def _agent_gateway() -> tuple[AgentModelGateway, list[Any]]:
    cfg = AgentModelGatewayConfig(
        mode=GatewayMode.PROXY, url="http://up/v1", model="qwen"
    )
    gateway = AgentModelGateway(None, cfg)  # type: ignore[arg-type]
    groups: list[Any] = []
    gateway.set_facade_group_originator(
        lambda task, group: groups.append((task, group))
    )
    gateway.set_facade_resolver(lambda task: [_spawn_facade(), _search_facade()])
    return gateway, groups


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
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200
    # The group is originated server-side, keyed to the episode's task.
    assert len(groups) == 1
    task_id, group = groups[0]
    assert task_id == "tsk-1"
    (member,) = group.members
    assert member.kind is BoundaryEventKind.SPAWN
    assert member.completion_mode is FacadeCompletionMode.ADMIT_AND_CLOSE
    assert member.call_correlation == "tsk-1:0:0"
    assert member.interface_or_region == "reviewer"
    # Codex sees a clean turn-completing message, never the raw tool call.
    body = resp.content.decode()
    assert "Dispatched 1 spawn(s)" in body
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
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200 and not groups
    assert "just reasoning" in resp.content.decode()


def test_facade_turn_preserves_co_emitted_reasoning(monkeypatch: Any) -> None:
    # A reasoning item co-emitted with the facade is kept for continuity; only the
    # facade call becomes the dispatch summary.
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
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "gpt-5-codex", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    body = resp.content.decode()
    assert len(groups) == 1
    assert '"reasoning"' in body and "rs_1" in body  # reasoning preserved
    assert "Dispatched 1 spawn(s)" in body
    assert '"function_call"' not in body  # the facade call itself is not surfaced


def _fn(name: str, call_id: str, args: str = "{}") -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": args,
    }


def test_mixed_and_multi_spawn_facade_turns_form_one_group(monkeypatch: Any) -> None:
    # Real models co-emit multiple spawns and search on one turn; they all join a single
    # ordered group with kind-specific completion, never a fail-closed refusal.
    output = [
        _fn("spawn_agent", "c1", '{"region": "a"}'),
        _fn("spawn_agent", "c2", '{"region": "a"}'),
        _fn("web_search", "c3", '{"query": "x"}'),
    ]
    _upstream_client(monkeypatch, output)
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "m", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)

    assert resp.status_code == 200
    assert len(groups) == 1
    _, group = groups[0]
    assert [m.kind for m in group.members] == [
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.INVOCATION,
    ]
    assert [m.completion_mode for m in group.members] == [
        FacadeCompletionMode.ADMIT_AND_CLOSE,
        FacadeCompletionMode.ADMIT_AND_CLOSE,
        FacadeCompletionMode.AWAIT_OUTCOME,
    ]
    body = resp.content.decode()
    assert "2 spawn(s)" in body and "1 web search" in body


def test_a_fenced_second_facade_group_returns_typed_busy(monkeypatch: Any) -> None:
    # With a group already holding the resume gate, a distinct second group returns a
    # typed busy turn — never a 500, never a dropped-silently child, never a raw call.
    output = [_fn("web_search", "s0", '{"query": "again"}')]
    _upstream_client(monkeypatch, output)
    gateway, groups = _agent_gateway()
    gateway.set_facade_fence(lambda task: True)  # a group is already open
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "m", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)
    assert resp.status_code == 200
    assert not groups  # no second group captured
    body = resp.content.decode()
    assert "still in flight" in body
    assert '"function_call"' not in body  # the deferred facade is not surfaced


def test_parallel_search_facades_form_one_ordered_group(monkeypatch: Any) -> None:
    # Three web_search calls in one turn become one ordered group; a co-emitted native
    # tool call is preserved verbatim so the harness runs it.
    output = [
        _fn("web_search", "s0", '{"query": "retrieval"}'),
        _fn("web_search", "s1", '{"query": "grounding"}'),
        _fn("web_search", "s2", '{"query": "evaluation"}'),
        _fn("exec_command", "n0", '{"cmd": "ls"}'),
    ]
    _upstream_client(monkeypatch, output)
    gateway, groups = _agent_gateway()
    app = FastAPI()
    app.include_router(build_agent_model_router(gateway))
    request = {"model": "m", "input": [], "tools": [], "stream": True}
    resp = TestClient(app).post("/agent/tsk-1/v1/responses", json=request)
    assert resp.status_code == 200
    assert len(groups) == 1
    task_id, group = groups[0]
    members = group.members
    assert task_id == "tsk-1" and len(members) == 3
    assert [m.ordinal for m in members] == [0, 1, 2]
    assert [m.harness_call_id for m in members] == ["s0", "s1", "s2"]
    assert all(m.interface_or_region == "search/v1" for m in members)
    assert all(m.completion_mode is FacadeCompletionMode.AWAIT_OUTCOME for m in members)
    assert members[0].call_correlation != members[1].call_correlation
    body = resp.content.decode()
    assert "exec_command" in body  # native call preserved verbatim
    assert "Dispatched 3 web search" in body
