"""The agent-model gateway settles a canned or echo model boundary and resumes it.

A model request an agent defers with a ``canned``/``echo`` binding becomes a durable
invocation the gateway settles off the agent's lane; the outcome injects at the
originating call so the episode resumes with the model result. A settle failure fails
the boundary rather than resuming as a phantom empty success. An external binding
egresses on the agent's worker and never reaches this path.
"""

import asyncio

from server.config import AgentModelGatewayConfig, GatewayMode
from server.orchestration import WorkItemStatus
from server.services.agent_model_gateway import AgentModelGateway
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
    # A settle error must not resume the agent with an empty RESULT; it fails the
    # boundary so the workflow surfaces the failure.
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


def test_gateway_canned_and_echo_modes() -> None:
    canned = AgentModelGateway(None, AgentModelGatewayConfig(mode=GatewayMode.CANNED))  # type: ignore[arg-type]
    echo = AgentModelGateway(None, AgentModelGatewayConfig(mode=GatewayMode.ECHO))  # type: ignore[arg-type]
    assert canned.invoke("hi") == "canned-response:hi"
    assert echo.invoke("hi") == "hi"
    # A JSON payload's prompt is extracted before the settle runs.
    assert echo.invoke('{"prompt": "deep"}') == "deep"
