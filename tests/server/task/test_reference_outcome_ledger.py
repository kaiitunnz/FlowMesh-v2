"""A reference-backed settle keeps the payload out of the ledger and re-dispatch.

The orchestration ledger records only a bounded manifest for a mediated outcome
materialized by reference; the outcome content never lands on the boundary envelope, the
snapshot, or the re-dispatched delivered outcome. A restart rehydrates the same
reference.
"""

import asyncio

from server.orchestration import WorkItemStatus
from server.orchestration.state import LedgerSnapshot
from shared.harness import BoundaryEventKind
from shared.outcome import OutcomeAccessBinding, OutcomeManifest, content_digest
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_AGENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: agent-ref}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _manifest() -> OutcomeManifest:
    return OutcomeManifest(
        content_digest=content_digest(b"draft-result"),
        size_bytes=12,
        media_type="application/json",
        idempotency_key="idm-1",
        access=OutcomeAccessBinding(tenant="local"),
    )


def test_reference_settle_keeps_payload_out_of_ledger() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        # No model settler is installed, so the model boundary suspends held until the
        # test settles it by reference.
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        _one_step(runtime, adapter, writer)  # model boundary → suspend, held

        manifest = _manifest()
        assert runtime.settle_episode_invocation(writer, "m0", ref=manifest) is True

        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.READY

        # The re-dispatch injects the reference, never an inline value.
        _capsule, outcomes = engine.episode_context(writer)
        assert len(outcomes) == 1
        assert outcomes[0].outcome_ref == manifest and outcomes[0].value is None

        # The snapshot holds the bounded manifest and never the outcome content.
        snap = engine.to_snapshot()
        m0 = next(e for e in snap.boundary_events if e.call_correlation == "m0")
        assert m0.outcome_ref == manifest and m0.outcome_value is None
        assert "draft-result" not in snap.model_dump_json()

        # A restart rehydrates the same reference.
        restored = LedgerSnapshot.model_validate_json(snap.model_dump_json())
        r0 = next(e for e in restored.boundary_events if e.call_correlation == "m0")
        assert r0.outcome_ref == manifest

    asyncio.run(run())


def _one_step(runtime, adapter, task_id: str, worker: str = "wkr-1") -> None:
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id)
    assert engine is not None and dispatch is not None
    engine.on_dispatched(task_id, worker)
    result = adapter.start(task_id, capsule=None, outcomes=dispatch.delivered_outcomes)
    runtime.mark_succeeded(
        task_id,
        worker,
        {"agent_episode": result.model_dump(mode="json")},
        "2026-08-29T00:00:00Z",
    )
