import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from server.config import OrchestrationConfig
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from server.task.v2 import PersistedV2Workflow
from shared.harness import HarnessCapsule
from tests.server.task.test_v2_orchestration import FakeRegistry, _WorkerRegistryStub
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_RAW_KEY = "sk-super-secret-USER-KEY"

_WF = f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: bind}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: solver
        spec:
          taskType: agent
          v2: {{authority: {{invoke: [model], delegate: []}}, tools: [{{name: model}}]}}
          harness: {{backend: scripted, version: v1, params: {{script: []}}}}
          model_binding:
            mode: openai
            url: "https://api.example/v1"
            model: m
            api_key: "{_RAW_KEY}"
"""


class _RecordingVault:
    def __init__(self) -> None:
        self.stored: dict[tuple[str, str], SecretStr] = {}
        self.purged: list[str] = []

    async def store(self, workflow_id: str, ref: str, secret: SecretStr) -> None:
        self.stored[(workflow_id, ref)] = secret

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        return self.stored.get((workflow_id, ref)) if ref else None

    def purge(self, workflow_id: str) -> None:
        self.purged.append(workflow_id)


def _runtime(vault: _RecordingVault, registry: FakeRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("v2-register-vaulting"),
        secret_vault=cast(Any, vault),
    )


@pytest.mark.anyio
async def test_inline_key_is_vaulted_and_absent_from_every_persisted_surface():
    vault = _RecordingVault()
    registry = FakeRegistry()
    runtime = _runtime(vault, registry)

    workflow_id, results = await runtime.register("owner", "org", _WF, format="native")
    task_id = next(r.task_id for r in results if r.graph_node_name == "solver")

    # The credential was vaulted under its workflow, and a generated ref was minted.
    assert len(vault.stored) == 1
    (vault_wfl, ref), secret = next(iter(vault.stored.items()))
    assert vault_wfl == workflow_id
    assert ref.startswith("msk-")
    assert secret.get_secret_value() == _RAW_KEY

    # The pinned binding carries the generated ref, never the raw key.
    binding = runtime.resolve_model_binding(task_id)
    assert binding is not None
    assert binding.secret_ref == ref

    # The raw key appears in no persisted surface: template/source, task record, ledger.
    blobs = [
        *registry.v2_blobs.values(),
        *registry.task_blobs.values(),
        *registry.ledger_blobs.values(),
    ]
    assert blobs
    assert all(_RAW_KEY not in blob for blob in blobs)

    # The captured source is structurally redacted.
    bundle = PersistedV2Workflow.model_validate_json(registry.v2_blobs[workflow_id])
    assert _RAW_KEY not in bundle.source.raw_payload
    assert "***redacted***" in bundle.source.raw_payload


def test_inspect_does_not_echo_the_raw_inline_key():
    runtime = _runtime(_RecordingVault(), FakeRegistry())
    report = runtime.inspect_v2(_WF, format="native")
    assert report is not None
    assert _RAW_KEY not in report.model_dump_json()


@pytest.mark.anyio
async def test_cancel_purges_the_workflow_vault():
    vault = _RecordingVault()
    runtime = _runtime(vault, FakeRegistry())
    workflow_id, _ = await runtime.register("owner", "org", _WF, format="native")
    runtime.cancel_workflow(workflow_id)
    assert workflow_id in vault.purged


def _drive_agent_to_done(runtime: TaskRuntime, task_id: str) -> None:
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id)
    assert engine is not None and dispatch is not None
    adapter = ScriptedHarnessAdapter([ScriptedStep(op="complete", value="done")], "v1")
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    engine.on_dispatched(task_id, "wkr-1")
    result = adapter.start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    runtime.mark_succeeded(
        task_id,
        "wkr-1",
        {"agent_episode": result.model_dump(mode="json")},
        "2026-08-30T00:00:00Z",
    )


@pytest.mark.anyio
async def test_done_workflow_purges_the_vault():
    vault = _RecordingVault()
    runtime = _runtime(vault, FakeRegistry())
    workflow_id, results = await runtime.register("owner", "org", _WF, format="native")
    task_id = next(r.task_id for r in results if r.graph_node_name == "solver")
    _drive_agent_to_done(runtime, task_id)
    record = runtime.get_record(task_id)
    assert record is not None and record.status is TaskStatus.DONE
    assert workflow_id in vault.purged
