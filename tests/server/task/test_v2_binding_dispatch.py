import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import AgentBindingConfig, OrchestrationConfig
from server.task.runtime import TaskRuntime
from shared.tasks.specs import ModelBindingMode
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _WorkerRegistryStub,
)

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: bind}
spec:
  taskType: echo
  graph:
    nodes:
      - name: solver
        spec:
          taskType: agent
          v2: {authority: {invoke: [model], delegate: []}, tools: [{name: model}]}
          harness: {backend: scripted, version: v9, params: {script: []}}
          model_binding: {mode: canned}
"""


def _runtime() -> TaskRuntime:
    # A deployment default that differs from the source binding: the pin must ignore it.
    config = OrchestrationConfig(
        agent_binding=AgentBindingConfig(
            default_backend="codex",
            default_mode=ModelBindingMode.OPENAI,
            default_url="https://deployment/v1",
        )
    )
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        config,
        Path(tempfile.gettempdir()),
        logging.getLogger("v2-binding-dispatch"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


@pytest.mark.anyio
async def test_dispatch_and_model_binding_read_the_source_pin_not_the_default():
    runtime = _runtime()
    _, results = await runtime.register("owner", "org", _WF, format="native")
    task_id = next(r.task_id for r in results if r.graph_node_name == "solver")

    binding = runtime.resolve_model_binding(task_id)
    assert binding is not None
    # Source model binding wins over the deployment openai default (pin is inert).
    assert binding.mode is ModelBindingMode.CANNED

    dispatch = runtime.agent_episode_dispatch(task_id)
    assert dispatch is not None
    assert dispatch.backend.backend == "scripted"
    assert dispatch.backend.version == "v9"
