"""Static acceptance gate for the worker-egress-sidecar cutover.

Proves the server-driven / task-backed fabric-tool carriage is gone: the migration flag,
the ``TOOL_OPERATION`` task path, and the remote-carriage modules no longer exist; the
control broker holds no provider client or egress path; a mediated permit and a worker
message carry no credential; and a ``spawn_agent`` boundary never becomes a sidecar
egress. The behavioral capture -> permit -> sidecar -> fenced-outcome trace, a
fence-reject terminal, and crash/re-drive/cancel custody are covered by the
mediated-egress-sidecar and worker-originated-boundary suites.
"""

import importlib

import pytest

_REMOVED_MODULES = [
    "server.tools.tool_carriage",
    "server.tools.wiring",
    "server.tools.external_tool_sidecar",
    "server.tools.tool_relay_delivery",
    "server.tools.tool_sidecar_wire",
    "server.tools.tool_egress",
    "worker.external_tool_executor",
    "server.supervisor.services.tool_egress_deputy",
    "shared.tools.wire",
]


@pytest.mark.parametrize("module", _REMOVED_MODULES)
def test_remote_carriage_modules_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_migration_flag_is_gone() -> None:
    from server.config import OrchestrationConfig

    assert "worker_originated_boundaries" not in OrchestrationConfig().__dict__


def test_tool_operation_task_path_is_gone() -> None:
    from shared.tasks import specs
    from shared.tasks.task_type import TaskType
    from worker.executors import EXECUTOR_REGISTRY

    assert not any(t.name == "TOOL_OPERATION" for t in TaskType)
    assert not hasattr(specs, "ToolOperationSpecStrict")
    assert "tool_operation" not in EXECUTOR_REGISTRY


def test_control_broker_holds_no_provider_or_egress() -> None:
    from server.config import WebSearchConfig
    from server.tools.fabric_tool_broker import FabricToolBroker

    broker = FabricToolBroker.build(WebSearchConfig(), lambda *a: None)
    # No locality policy, provider client, or egress sidecar on the control broker.
    assert not hasattr(broker, "_policy")
    assert not any("provider" in name.lower() for name in vars(broker))
    assert not any("sidecar" in name.lower() for name in vars(broker))


def test_permit_and_worker_message_carry_no_credential() -> None:
    from shared.tasks.worker_message import WorkerTaskMessage
    from shared.tools.contract import MediatedOperationPermit

    fields = MediatedOperationPermit.model_fields
    assert "api_key" not in fields and "request_payload" not in fields
    # The permit commits to a digest only; the worker message no longer ships a permit.
    assert "request_digest" in fields
    assert "tool_operation" not in WorkerTaskMessage.model_fields


def test_web_search_config_holds_no_credential() -> None:
    from server.config import WebSearchConfig

    assert "api_key" not in WebSearchConfig().__dict__


def test_spawn_boundary_is_never_a_sidecar_egress() -> None:
    from shared.harness import (
        BoundaryEventKind,
        BoundaryRequest,
        HarnessResult,
        HarnessResultKind,
    )
    from worker.executors.agent_episode_executor import AgentEpisodeExecutor

    spawn = HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryRequest(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            interface="lit-review",
            request_payload="{}",
        ),
    )
    assert not AgentEpisodeExecutor._is_capturable_boundary(spawn)
