"""The generic harness contract is worker-safe and its transport round-trips.

A harness adapter runs on a worker, which packages only ``shared`` (never ``server``).
These prove the contract imports without dragging in the server, that the agent-episode
dispatch context survives the ``WorkerTaskMessage`` dedup-JSON round-trip, and that the
durable boundary envelope carries a settled outcome value.
"""

import subprocess
import sys

from shared.harness import (
    AgentEpisodeDispatch,
    BoundaryEventKind,
    BoundaryRequest,
    DeliveredOutcome,
    HarnessBackendKey,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks import TaskType
from tests.worker.factories import make_worker_task_message


def test_harness_contract_imports_without_server() -> None:
    # The worker image has no ``server`` package, so the shared contract must not pull
    # it in transitively.
    code = "import shared.harness, sys; assert 'server' not in sys.modules"
    result = subprocess.run(  # nosec B603 - fixed argv, no shell, sys.executable
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr


def test_agent_episode_dispatch_round_trips_on_worker_message() -> None:
    dispatch = AgentEpisodeDispatch(
        backend=HarnessBackendKey(backend="scripted", version="v1"),
        capsule_blob="after:c0",
        delivered_outcomes=(
            DeliveredOutcome(
                call_correlation="c0",
                idempotency_key="idm-1",
                kind=OutcomeKind.RESULT,
                value="answer",
            ),
        ),
    )
    msg = make_worker_task_message(
        {"taskType": "agent"}, task_type=TaskType.AGENT, agent_episode=dispatch
    )
    restored = type(msg).model_validate_json(msg.model_dump_json())
    assert restored.agent_episode is not None
    assert restored.agent_episode.backend.backend == "scripted"
    assert restored.agent_episode.capsule_blob == "after:c0"
    outcome = restored.agent_episode.delivered_outcomes[0]
    assert outcome.value == "answer" and outcome.idempotency_key == "idm-1"


def test_boundary_request_stays_worker_emittable() -> None:
    # A worker emits only the request subset — never the fabric-assigned identity.
    request = BoundaryRequest(
        kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="model"
    )
    result = HarnessResult(kind=HarnessResultKind.BOUNDARY, request=request)
    assert result.request is not None and result.request.interface == "model"
    assert not hasattr(request, "idempotency_key")
    assert not hasattr(request, "activation")
