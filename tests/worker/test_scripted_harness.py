"""The scripted harness backend defers, injects, and resumes from its capsule.

The scripted backend is deterministic: it walks a declared step sequence, defers each
boundary before it executes, resumes a fresh instance purely from the opaque capsule,
and threads a delivered outcome into a later completion — the behavior the worker seam
and the engine boundary path rely on.
"""

import pytest

from shared.harness import (
    BoundaryEventKind,
    DeliveredOutcome,
    HarnessBackendKey,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.harness.scripted import (
    ScriptedHarnessAdapter,
    ScriptedStep,
    build_scripted_adapter,
)

_SCRIPT = [
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.INVOCATION, call="c0", interface="model"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN, call="c1", region="worker"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN_SEAL, call="c2", region="worker"
    ),
    ScriptedStep(op="complete", value_from="c0"),
]


def _adapter() -> ScriptedHarnessAdapter:
    return ScriptedHarnessAdapter(_SCRIPT, "v1")


def test_backend_key_pins_the_version() -> None:
    key = _adapter().backend_key()
    assert key.backend == "scripted" and key.version == "v1"


def test_first_step_defers_a_model_boundary() -> None:
    result = _adapter().start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.BOUNDARY
    assert result.request is not None
    assert result.request.kind is BoundaryEventKind.INVOCATION
    assert result.request.interface == "model"
    # The capsule is the durable continuation — it advanced past the first step.
    assert result.capsule is not None and '"cursor":1' in result.capsule.blob


def test_a_fresh_adapter_resumes_from_the_capsule_and_injects() -> None:
    first = _adapter().start("a", capsule=None, outcomes=[])
    # A distinct instance (a re-dispatch to a fresh process) resumes purely from the
    # capsule and the injected outcome — no in-memory carry-over.
    second = _adapter().start(
        "a",
        capsule=first.capsule,
        outcomes=[DeliveredOutcome(call_correlation="c0", value="ANSWER")],
    )
    assert second.request is not None
    assert second.request.kind is BoundaryEventKind.SPAWN


def test_completion_takes_its_value_from_the_injected_outcome() -> None:
    adapter = _adapter()
    step = adapter.start("a", capsule=None, outcomes=[])
    injected = DeliveredOutcome(call_correlation="c0", value="ANSWER")
    for _ in range(2):  # advance through spawn and spawn_seal
        step = adapter.start("a", capsule=step.capsule, outcomes=[injected])
        injected = DeliveredOutcome(call_correlation="ack", value="")  # only c0 matters
    final = adapter.start("a", capsule=step.capsule, outcomes=[])
    assert final.kind is HarnessResultKind.COMPLETION and final.value == "ANSWER"


def test_denied_outcome_is_recorded_in_the_capsule() -> None:
    script = [ScriptedStep(op="complete", value="ok")]
    adapter = ScriptedHarnessAdapter(script, "v1")
    result = adapter.start(
        "a",
        capsule=None,
        outcomes=[DeliveredOutcome(call_correlation="c0", kind=OutcomeKind.DENIED)],
    )
    assert result.kind is HarnessResultKind.COMPLETION and result.value == "ok"
    assert result.capsule is not None and '"denied":["c0"]' in result.capsule.blob


def test_build_from_spec_reads_the_script() -> None:
    msg = make_worker_task_message(
        {
            "taskType": "agent",
            "harness": {
                "backend": "scripted",
                "version": "v2",
                "params": {
                    "script": [{"op": "complete", "value": "done"}],
                },
            },
        },
        task_type=TaskType.AGENT,
    )
    adapter = build_scripted_adapter(
        msg.agent_episode.backend if msg.agent_episode else _key(),
        msg,
        make_worker_config(),
    )
    assert adapter.backend_key().version == "v2"


def _key() -> HarnessBackendKey:
    return HarnessBackendKey(backend="scripted", version="v2")


def test_build_from_non_agent_spec_is_rejected() -> None:
    msg = make_worker_task_message({"taskType": "echo"}, task_type=TaskType.ECHO)
    with pytest.raises(ValueError, match="agent harness spec"):
        build_scripted_adapter(_key(), msg, make_worker_config())
