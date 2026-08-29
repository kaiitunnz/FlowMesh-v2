"""A deterministic scripted harness backend.

The scripted backend drives an agent through a declared sequence of run-to-yield steps:
each step either defers a boundary before it executes or terminates the episode. It is a
legitimate harness binding — it exposes the fabric-owned facade, defers per call,
resumes purely from its opaque capsule, and injects delivered outcomes — so it exercises
the same worker seam and engine boundary path a live backend does, deterministically and
without credentials.

The step sequence lives in the agent's ``harness.params['script']``; a completion may
take its value from an injected outcome, proving the outcome reached the agent.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    DeliveredOutcome,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks.specs import AgentSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from worker.config import WorkerConfig

_BACKEND = "scripted"


class ScriptedStep(BaseModel):
    """One declared move of a scripted agent."""

    op: Literal["boundary", "complete", "fail"]
    kind: BoundaryEventKind | None = None
    call: str | None = None
    interface: str | None = None
    region: str | None = None
    payload: str | None = None
    value: str | None = None
    value_from: str | None = None  # take the value from this call's injected outcome
    error: str | None = None


class _ScriptedState(BaseModel):
    """The durable capsule: how far the script advanced and what was injected."""

    cursor: int = 0
    injected: dict[str, str] = {}
    denied: list[str] = []


class ScriptedHarnessAdapter(HarnessAdapter):
    """Replay a declared script, deferring each boundary and resuming from a capsule."""

    def __init__(self, script: Sequence[ScriptedStep], version: str) -> None:
        self._script = list(script)
        self._version = version

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend=_BACKEND, version=self._version)

    def start(
        self,
        activation_id: str,
        *,
        capsule: HarnessCapsule | None,
        outcomes: Sequence[DeliveredOutcome],
    ) -> HarnessResult:
        state = (
            _ScriptedState.model_validate_json(capsule.blob)
            if capsule is not None
            else _ScriptedState()
        )
        for outcome in outcomes:
            if outcome.kind is OutcomeKind.DENIED:
                state.denied.append(outcome.call_correlation)
            elif outcome.value is not None:
                state.injected[outcome.call_correlation] = outcome.value
        if state.cursor >= len(self._script):
            return HarnessResult(kind=HarnessResultKind.COMPLETION, value=None)
        step = self._script[state.cursor]
        state.cursor += 1
        return self._emit(step, state)

    def cancel(self, activation_id: str) -> None:
        return None

    def _emit(self, step: ScriptedStep, state: _ScriptedState) -> HarnessResult:
        capsule = HarnessCapsule(
            backend=self.backend_key(), blob=state.model_dump_json()
        )
        if step.op == "boundary":
            if step.kind is None or step.call is None:
                raise ValueError("a scripted boundary needs a kind and a call")
            request = BoundaryRequest(
                kind=step.kind,
                call_correlation=step.call,
                interface=step.interface,
                child_region_ref=step.region,
                request_payload=step.payload,
            )
            return HarnessResult(
                kind=HarnessResultKind.BOUNDARY, request=request, capsule=capsule
            )
        if step.op == "fail":
            return HarnessResult(kind=HarnessResultKind.FAILURE, error=step.error)
        value = step.value
        if step.value_from is not None:
            value = state.injected.get(step.value_from)
        return HarnessResult(
            kind=HarnessResultKind.COMPLETION, value=value, capsule=capsule
        )


def build_scripted_adapter(
    backend: HarnessBackendKey, task: WorkerTaskMessage, config: WorkerConfig
) -> ScriptedHarnessAdapter:
    spec = task.spec
    if not isinstance(spec, AgentSpecStrict) or spec.harness is None:
        raise ValueError("the scripted backend requires an agent harness spec")
    raw = spec.harness.params.get("script")
    if not isinstance(raw, list):
        raise ValueError("the scripted backend requires a 'script' list in its params")
    script = [ScriptedStep.model_validate(item) for item in raw]
    return ScriptedHarnessAdapter(script, backend.version)
