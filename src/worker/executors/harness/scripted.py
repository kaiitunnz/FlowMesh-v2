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
    EgressHandoffMode,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    MediatedFacade,
    OutcomeKind,
    sandbox_mediated,
)
from shared.sandbox import (
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxDenied,
)
from shared.tasks.specs import AgentSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from worker.config import WorkerConfig
from worker.model_turn import ResponsesFacade
from worker.private_state import MaterializedState

_BACKEND = "scripted"


class ScriptedStep(BaseModel):
    """One declared move of a scripted agent.

    An ``exec`` step runs a command in the agent's own workspace and records its stdout
    under ``call``, so a later step can complete from it. It is not a boundary: it
    neither ends the dispatch nor produces a request the fabric settles.
    """

    op: Literal["boundary", "complete", "fail", "exec"]
    argv: list[str] | None = None
    timeout_sec: float | None = None
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

    def __init__(
        self,
        script: Sequence[ScriptedStep],
        version: str,
        sandbox: LocalSandboxExecutor | None = None,
    ) -> None:
        self._script = list(script)
        self._version = version
        self._sandbox = sandbox

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend=_BACKEND, version=self._version)

    def egress_handoff_mode(self) -> EgressHandoffMode:
        # Every boundary defers to a capsule and resumes from the committed outcome.
        return EgressHandoffMode.DURABLE_PRE_EGRESS_YIELD

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        return sandbox_mediated(self._sandbox is not None)

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
        while state.cursor < len(self._script):
            step = self._script[state.cursor]
            state.cursor += 1
            if step.op != "exec":
                return self._emit(step, state)
            # Several commands run in one dispatch: a local action is not a boundary, so
            # the episode neither yields its lane nor tells the fabric between them.
            self._run(step, state)
        return HarnessResult(kind=HarnessResultKind.COMPLETION, value=None)

    def cancel(self, activation_id: str) -> None:
        return None

    def _run(self, step: ScriptedStep, state: "_ScriptedState") -> None:
        if self._sandbox is None:
            raise SandboxDenied("this agent declares no sandbox to run a command in")
        if not step.argv:
            raise ValueError("a scripted exec step needs an argv")
        result = self._sandbox.execute(
            SandboxCommand(argv=tuple(step.argv), timeout_sec=step.timeout_sec)
        )
        if step.call is not None:
            state.injected[step.call] = result.stdout.strip()

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
    backend: HarnessBackendKey,
    task: WorkerTaskMessage,
    config: WorkerConfig,
    facade: ResponsesFacade | None = None,
    state: MaterializedState | None = None,
    sandbox: LocalSandboxExecutor | None = None,
) -> ScriptedHarnessAdapter:
    # The scripted backend yields its lane per boundary, so it never binds the facade.
    spec = task.spec
    if not isinstance(spec, AgentSpecStrict) or spec.harness is None:
        raise ValueError("the scripted backend requires an agent harness spec")
    raw = spec.harness.params.get("script")
    if not isinstance(raw, list):
        raise ValueError("the scripted backend requires a 'script' list in its params")
    script = [ScriptedStep.model_validate(item) for item in raw]
    return ScriptedHarnessAdapter(script, backend.version, sandbox)
