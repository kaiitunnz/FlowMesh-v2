"""The version-pinned Codex app-server harness backend.

The Codex binding maps the app-server's turn and inject-items surface onto the generic
harness contract: a delivered outcome is injected back at its originating call, and a
turn that finishes completes the episode. A mediated facade originates at the FlowMesh
agent-model gateway — Codex's model provider — which captures the model's native facade
call and clean-completes the turn, so the adapter here only ever completes or fails and
never observes a facade. Native multi-agent mode is off, so every side effect crosses
the fabric's validation.

Durability is a turn-boundary property of the on-disk rollout: recovery is a restart
against the same persisted rollout, reattached through ``thread/resume``. The capsule
carries the rollout metadata and the committed idempotency keys; a re-delivered outcome
maps to its ``idempotency_key`` and injects at most once, so a settled effect never
double-applies on a resume.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from shared.harness import (
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

_BACKEND = "codex"
_CODEX_ADAPTER_VERSION = "v1"


class CodexEvent(BaseModel):
    """The terminal event of one app-server turn: a completion or a failure."""

    kind: str  # "completed" | "error"
    value: str | None = None


class CodexInjectItem(BaseModel):
    """A settled outcome mapped onto the app-server's inject-items surface."""

    call_correlation: str
    idempotency_key: str | None = None
    denied: bool = False
    value: str | None = None
    # When set, the outcome injects back at the model's own call id under its tool name
    # (a captured facade), rather than a synthetic fabric-mediated call.
    injection_target: str | None = None
    injection_tool: str | None = None
    injection_arguments: str | None = None


class CodexAppServerTransport(Protocol):
    """The app-server surface the adapter drives, injectable for a test double."""

    def thread_start(self) -> str: ...
    def thread_resume(self, thread_id: str, rollout_ref: str) -> None: ...
    def thread_inject_items(
        self, thread_id: str, items: Sequence[CodexInjectItem]
    ) -> None: ...
    def turn_start(self, thread_id: str) -> str: ...
    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent: ...
    def cancel(self, thread_id: str) -> None: ...


class _CodexState(BaseModel):
    """The opaque capsule: rollout metadata plus committed inject-dedup keys."""

    thread_id: str
    rollout_ref: str
    committed_keys: list[str] = []


class CodexAppServerHarnessAdapter(HarnessAdapter):
    """Drive a Codex app-server thread as a run-to-yield agent episode."""

    def __init__(
        self,
        transport: CodexAppServerTransport,
        version: str = _CODEX_ADAPTER_VERSION,
    ) -> None:
        self._transport = transport
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
        if capsule is None:
            state = _CodexState(
                thread_id=(tid := self._transport.thread_start()), rollout_ref=tid
            )
        else:
            state = _CodexState.model_validate_json(capsule.blob)
            self._transport.thread_resume(state.thread_id, state.rollout_ref)
        self._inject(state, outcomes)
        turn_id = self._transport.turn_start(state.thread_id)
        event = self._transport.next_event(state.thread_id, turn_id)
        return self._on_event(state, event)

    def cancel(self, activation_id: str) -> None:
        return None

    def _inject(self, state: _CodexState, outcomes: Sequence[DeliveredOutcome]) -> None:
        items: list[CodexInjectItem] = []
        for outcome in outcomes:
            # A re-dispatch after a crash re-ships the same pending outcome; dedupe
            # by the fabric idempotency key so a held effect injects exactly once.
            if (
                outcome.idempotency_key is not None
                and outcome.idempotency_key in state.committed_keys
            ):
                continue
            items.append(
                CodexInjectItem(
                    call_correlation=outcome.call_correlation,
                    idempotency_key=outcome.idempotency_key,
                    denied=outcome.kind is OutcomeKind.DENIED,
                    value=outcome.value,
                    injection_target=outcome.injection_target,
                    injection_tool=outcome.injection_tool,
                    injection_arguments=outcome.injection_arguments,
                )
            )
            if outcome.idempotency_key is not None:
                state.committed_keys.append(outcome.idempotency_key)
        if items:
            self._transport.thread_inject_items(state.thread_id, items)

    def _on_event(self, state: _CodexState, event: CodexEvent) -> HarnessResult:
        if event.kind == "completed":
            capsule = HarnessCapsule(
                backend=self.backend_key(), blob=state.model_dump_json()
            )
            return HarnessResult(
                kind=HarnessResultKind.COMPLETION, value=event.value, capsule=capsule
            )
        if event.kind == "error":
            return HarnessResult(kind=HarnessResultKind.FAILURE, error=event.value)
        raise ValueError(f"unexpected Codex event kind {event.kind!r}")


def build_codex_adapter(
    backend: HarnessBackendKey, task: WorkerTaskMessage, config: WorkerConfig
) -> CodexAppServerHarnessAdapter:
    spec = task.spec
    if not isinstance(spec, AgentSpecStrict) or spec.harness is None:
        raise ValueError("the codex backend requires an agent harness spec")
    params = spec.harness.params
    base_url, model = params.get("base_url"), params.get("model")
    if not isinstance(base_url, str) or not isinstance(model, str):
        raise ValueError(
            "the codex backend requires string 'base_url' and 'model' harness params"
        )
    override = params.get("codex_home")
    codex_home = (
        Path(override)
        if isinstance(override, str)
        else _isolated_codex_home(config.results_dir, task.workflow_id, task.task_id)
    )
    # The live binding pulls in the openai-codex SDK and its bundled app-server binary;
    # keep both off the import path of a worker that never selects the codex backend.
    from .codex_transport import CodexTransportConfig, RealCodexAppServerTransport

    transport = RealCodexAppServerTransport(
        CodexTransportConfig(
            base_url=base_url,
            model=model,
            codex_home=codex_home,
            initial_input=_agent_task(spec),
            task_id=task.task_id,
        )
    )
    return CodexAppServerHarnessAdapter(transport, backend.version)


def _agent_task(spec: AgentSpecStrict) -> str:
    """The first-turn input driving a fresh Codex thread: the agent's task text."""
    if spec.task:
        return spec.task
    data = spec.data or {}
    if isinstance(task := data.get("task"), str) and task:
        return task
    raise ValueError("the codex backend requires 'spec.task' or 'spec.data.task'")


def _isolated_codex_home(results_dir: Path, workflow_id: str, task_id: str) -> Path:
    """The rollout home for one agent activation: isolated per workflow and task.

    ``workflow_id`` and ``task_id`` are stable across an agent's run-to-yield steps and
    restart, so the rollout resumes; distinct activations never co-mingle rollouts, so a
    leaked thread id cannot reattach another activation's thread under a shared home.
    """
    safe = (re.sub(r"[^A-Za-z0-9._-]", "_", part) for part in (workflow_id, task_id))
    return results_dir.joinpath("codex_home", *safe)
