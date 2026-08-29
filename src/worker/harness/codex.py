"""The version-pinned Codex app-server harness backend.

The Codex binding maps the app-server's dynamic item/tool/call and inject-items surface
onto the generic harness contract: a facade tool the model emits defers as a boundary
before it executes, a delivered outcome is injected back at that call, and a turn that
finishes completes the episode. Native multi-agent mode is off and ``spawn_agent`` is a
facade, so every side effect crosses the fabric's validation.

Durability is a turn-boundary property of the on-disk rollout: a live code-mode cell
is an in-memory lease, not a checkpoint, so recovery is a restart against the same
persisted rollout, reattached through ``thread/resume``. The capsule carries the rollout
metadata and re-drive evidence — dangling call ids and the held facade call — but that
evidence is never the dedupe authority; the fabric-assigned ``idempotency_key`` on an
injected outcome is, so a reissued facade call maps to its key and runs exactly once.
The binding supports one single-forward facade call per defer unit.
"""

from collections.abc import Sequence
from typing import Protocol

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

_BACKEND = "codex"
_CODEX_ADAPTER_VERSION = "v1"

# Facade tools whose boundary is a child-region spawn or seal rather than an invocation.
_SPAWN_TOOLS = {
    "spawn_agent": BoundaryEventKind.SPAWN,
    "spawn_seal": BoundaryEventKind.SPAWN_SEAL,
}


class CodexEvent(BaseModel):
    """One item the app-server emits while running a turn."""

    kind: str  # "facade_call" | "completed" | "error"
    call_id: str | None = (
        None  # the model-visible Codex call id, re-drive evidence only
    )
    tool: str | None = None
    interface: str | None = None
    region: str | None = None
    arguments: str | None = None
    value: str | None = None


class CodexInjectItem(BaseModel):
    """A settled outcome mapped onto the app-server's inject-items surface."""

    call_correlation: str
    idempotency_key: str | None = None
    denied: bool = False
    value: str | None = None


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


class RealCodexAppServerTransport:
    """The seam a live app-server binding fills; unbound here."""

    def thread_start(self) -> str:
        raise NotImplementedError("a live Codex app-server transport is not bound here")

    def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
        raise NotImplementedError("a live Codex app-server transport is not bound here")

    def thread_inject_items(
        self, thread_id: str, items: Sequence[CodexInjectItem]
    ) -> None:
        raise NotImplementedError("a live Codex app-server transport is not bound here")

    def turn_start(self, thread_id: str) -> str:
        raise NotImplementedError("a live Codex app-server transport is not bound here")

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        raise NotImplementedError("a live Codex app-server transport is not bound here")

    def cancel(self, thread_id: str) -> None:
        raise NotImplementedError("a live Codex app-server transport is not bound here")


class _Outstanding(BaseModel):
    """The single held facade call awaiting its injected outcome."""

    call_correlation: str  # the adapter's stable id, kept across a re-drive
    codex_call_id: str | None = None  # the last Codex call id seen, evidence only
    kind: BoundaryEventKind
    interface: str | None = None
    region: str | None = None
    arguments: str | None = None


class _CodexState(BaseModel):
    """The opaque capsule: rollout metadata plus re-drive evidence."""

    thread_id: str
    rollout_ref: str
    forward_index: int = 0
    outstanding: _Outstanding | None = None
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
                )
            )
            if outcome.idempotency_key is not None:
                state.committed_keys.append(outcome.idempotency_key)
            if (
                state.outstanding is not None
                and state.outstanding.call_correlation == outcome.call_correlation
            ):
                # The held call is resolved; the next turn advances past its block.
                state.outstanding = None
        if items:
            self._transport.thread_inject_items(state.thread_id, items)

    def _on_event(self, state: _CodexState, event: CodexEvent) -> HarnessResult:
        capsule = HarnessCapsule(
            backend=self.backend_key(), blob=state.model_dump_json()
        )
        if event.kind == "completed":
            return HarnessResult(
                kind=HarnessResultKind.COMPLETION, value=event.value, capsule=capsule
            )
        if event.kind == "error":
            return HarnessResult(kind=HarnessResultKind.FAILURE, error=event.value)
        if event.kind != "facade_call":
            raise ValueError(f"unexpected Codex event kind {event.kind!r}")
        request = self._defer(state, event)
        capsule = HarnessCapsule(
            backend=self.backend_key(), blob=state.model_dump_json()
        )
        return HarnessResult(
            kind=HarnessResultKind.BOUNDARY, request=request, capsule=capsule
        )

    def _defer(self, state: _CodexState, event: CodexEvent) -> BoundaryRequest:
        kind = _SPAWN_TOOLS.get(event.tool or "", BoundaryEventKind.INVOCATION)
        if state.outstanding is not None:
            if state.outstanding.kind is not kind:
                # A second, different facade call before the first resolves would be a
                # multi-facade block; the single-forward binding does not lift one.
                raise ValueError(
                    "multi-facade code-mode block is unsupported; it needs the "
                    "fabric idempotency-key correlation protocol"
                )
            # A re-drive of the held call reuses its stable correlation; the fresh Codex
            # call id is recorded as evidence, never as the dedupe authority.
            state.outstanding = state.outstanding.model_copy(
                update={"codex_call_id": event.call_id}
            )
            correlation = state.outstanding.call_correlation
        else:
            correlation = f"{state.thread_id}:{state.forward_index}"
            state.outstanding = _Outstanding(
                call_correlation=correlation,
                codex_call_id=event.call_id,
                kind=kind,
                interface=event.interface or event.tool,
                region=event.region,
                arguments=event.arguments,
            )
            state.forward_index += 1
        interface = (
            None if kind in _SPAWN_TOOLS.values() else (event.interface or event.tool)
        )
        return BoundaryRequest(
            kind=kind,
            call_correlation=correlation,
            interface=interface,
            child_region_ref=event.region,
            request_payload=event.arguments,
        )


def build_codex_adapter(
    backend: HarnessBackendKey, task: WorkerTaskMessage, config: WorkerConfig
) -> CodexAppServerHarnessAdapter:
    spec = task.spec
    if not isinstance(spec, AgentSpecStrict) or spec.harness is None:
        raise ValueError("the codex backend requires an agent harness spec")
    return CodexAppServerHarnessAdapter(RealCodexAppServerTransport(), backend.version)
