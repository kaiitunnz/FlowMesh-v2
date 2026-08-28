"""The generic harness-adapter contract below the agent boundary signature.

A harness adapter is a physical binding that drives an agent's local turns for one
backend. It is the reference form a concrete binding maps its own item/tool/call and
injection surface onto: named, fabric-owned facade tools; a stable per-call correlation;
and per-call outcome injection. The fabric persists and dispatches the mediated work and
the engine owns durable request creation; the adapter owns only harness-local session,
checkpoint, export/import, cancellation, and injection mechanics.

The contract is defined here as a harness-agnostic ABC and exercised by a test-double;
concrete backend bindings live behind their own versioned backend key.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ..state import BoundaryEvent, DenialKind


class FacadeTool(StrEnum):
    """A named, fabric-owned facade tool an adapter exposes in place of a native path.

    ``spawn_agent`` replaces native harness delegation: it emits a spawn request the
    engine turns into one attenuated child activation, never a native subagent.
    """

    SPAWN_AGENT = "spawn_agent"


FACADE_TOOLS = frozenset(FacadeTool)


class HarnessBackendKey(BaseModel):
    """A versioned identity for one harness binding.

    ``backend`` names the harness (a test double here; a concrete app-server binding
    later) and ``version`` pins the adapter/protocol so a capsule is only resumed by a
    compatible binding.
    """

    model_config = ConfigDict(frozen=True)

    backend: str
    version: str


class HarnessCapsule(BaseModel):
    """An opaque, durable continuation of a harness session.

    ``blob`` is adapter-owned serialized state the fabric never interprets. ``portable``
    declares whether the capsule is restartable only against local durable state (the
    default) or relocatable across a worker, which needs the exported ``state_ref``
    backing rather than a bare session identifier.
    """

    model_config = ConfigDict(frozen=True)

    backend: HarnessBackendKey
    blob: str
    portable: bool = False
    state_ref: str | None = None


class OutcomeKind(StrEnum):
    """The class of a durably delivered outcome injected back at a call."""

    RESULT = "result"
    DENIED = "denied"
    CANCELLED = "cancelled"


class DeliveredOutcome(BaseModel):
    """A durable outcome injected back into the harness at its originating call.

    ``call_correlation`` is the stable adapter-local id the outcome returns at, and
    ``idempotency_key`` is the fabric-assigned dedupe authority for the mediated effect.
    A ``DENIED`` outcome carries its ``denial`` kind; a ``RESULT`` carries an opaque
    ``value``.
    """

    model_config = ConfigDict(frozen=True)

    call_correlation: str
    idempotency_key: str | None = None
    kind: OutcomeKind = OutcomeKind.RESULT
    denial: DenialKind | None = None
    value: str | None = None


class HarnessResultKind(StrEnum):
    """What a harness step returned."""

    COMPLETION = "completion"
    FAILURE = "failure"
    CANCELLATION = "cancellation"
    YIELD = "yield"
    BOUNDARY = "boundary"  # a typed boundary request emitted before it executes


class HarnessResult(BaseModel):
    """The result of one harness step.

    A ``BOUNDARY`` (or ``YIELD``) carries the ``request`` the adapter deferred before it
    executed and the ``capsule`` to resume from; a terminal kind carries ``value`` or
    ``error``.
    """

    model_config = ConfigDict(frozen=True)

    kind: HarnessResultKind
    request: BoundaryEvent | None = None
    capsule: HarnessCapsule | None = None
    value: str | None = None
    error: str | None = None


class HarnessAdapter(ABC):
    """The physical binding that drives one agent's local turns for a backend.

    An adapter starts or resumes an activation from an opaque capsule plus durably
    delivered outcomes, and returns the next step: completion, failure, cancellation, a
    yield, or a typed boundary request emitted before that request executes. The engine
    validates and creates the durable request; the adapter never reaches a raw tool,
    endpoint, or native subagent — ``spawn_agent`` is a facade and native bypass paths
    are disabled in a supported configuration.
    """

    @abstractmethod
    def backend_key(self) -> HarnessBackendKey:
        """The versioned backend key this adapter binds."""

    @abstractmethod
    def start(
        self,
        activation_id: str,
        *,
        capsule: HarnessCapsule | None,
        outcomes: Sequence[DeliveredOutcome],
    ) -> HarnessResult:
        """Start (``capsule`` None) or resume an activation, injecting the outcomes.

        A sampled harness is never assumed to replay deterministically: a resume
        restores the capsule's checkpoint or re-drives only from the recorded
        decision-relevant outcomes, never a blind re-execution.
        """

    @abstractmethod
    def cancel(self, activation_id: str) -> None:
        """Cancel the harness session for an activation."""

    def bypass_disabled(self) -> bool:
        """Whether native tool and subagent bypass paths are disabled.

        A supported configuration returns True: every side effect flows through a
        fabric-owned facade, so no raw tool, endpoint, or native subagent escapes the
        engine's validation.
        """
        return True

    def export_state(self, activation_id: str) -> str | None:
        """Export activation-private harness state for a relocatable capsule.

        The default None marks a local-only capsule that restarts against persisted
        local state; a relocatable binding overrides this with a portable state export.
        """
        return None

    def import_state(self, activation_id: str, state: str) -> None:
        """Import previously exported harness state before a relocated resume."""
