"""The generic harness-adapter contract below the agent boundary signature.

A harness adapter is a physical binding that drives an agent's local turns for one
backend, running on a worker. It is the reference form a concrete binding maps its own
item/tool/call and injection surface onto: named, fabric-owned facade tools; a stable
per-call correlation; and per-call outcome injection. The fabric persists and dispatches
the mediated work and the engine owns durable request creation; the adapter owns only
harness-local session, checkpoint, export/import, cancellation, and injection mechanics.

The contract is harness-agnostic; concrete backend bindings live behind their own
versioned backend key.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .boundary import BoundaryRequest, DenialKind


class FacadeTool(StrEnum):
    """Named, fabric-owned facade tools an adapter exposes in place of native paths."""

    # Replaces native harness delegation: the engine turns it into one attenuated child
    # activation, never a native subagent.
    SPAWN_AGENT = "spawn_agent"


FACADE_TOOLS = frozenset(FacadeTool)


class MediatedFacade(StrEnum):
    """A capability class the fabric mediates instead of a native harness path.

    An adapter reports which of these it fully mediates (no native bypass path); a
    capability outside the reported set may still run as a native harness tool.
    """

    MODEL = "model"
    SPAWN_AGENT = "spawn_agent"
    SEARCH = "search"


# The set the fabric requires a supported agent backend to mediate: the model boundary,
# child delegation, and web search all cross the fabric's validation, not a native path.
REQUIRED_MEDIATED_FACADES = frozenset(
    {MediatedFacade.MODEL, MediatedFacade.SPAWN_AGENT, MediatedFacade.SEARCH}
)


class HarnessBackendKey(BaseModel):
    """A versioned identity for one harness binding."""

    model_config = ConfigDict(frozen=True)

    backend: str  # the harness binding this key names
    version: str  # pins the adapter/protocol so a capsule resumes only on a match


class HarnessCapsule(BaseModel):
    """An opaque, durable continuation of a harness session."""

    model_config = ConfigDict(frozen=True)

    backend: HarnessBackendKey
    blob: str  # adapter-owned serialized state the fabric never interprets
    # False: restartable only against local durable state; True: relocatable across a
    # worker, which needs the exported state_ref rather than a bare session id.
    portable: bool = False
    state_ref: str | None = None  # exported state backing a relocatable capsule


class OutcomeKind(StrEnum):
    """The class of a durably delivered outcome injected back at a call."""

    RESULT = "result"
    DENIED = "denied"
    CANCELLED = "cancelled"


class DeliveredOutcome(BaseModel):
    """A durable outcome injected back into the harness at its originating call."""

    model_config = ConfigDict(frozen=True)

    call_correlation: str  # the stable adapter-local id the outcome returns at
    idempotency_key: str | None = None  # fabric-assigned dedupe authority
    kind: OutcomeKind = OutcomeKind.RESULT
    denial: DenialKind | None = None  # set when kind is DENIED
    value: str | None = None  # opaque result payload when kind is RESULT


class AgentEpisodeDispatch(BaseModel):
    """The agent-episode context the fabric ships to a worker for one run-to-yield step.

    The backend key selects the adapter; ``capsule_blob`` resumes a prior step (None on
    the first dispatch); ``delivered_outcomes`` are the durable outcomes to inject at
    their originating calls before the adapter steps.
    """

    model_config = ConfigDict(frozen=True)

    backend: HarnessBackendKey
    capsule_blob: str | None = None
    delivered_outcomes: tuple[DeliveredOutcome, ...] = ()


class HarnessResultKind(StrEnum):
    """What a harness step returned."""

    COMPLETION = "completion"
    FAILURE = "failure"
    CANCELLATION = "cancellation"
    YIELD = "yield"
    BOUNDARY = "boundary"  # a typed boundary request emitted before it executes


class HarnessResult(BaseModel):
    """The result of one harness step."""

    model_config = ConfigDict(frozen=True)

    kind: HarnessResultKind
    request: BoundaryRequest | None = None  # the deferred request, for a BOUNDARY/YIELD
    capsule: HarnessCapsule | None = None  # the continuation to resume from
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

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        """The capability classes this backend fully mediates through a fabric facade.

        A supported configuration mediates at least ``REQUIRED_MEDIATED_FACADES``: the
        model, child delegation, and web search each flow through a fabric-owned facade,
        so no raw endpoint, native subagent, or native web search escapes validation. A
        capability outside the set may still run as an ordinary native harness tool.
        """
        return REQUIRED_MEDIATED_FACADES

    def export_state(self, activation_id: str) -> str | None:
        """Export activation-private harness state for a relocatable capsule.

        The default None marks a local-only capsule that restarts against persisted
        local state; a relocatable binding overrides this with a portable state export.
        """
        return None

    def import_state(self, activation_id: str, state: str) -> None:
        """Import previously exported harness state before a relocated resume."""
