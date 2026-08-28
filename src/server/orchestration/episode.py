"""The run-to-yield execution contract between a work item and a worker attempt.

An episode consumes inputs and state references and returns an :class:`EpisodeOutcome`:
completion, a cancellation-safe outcome, failure, or a durable tagged
:class:`BoundaryEvent`. The semantic handling of a boundary event lives in the
orchestration engine; the physical layer carries the event across the episode boundary
and routes it back in.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ..task.v2.representations.operators import BoundaryEventKind
from .state import ValueRef


class EpisodeOutcomeKind(StrEnum):
    """What a run-to-yield episode returned."""

    COMPLETION = "completion"  # the episode ran to completion
    CANCELLATION_SAFE = "cancellation_safe"  # stopped at a declared cancellation point
    FAILURE = "failure"
    BOUNDARY = "boundary"  # yielded a durable tagged boundary event


class BoundaryEvent(BaseModel):
    """A durable tagged event an episode yields at a run-to-yield boundary.

    The fields carried depend on ``kind``: an invocation or effect names an
    ``interface``, a spawn names the ``child_ref`` operator to materialize, a state
    access names a declared ``state_ref``, and a yield carries an opaque
    ``continuation``. ``value_ref`` is any accompanying input reference.
    """

    model_config = ConfigDict(frozen=True)

    kind: BoundaryEventKind
    interface: str | None = None
    child_ref: str | None = None
    state_ref: str | None = None
    continuation: str | None = None
    value_ref: ValueRef | None = None


class EpisodeOutcome(BaseModel):
    """The result a worker attempt returns for one bounded episode."""

    model_config = ConfigDict(frozen=True)

    kind: EpisodeOutcomeKind
    value_ref: ValueRef | None = None
    error: str | None = None
    retryable: bool = True
    event: BoundaryEvent | None = None
