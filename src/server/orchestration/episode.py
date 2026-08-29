"""The run-to-yield execution contract between a work item and a worker attempt.

An episode consumes inputs and state references and returns an :class:`EpisodeOutcome`:
completion, a cancellation-safe outcome, failure, or a durable tagged
:class:`BoundaryEvent`. The semantic handling of a boundary event lives in the
orchestration engine; the physical layer carries the event across the episode boundary
and routes it back in.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .state import BoundaryEvent, ValueRef

__all__ = ["BoundaryEvent", "EpisodeOutcome", "EpisodeOutcomeKind"]


class EpisodeOutcomeKind(StrEnum):
    """What a run-to-yield episode returned."""

    COMPLETION = "completion"  # the episode ran to completion
    CANCELLATION_SAFE = "cancellation_safe"  # stopped at a declared cancellation point
    FAILURE = "failure"
    BOUNDARY = "boundary"  # yielded a durable tagged boundary event


class EpisodeOutcome(BaseModel):
    """The result a worker attempt returns for one bounded episode."""

    model_config = ConfigDict(frozen=True)

    kind: EpisodeOutcomeKind
    value_ref: ValueRef | None = None
    error: str | None = None
    retryable: bool = True
    event: BoundaryEvent | None = None
