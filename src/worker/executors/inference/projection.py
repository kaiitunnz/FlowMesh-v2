"""Read the generated texts out of an inference leaf's own result.

A leaf whose contract the fabric resolves reports one result shape wherever it ran, so a
consumer cannot tell which embodiment produced it. Reading that result is worker-side
work: the two shapes it arrives in are the ones the embodiments produce here — a local
generation's ``InferenceResult`` and a relayed invocation's ``EpisodeStepResult`` — and
each is read as itself rather than as an untyped payload. A result of any other shape
declares nothing, so the run stores what it already reported.
"""

import json
from typing import cast

from shared.inference import CanonicalInferenceRequest
from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.catalog import InferenceResult

from ..episode_support import EpisodeStepResult


def generated_outputs(
    result: BaseExecutorResult, request: CanonicalInferenceRequest
) -> list[str] | None:
    """The generated texts, read the same way from either embodiment's own result.

    Reading the already declared shape is what makes a second pass over a projected
    result reproduce it. The contract's own prompt count decides how a relayed value
    reads, so a single completion is always taken verbatim and never mistaken for a
    batch because the model happened to generate a JSON array.
    """
    if isinstance(result, InferenceResult):
        return _declared_items(result, len(request.prompts))
    if isinstance(result, EpisodeStepResult):
        return _terminal_value(result.value, len(request.prompts))
    return None


def _declared_items(result: InferenceResult, expected: int) -> list[str] | None:
    """The outputs of an already-declared result, when it declares this contract.

    An item's output is polymorphic — a structured value under a template schema, or a
    group of outputs — and only generated text projects, so anything else reports
    nothing rather than being rendered into one.
    """
    if len(result.items) != expected:
        return None
    outputs = [item.output for item in result.items]
    if not all(isinstance(output, str) for output in outputs):
        return None
    return cast(list[str], outputs)


def _terminal_value(value: str | None, expected: int) -> list[str] | None:
    """The completions an episode's terminal value carries for this contract."""
    if value is None:
        return None
    if expected == 1:
        return [value]
    return _batch_value(value, expected)


def _batch_value(value: str, expected: int) -> list[str] | None:
    """Read a batch invocation's terminal value: one completion per declared prompt."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or len(decoded) != expected:
        return None
    return (
        cast(list[str], decoded)
        if all(isinstance(output, str) for output in decoded)
        else None
    )
