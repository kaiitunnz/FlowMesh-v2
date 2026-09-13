"""The declared result an inference leaf reports when it admits several embodiments.

Both embodiments of a local-eligible leaf report through one projection, so a consumer
reading the leaf's output — including a downstream guard branching on it — sees the same
declared result whichever one the scheduler ran. The projection carries what the leaf
declares: the pinned model, one item, its prompt, and its output. Fields only one
embodiment can produce are dropped rather than passed through, since a value present
under one embodiment and absent under the other is exactly what makes the two
distinguishable.
"""

import json

from shared.harness import HarnessResultKind
from shared.inference import (
    CanonicalInferenceRequest,
    CanonicalProjectionError,
    canonical_request,
    canonical_result,
)
from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.catalog import InferenceResult
from shared.tasks.specs import InferenceSpecStrict, TaskSpecStrictBase
from shared.tasks.worker_message import WorkerTaskMessage

from .base_executor import ExecutionError
from .episode_support import EpisodeStepResult


def canonical_projection(
    spec: TaskSpecStrictBase, task_id: str
) -> CanonicalInferenceRequest:
    """The request projection a leaf's embodiments share, or a controlled failure."""
    if not isinstance(spec, InferenceSpecStrict):
        raise ExecutionError(
            f"{task_id} carries a resolved inference embodiment but no inference spec"
        )
    try:
        return canonical_request(spec)
    except CanonicalProjectionError as exc:
        raise ExecutionError(f"{task_id}: {exc}") from exc


def declared_result(
    msg: WorkerTaskMessage, produced: BaseExecutorResult
) -> BaseExecutorResult:
    """The result to record for this dispatch.

    A leaf bound to one of several embodiments records the shared projection. Every
    other leaf records exactly what its executor produced.
    """
    if msg.embodiment is None:
        return produced
    if isinstance(produced, EpisodeStepResult):
        if produced.harness_result.kind is not HarnessResultKind.COMPLETION:
            return produced
        output = produced.value or ""
    elif isinstance(produced, InferenceResult):
        output = _first_output(produced)
    else:
        return produced
    return canonical_result(canonical_projection(msg.spec, msg.task_id), output)


def _first_output(result: InferenceResult) -> str:
    if not result.items:
        return ""
    output = result.items[0].output
    return output if isinstance(output, str) else json.dumps(output)
