"""The declared result an inference leaf reports whichever embodiment ran it.

A leaf that admits several embodiments stores one result shape, so a consumer reading
its output — including a guard branching on it — cannot tell which embodiment produced
it. The projection carries what the leaf declares: the pinned model, its prompt, and the
generated output. Fields only one embodiment can report are dropped rather than passed
through, since a value present under one and absent under the other is exactly what
makes the two distinguishable.

It runs where the fabric already owns the stored result, so the worker executes an
ordinary typed task and never learns which embodiment it is.
"""

import logging
from pathlib import Path
from typing import Any

from shared.inference import (
    CanonicalProjectionError,
    canonical_request,
    canonical_result,
)
from shared.schemas.result.catalog import ResultEnvelope
from shared.schemas.result.io import result_file_path, write_result
from shared.tasks.specs import InferenceSpecStrict, InferenceSpecTemplate

_LOG = logging.getLogger("inference-projection")


def project_menu_result(results_dir: Path, task_id: str, spec: Any) -> bool:
    """Rewrite a menu leaf's stored result into its declared shape.

    Idempotent in both senses a settlement path needs: re-running it over an already
    projected result reproduces that result, and it does nothing until the result is
    there to project, so whichever of the settlement and the result's own arrival
    completes last performs the projection.
    """
    if not isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
        return False
    try:
        request = canonical_request(spec)
    except CanonicalProjectionError:
        return False
    path = result_file_path(results_dir, task_id)
    if not path.exists():
        return False
    try:
        envelope = ResultEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        _LOG.warning("[fabric] unreadable result for menu leaf %s", task_id)
        return False
    output = _generated_output(envelope.result.model_dump())
    if output is None:
        return False
    write_result(
        results_dir,
        ResultEnvelope(
            task_id=task_id,
            result=canonical_result(request, output),
            metadata=envelope.metadata,
        ),
    )
    return True


def _generated_output(payload: dict[str, Any]) -> str | None:
    """The generated text, read the same way from either embodiment's own result.

    A local generation reports items and a relayed invocation reports the episode's
    terminal value. Reading the already projected shape first is what makes a second
    pass reproduce the first.
    """
    items = payload.get("items")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict) and isinstance(output := first.get("output"), str):
            return output
    value = payload.get("value")
    return value if isinstance(value, str) else None
