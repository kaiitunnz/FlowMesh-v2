"""The canonical inference request and result projection.

An inference leaf whose binding admits more than one embodiment derives its request
from one projection and reports its output in one declared shape, so a resident-served
and a self-contained run of the same leaf are interchangeable to everything downstream.
The projection is deliberately narrow: it accepts only the spec shape both embodiments
already read identically, and rejects anything whose equivalence is unproven.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..schemas.result.catalog import InferenceResult
from ..schemas.result.payloads import InferenceItem
from ..tasks.specs import InferenceSpecStrict, InferenceSpecTemplate

_LIST_DATA_TYPE = "list"

InferenceSpec = InferenceSpecStrict | InferenceSpecTemplate


class CanonicalProjectionError(ValueError):
    """A spec whose request projection is not provably the same for both embodiments."""


class CanonicalInferenceRequest(BaseModel):
    """The one request both embodiments of an inference contract derive from a spec."""

    model_config = ConfigDict(frozen=True)

    model: str
    prompt: str


def canonical_request(spec: InferenceSpec) -> CanonicalInferenceRequest:
    """Project a leaf's declared inputs into the request both embodiments run.

    The accepted shape is a single literal prompt under ``spec.data.items``: a local
    executor reads it as its one prompt and a resident invocation carries it as its one
    user message. A dataset, an upstream expression, an image group, or more than one
    prompt is rejected rather than projected, because the two embodiments do not read
    those identically.
    """
    model = (spec.model_name or "").strip()
    if not model:
        raise CanonicalProjectionError("the leaf declares no model source")
    data = spec.data if isinstance(spec.data, dict) else None
    if not data:
        raise CanonicalProjectionError("the leaf declares no spec.data inputs")
    if (declared := data.get("type")) != _LIST_DATA_TYPE:
        raise CanonicalProjectionError(
            f"spec.data.type {declared!r} is not projectable; declare "
            f"type: {_LIST_DATA_TYPE} with literal items"
        )
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise CanonicalProjectionError(
            "spec.data.items must be a non-empty literal list"
        )
    if len(items) != 1:
        raise CanonicalProjectionError(
            f"a projectable leaf declares exactly one prompt, not {len(items)}"
        )
    prompt = items[0]
    if not isinstance(prompt, str) or not prompt.strip():
        raise CanonicalProjectionError("spec.data.items must hold one literal string")
    if any(data.get(key) is not None for key in ("expr", "node", "path", "s3_cfg")):
        raise CanonicalProjectionError(
            "a projectable leaf resolves its prompt from literal items, not from an "
            "upstream expression or an object store"
        )
    return CanonicalInferenceRequest(model=model, prompt=prompt)


def canonical_result(
    request: CanonicalInferenceRequest,
    output: str,
    finish_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> InferenceResult:
    """Report a completion in the declared result shape of an inference leaf.

    Token accounting is telemetry rather than declared output, so an embodiment that
    does not report it leaves ``usage`` unset without changing what the leaf declares.
    """
    return InferenceResult(
        model=request.model,
        items=[
            InferenceItem(
                index=0,
                prompt=request.prompt,
                output=output,
                finish_reason=finish_reason,
                metadata=metadata,
            )
        ],
    )
