"""The canonical inference request and result projection.

An inference leaf whose binding admits more than one embodiment derives its request
from one projection and reports its output in one declared shape, so a resident-served
and a self-contained run of the same leaf are interchangeable to everything downstream.
The projection is deliberately narrow: it accepts only the spec shape both embodiments
already read identically, and rejects anything whose equivalence is unproven.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.result.catalog import InferenceResult
from ..schemas.result.payloads import InferenceItem
from ..tasks.specs import InferenceSpecStrict, InferenceSpecTemplate

_LIST_DATA_TYPE = "list"

# What a local generation applies when the leaf declares nothing. A resident replica
# would otherwise fall back to its engine's own defaults — generating until the context
# ends where a local run stops at 512 tokens — so an embodiment menu carries these
# explicitly and both embodiments issue one effective request.
SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "min_p": 0.0,
    "max_tokens": 512,
    "min_tokens": 0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "skip_special_tokens": True,
}

# Declared only when the leaf asks for them: an engine applies its own behavior for an
# absent one, and both embodiments read the same absence.
_OPTIONAL_SAMPLING_FIELDS = (
    "stop",
    "seed",
    "n",
    "logprobs",
    "stop_token_ids",
    "bad_words",
)

# Fields a local generation reports and a relayed engine response cannot. The shared
# projection drops them from both embodiments of a leaf: a field present under one and
# absent under the other is what would make the choice observable to a consumer, or to a
# guard branching on the result.
PROJECTION_DROPS = ("finish_reason", "metadata", "usage")

InferenceSpec = InferenceSpecStrict | InferenceSpecTemplate


class CanonicalProjectionError(ValueError):
    """A spec whose request projection is not provably the same for both embodiments."""


class CanonicalInferenceRequest(BaseModel):
    """The one request both embodiments of an inference contract derive from a spec.

    ``params`` is the effective sampling request, not only what the leaf declared: a
    value the author left out still has to be the same on both sides for the two runs
    to be one contract.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    prompt: str
    params: dict[str, Any] = Field(default_factory=dict)

    def chat_body(self) -> dict[str, Any]:
        """The engine request this contract issues, whichever embodiment runs it."""
        return {
            **self.params,
            "messages": [{"role": "user", "content": self.prompt}],
        }


def unforwarded_inference_keys(spec: InferenceSpec) -> tuple[str, ...]:
    """Declared inference settings a relayed request does not carry.

    A local generation reads ``spec.inference`` for more than sampling — guided decoding
    from a declared template, chat-template arguments — and those do not cross to a
    replica. A leaf declaring one is not running one contract on both embodiments.
    """
    declared = spec.inference if isinstance(spec.inference, dict) else {}
    forwarded = set(SAMPLING_DEFAULTS) | set(_OPTIONAL_SAMPLING_FIELDS)
    return tuple(sorted(key for key in declared if key not in forwarded))


def declared_sampling(inference: dict[str, Any]) -> dict[str, Any]:
    """The sampling a leaf declared, restricted to what a generation honours.

    The rest of a leaf's spec names its inputs and its executor, which are not part of
    an engine request.
    """
    forwarded = set(SAMPLING_DEFAULTS) | set(_OPTIONAL_SAMPLING_FIELDS)
    return {
        key: value
        for key, value in inference.items()
        if key in forwarded and value is not None
    }


def canonical_sampling(spec: InferenceSpec) -> dict[str, Any]:
    """The effective sampling request a leaf runs, declared values over the defaults."""
    declared = spec.inference if isinstance(spec.inference, dict) else {}
    params = {**SAMPLING_DEFAULTS}
    for key in SAMPLING_DEFAULTS:
        if key in declared:
            params[key] = declared[key]
    for key in _OPTIONAL_SAMPLING_FIELDS:
        if declared.get(key) is not None:
            params[key] = declared[key]
    return params


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
    return CanonicalInferenceRequest(
        model=model, prompt=prompt, params=canonical_sampling(spec)
    )


def canonical_result(
    request: CanonicalInferenceRequest, output: str
) -> InferenceResult:
    """Report a completion in the declared result shape of an inference leaf.

    It carries what the leaf declares — the pinned model, one item, its prompt, and its
    output — and leaves the fields in ``PROJECTION_DROPS`` unset.
    """
    return InferenceResult(
        model=request.model,
        items=[InferenceItem(index=0, prompt=request.prompt, output=output)],
    )
