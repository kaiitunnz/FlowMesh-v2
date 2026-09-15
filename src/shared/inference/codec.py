"""The canonical inference request and result projection.

An inference leaf whose binding admits more than one embodiment derives its request
from one projection and reports its output in one declared shape, so a resident-served
and a self-contained run of the same leaf are interchangeable to everything downstream.
The projection is deliberately narrow: it accepts only the spec shape both embodiments
already read identically, and rejects anything whose equivalence is unproven.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.result.catalog import InferenceResult
from ..schemas.result.payloads import InferenceItem
from ..tasks.specs import InferenceSpecStrict, InferenceSpecTemplate
from .source import (
    INPUT_RESOLVER_VERSION,
    CanonicalInferenceInputSource,
    InferenceSourceKind,
    InputResolutionBinding,
    InputResolutionError,
    UpstreamProvenance,
    request_digest,
)

_LIST_DATA_TYPE = "list"

# Everything a projectable leaf may declare about its inputs. An allowlist, so a leaf
# naming inputs only one embodiment reads — an object store, an image group, a dataset —
# fails toward no menu instead of toward a silently different request.
_PROJECTABLE_DATA_KEYS = frozenset(
    {"type", "items", "expr", "node", "path", "max_items"}
)

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

    A contract carries the leaf's prompts in declared order, each as its own
    conversation: a chat request serves one conversation, and the engine batches whole
    requests itself.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    prompts: tuple[str, ...]
    params: dict[str, Any] = Field(default_factory=dict)

    def chat_bodies(self) -> tuple[dict[str, Any], ...]:
        """The engine requests this contract issues, whichever embodiment runs it."""
        return tuple(
            {**self.params, "messages": [{"role": "user", "content": prompt}]}
            for prompt in self.prompts
        )


class CanonicalInferenceContract(BaseModel):
    """The one contract every embodiment of an inference leaf resolves and runs.

    It pins the model and the effective sampling request, and names the source its
    prompts come from. It travels with the leaf; the prompts it resolves to are produced
    once, where the upstream value is in hand.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    source: CanonicalInferenceInputSource
    params: dict[str, Any] = Field(default_factory=dict)


class ResolvedCanonicalInferenceRequest(BaseModel):
    """One materialized contract: the request to issue, and the record of resolving it.

    Both embodiments consume this one resolution — the self-contained entry runs the
    request locally and the resident entry serializes the same request across its
    service boundary — so their inputs are identical by construction.
    """

    model_config = ConfigDict(frozen=True)

    request: CanonicalInferenceRequest
    binding: InputResolutionBinding


def resolve_contract(
    contract: CanonicalInferenceContract,
    projected: Any,
    upstream: Sequence[UpstreamProvenance] = (),
) -> ResolvedCanonicalInferenceRequest:
    """Materialize a contract's prompt vector into the request its embodiments run.

    ``projected`` is what the contract's source projected out of the pinned upstream
    snapshot, and is unread for a literal source, which carries its own items;
    ``upstream`` is the provenance of the inputs that projection read. Both kinds are
    validated against the same declared envelope here, so a resolution that exceeds what
    the leaf declared fails as a typed input error before any model I/O and before any
    admission.
    """
    source = contract.source
    if source.resolver_version != INPUT_RESOLVER_VERSION:
        raise InputResolutionError(
            f"the contract declares resolver version {source.resolver_version!r} and "
            f"this resolver implements {INPUT_RESOLVER_VERSION!r}"
        )
    prompts = (
        source.items
        if source.kind is InferenceSourceKind.LITERAL
        else _projected_prompts(source, projected)
    )
    if len(prompts) > source.max_items:
        raise InputResolutionError(
            f"the source resolved {len(prompts)} prompts and declares at most "
            f"{source.max_items}"
        )
    request = CanonicalInferenceRequest(
        model=contract.model, prompts=prompts, params=contract.params
    )
    return ResolvedCanonicalInferenceRequest(
        request=request,
        binding=InputResolutionBinding(
            source_digest=source.digest(),
            resolver_version=source.resolver_version,
            request_digest=request_digest(
                contract.model, prompts, dict(contract.params)
            ),
            cardinality=len(prompts),
            upstream=tuple(upstream),
            projected_output_tokens=_projected_output_tokens(contract, len(prompts)),
        ),
    )


def _projected_prompts(
    source: CanonicalInferenceInputSource, projected: Any
) -> tuple[str, ...]:
    """The prompt vector an upstream projection yielded, or a typed input failure."""
    if projected is None:
        raise InputResolutionError(
            f"{source.expression} resolved nothing; the upstream input a leaf projects "
            "its prompts from must be declared and settled"
        )
    if not isinstance(projected, list) or not projected:
        raise InputResolutionError(
            f"{source.expression} resolved {type(projected).__name__}, and a leaf "
            "projects its prompts from a non-empty list of strings"
        )
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in projected):
        raise InputResolutionError(
            f"{source.expression} resolved a list holding a non-string or empty value, "
            "and a leaf projects its prompts from non-empty strings"
        )
    return tuple(projected)


def _projected_output_tokens(
    contract: CanonicalInferenceContract, cardinality: int
) -> int | None:
    """The whole vector's conservative output-token demand, when the contract bounds
    it."""
    per_prompt = contract.params.get("max_tokens")
    if not isinstance(per_prompt, int) or isinstance(per_prompt, bool):
        return None
    return per_prompt * cardinality


# The fields a chat leaf names a literal list of prompts under, in the order a request
# builder reads them. A ``messages`` array is one multi-turn conversation, not several
# prompts, so it is not one of them.
LIST_PROMPT_FIELDS = ("prompts", "items")


def declares_multiple_prompts(spec: InferenceSpec) -> bool:
    """Whether a leaf declares more than one prompt.

    A chat request serves exactly one conversation, so a leaf declaring several prompts
    needs each of them issued as its own request.
    """
    sources = (
        spec.data if isinstance(spec.data, dict) else {},
        spec.inference if isinstance(spec.inference, dict) else {},
    )
    for source in sources:
        if isinstance(source.get("messages"), list):
            return False
        for field in LIST_PROMPT_FIELDS:
            if isinstance(prompts := source.get(field), list):
                return len(prompts) > 1
    return False


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


def canonical_source(spec: InferenceSpec) -> CanonicalInferenceInputSource:
    """Project a leaf's declared inputs into the source its contract resolves from.

    Two shapes project. Literal prompts under ``spec.data.items`` resolve to exactly
    themselves. One bounded projection of a declared direct upstream input — written as
    ``spec.data.expr``, or as ``spec.data.node`` plus ``spec.data.path`` — resolves to
    the prompt vector that projection yields, within the envelope the leaf declares
    under ``spec.data.max_items``. A dataset, an object store, or an image group is
    rejected rather than projected, because the two embodiments do not read those
    identically.

    This proves the descriptor alone. What an upstream projection will yield is not
    known until the value exists, so nothing here reads one.
    """
    data = spec.data if isinstance(spec.data, dict) else None
    if not data:
        raise CanonicalProjectionError("the leaf declares no spec.data inputs")
    if (declared := data.get("type")) != _LIST_DATA_TYPE:
        raise CanonicalProjectionError(
            f"spec.data.type {declared!r} is not projectable; declare "
            f"type: {_LIST_DATA_TYPE} with literal items or one upstream projection"
        )
    if extra := tuple(sorted(key for key in data if key not in _PROJECTABLE_DATA_KEYS)):
        raise CanonicalProjectionError(
            f"spec.data declares {', '.join(extra)}; a projectable leaf resolves its "
            "prompts from literal items or one bounded upstream projection"
        )
    if (items := data.get("items")) is not None:
        return _literal_source(data, items)
    return _upstream_source(data)


def _literal_source(data: dict[str, Any], items: Any) -> CanonicalInferenceInputSource:
    if not isinstance(items, list) or not items:
        raise CanonicalProjectionError(
            "spec.data.items must be a non-empty literal list"
        )
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in items):
        raise CanonicalProjectionError(
            "spec.data.items must hold literal non-empty strings"
        )
    if any(key in data for key in ("expr", "node", "path")):
        raise CanonicalProjectionError(
            "spec.data declares literal items and an upstream projection; a leaf "
            "resolves its prompts from exactly one source"
        )
    return CanonicalInferenceInputSource(
        kind=InferenceSourceKind.LITERAL,
        items=tuple(items),
        max_items=len(items),
    )


def _upstream_source(data: dict[str, Any]) -> CanonicalInferenceInputSource:
    node, path = _normalized_projection(data)
    if (max_items := _positive_int(data, "max_items")) is None:
        raise CanonicalProjectionError(
            "an upstream projection must declare spec.data.max_items; the admission "
            "capacity its embodiments are screened against is bounded before the "
            "upstream value exists"
        )
    return CanonicalInferenceInputSource(
        kind=InferenceSourceKind.UPSTREAM,
        node=node,
        path=path,
        max_items=max_items,
    )


def _normalized_projection(data: dict[str, Any]) -> tuple[str, str]:
    """The upstream node and path a leaf's projection names, however it wrote them."""
    node, path = data.get("node"), data.get("path")
    if (expr := data.get("expr")) is not None:
        if node is not None or path is not None:
            raise CanonicalProjectionError(
                "spec.data declares both expr and node/path; declare one projection"
            )
        if not isinstance(expr, str) or "." not in expr.strip():
            raise CanonicalProjectionError(
                "spec.data.expr must project a path out of one upstream node, as "
                "'<node>.<path>'"
            )
        node, _, path = expr.strip().partition(".")
    if not isinstance(node, str) or not node.strip():
        raise CanonicalProjectionError(
            "spec.data declares no prompts; declare literal items, or an upstream "
            "projection as expr or node plus path"
        )
    if not isinstance(path, str) or not path.strip():
        raise CanonicalProjectionError(
            "spec.data.node must be projected by a spec.data.path; the whole result of "
            "an upstream node is not a prompt vector"
        )
    return node.strip(), path.strip()


def _positive_int(data: dict[str, Any], key: str) -> int | None:
    if (value := data.get(key)) is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CanonicalProjectionError(f"spec.data.{key} must be a positive integer")
    return value


def canonical_contract(spec: InferenceSpec) -> CanonicalInferenceContract:
    """The contract a leaf's embodiments resolve and run, proven from the spec alone."""
    model = (spec.model_name or "").strip()
    if not model:
        raise CanonicalProjectionError("the leaf declares no model source")
    return CanonicalInferenceContract(
        model=model, source=canonical_source(spec), params=canonical_sampling(spec)
    )


def canonical_result(
    request: CanonicalInferenceRequest, outputs: Sequence[str]
) -> InferenceResult:
    """Report completions in the declared result shape of an inference leaf.

    It carries what the leaf declares — the pinned model, one item per declared prompt
    in declared order — and leaves the fields in ``PROJECTION_DROPS`` unset.
    """
    if len(outputs) != len(request.prompts):
        raise CanonicalProjectionError(
            f"the contract declares {len(request.prompts)} prompts and the run "
            f"reported {len(outputs)} outputs"
        )
    return InferenceResult(
        model=request.model,
        items=[
            InferenceItem(index=index, prompt=prompt, output=output)
            for index, (prompt, output) in enumerate(zip(request.prompts, outputs))
        ],
    )
