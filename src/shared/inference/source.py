"""Where an inference contract's prompts come from, and the record of resolving them.

A leaf names its prompts either as literal items or as one bounded projection of a
declared direct upstream input. Both are one source contract resolved through one seam,
so a literal leaf and an upstream-sourced one reach the same canonical request and every
embodiment of either runs that one request. The descriptor travels with the contract;
the values it resolves to are attempt facts.

The declared envelope is what makes an upstream source safe to admit: it bounds the
request before anything is known about the upstream value, so a scheduler can screen
feasibility and a resolution that exceeds it fails as a typed input error.
"""

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The projection grammar a source is resolved under. A contract records the version it
# was proven against, and a resolver refuses one it does not implement.
INPUT_RESOLVER_VERSION = "1"

# The conservative per-prompt character bound a source carries when its leaf declares
# none. Generous enough for a document-sized prompt, finite enough that an unbounded
# upstream value fails before it reaches an engine.
DEFAULT_MAX_PROMPT_CHARS = 64_000


class InferenceSourceKind(StrEnum):
    LITERAL = "literal"
    UPSTREAM = "upstream"


class InputResolutionError(ValueError):
    """A source that does not resolve to the bounded prompt vector it declares."""


class CanonicalInferenceInputSource(BaseModel):
    """The one input source an inference contract resolves its prompts from.

    A literal source carries its items and resolves to exactly them. An upstream source
    names one declared direct upstream input by node and the path projecting it into a
    prompt vector; ``data.expr`` and ``data.node`` plus ``data.path`` normalize to this
    same form.
    """

    model_config = ConfigDict(frozen=True)

    kind: InferenceSourceKind
    items: tuple[str, ...] = ()
    node: str | None = None
    path: str | None = None
    resolver_version: str = INPUT_RESOLVER_VERSION
    # The most prompts a resolution of this source may yield. For a literal source it is
    # the item count; for an upstream one the leaf declares it, because the admission
    # slots a resident embodiment reserves are screened against it long before a value
    # exists.
    max_items: int = Field(ge=1)
    max_prompt_chars: int = Field(default=DEFAULT_MAX_PROMPT_CHARS, ge=1)

    @model_validator(mode="after")
    def _validate_kind(self) -> "CanonicalInferenceInputSource":
        if self.kind is InferenceSourceKind.LITERAL:
            if not self.items:
                raise ValueError("a literal source carries at least one item")
            if self.node or self.path:
                raise ValueError("a literal source names no upstream node or path")
            if len(self.items) != self.max_items:
                raise ValueError(
                    "a literal source resolves to exactly its items, so its envelope "
                    "is their count"
                )
        else:
            if self.items:
                raise ValueError("an upstream source carries no literal items")
            if not self.node or not self.path:
                raise ValueError("an upstream source names both a node and a path")
        return self

    @property
    def expression(self) -> str:
        """The dotted projection an upstream source resolves."""
        if self.kind is not InferenceSourceKind.UPSTREAM:
            raise InputResolutionError("a literal source has no upstream expression")
        return f"{self.node}.{self.path}"

    def digest(self) -> str:
        """The stable identity of this descriptor, for a contract proof or a binding."""
        return _digest(self.model_dump(mode="json"))


class UpstreamProvenance(BaseModel):
    """The identity of one upstream input a resolution read.

    The content version is what pins the snapshot: a re-drive reading a different
    upstream value carries a different one, so a changed input is visible without the
    value itself ever leaving the worker.
    """

    model_config = ConfigDict(frozen=True)

    node: str
    content_digest: str


class InputResolutionBinding(BaseModel):
    """The record of one source resolution, made before any candidate-specific I/O.

    It carries the identity of what was resolved and what it resolved to — never the
    resolved values. That is what lets a re-drive prove it reached the same request, and
    what lets a resident admission size its credit from the cardinality that actually
    materialized rather than a compile-time guess.
    """

    model_config = ConfigDict(frozen=True)

    source_digest: str
    resolver_version: str
    request_digest: str
    cardinality: int = Field(ge=1)
    # The upstream inputs the resolution read, in declared order. Empty for a literal
    # source, which reads none.
    upstream: tuple[UpstreamProvenance, ...] = ()
    # The conservative output-token demand the resolved vector implies, when its
    # contract declares one.
    projected_output_tokens: int | None = None

    def matches(self, other: "InputResolutionBinding") -> bool:
        """Whether a later resolution reached the same request from the same inputs."""
        return (
            self.source_digest == other.source_digest
            and self.resolver_version == other.resolver_version
            and self.request_digest == other.request_digest
            and self.upstream == other.upstream
        )


def content_version(value: object) -> str:
    """The content identity of an upstream value, for a binding's provenance."""
    return _digest(value)


def request_digest(model: str, prompts: tuple[str, ...], params: dict) -> str:
    """The identity of a resolved request, over everything an embodiment issues."""
    return _digest({"model": model, "prompts": list(prompts), "params": params})


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "DEFAULT_MAX_PROMPT_CHARS",
    "INPUT_RESOLVER_VERSION",
    "CanonicalInferenceInputSource",
    "InferenceSourceKind",
    "InputResolutionBinding",
    "InputResolutionError",
    "UpstreamProvenance",
    "content_version",
    "request_digest",
]
