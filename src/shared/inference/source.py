"""Where an inference contract's prompts come from, and the record of resolving them.

A leaf names its prompts either as literal items or as one bounded projection of a
declared direct upstream input. Both are one source contract resolved through one seam,
so a literal leaf and an upstream-sourced one reach the same canonical request and every
embodiment of either runs that one request. The descriptor travels with the contract;
the values it resolves to are attempt facts.

An upstream source that declares an envelope is screenable before its value exists: the
envelope bounds the request, a scheduler screens feasibility against it, and a
resolution exceeding it fails as a typed input error. One that declares none is prepared
first and screened against the request it actually produced.
"""

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

# The projection grammar a source is resolved under. A contract records the version it
# was proven against, and a resolver refuses one it does not implement.
INPUT_RESOLVER_VERSION = "1"


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
    same form. An element source names one element of a producer's collection by the
    producer's task and the element's index, and resolves to that one prompt.
    """

    model_config = ConfigDict(frozen=True)

    kind: InferenceSourceKind
    items: tuple[str, ...] = ()
    node: str | None = None
    path: str | None = None
    element: int | None = Field(default=None, ge=0)
    resolver_version: str = INPUT_RESOLVER_VERSION
    # The most prompts a resolution of this source may yield. A literal source resolves
    # to its own items, so it is their count. An upstream source declares it to be
    # screened against before its value exists, or leaves it out and is prepared before
    # anything is screened.
    max_items: int | None = Field(default=None, ge=1)

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
            if not self.node:
                raise ValueError("an upstream source names a node")
            if self.element is None and not self.path:
                raise ValueError("an upstream source names a path or an element")
            if self.element is not None and (self.path or self.max_items != 1):
                raise ValueError(
                    "an element source names no path and resolves to exactly one item"
                )
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_element(self, serializer: Any) -> Any:
        # A path or literal source serializes, and so digests, without the element key.
        dumped = serializer(self)
        if isinstance(dumped, dict) and dumped.get("element") is None:
            dumped.pop("element", None)
        return dumped

    @property
    def prepared_before_selection(self) -> bool:
        """Whether this source is resolved before an embodiment can be screened.

        Without a declared envelope there is nothing to screen a candidate against, so
        the request is materialized first and the choice follows from what it holds.
        """
        return self.kind is InferenceSourceKind.UPSTREAM and self.max_items is None

    @property
    def expression(self) -> str:
        """The projection an upstream source resolves."""
        if self.kind is not InferenceSourceKind.UPSTREAM:
            raise InputResolutionError("a literal source has no upstream expression")
        if self.element is not None:
            return f"{self.node}[{self.element}]"
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


def content_version(value: Any) -> str:
    """The content identity of an upstream value, for a binding's provenance."""
    return _digest(value)


def request_digest(model: str, prompts: tuple[str, ...], params: dict[str, Any]) -> str:
    """The identity of a resolved request, over everything an embodiment issues."""
    return _digest({"model": model, "prompts": list(prompts), "params": params})


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "INPUT_RESOLVER_VERSION",
    "CanonicalInferenceInputSource",
    "InferenceSourceKind",
    "InputResolutionBinding",
    "InputResolutionError",
    "UpstreamProvenance",
    "content_version",
    "request_digest",
]
