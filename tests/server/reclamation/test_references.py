"""Which fields of a persisted model name content, pinned field by field.

Each expected set below is written by hand, so a new field that can hold a content
reference fails this suite until it is classified here; a reference a binding names
but the walk misses is content the audit reports unbound.
"""

import itertools
import types
import typing
from collections.abc import Mapping
from typing import Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from server.orchestration.ledger_layout import HISTORIES, KEYED
from server.reclamation.references import (
    reference_paths,
    references_of,
)
from server.registries.task_states import PersistedTask
from server.task.history import OutputHistory, TaskHistory
from shared.content import ContentReference, reference_for
from shared.outcome.manifest import OutcomeManifest
from shared.tasks.artifact_binding import ArtifactBinding

_VALUE_REF = {"content", "members[].value_ref@ValueRef"}


def _under(prefix: str, paths: set[str]) -> set[str]:
    return {f"{prefix}{path}" for path in paths}


_KEYED: dict[str, set[str]] = {
    "scopes": set(),
    "activations": set(),
    "work_items": _under("child_input.", _VALUE_REF) | _under("value_ref.", _VALUE_REF),
    "continuations": set(),
    "invocations": set(),
    "attempts": set(),
    "embodiment_selections": set(),
    "input_resolutions": {"reference"},
    "input_preparations": set(),
    "boundary_events": {"outcome_ref.content"} | _under("value_ref.", _VALUE_REF),
    "effect_receipts": set(),
    "delegated_grants": set(),
    "progress_capabilities": set(),
    "result_slots": set(),
    "result_publications": _under("value_ref.", _VALUE_REF),
    "private_state": set(),
    "occurrences": set(),
    "control_states": _under("inputs[].", _VALUE_REF)
    | _under("outputs[].", _VALUE_REF),
    "branch_decisions": _under("input_ref.", _VALUE_REF),
    "loop_instances": _under("carried[].", _VALUE_REF)
    | _under("exit_bundle[].", _VALUE_REF)
    | _under("invariants[].", _VALUE_REF),
    "iteration_resolutions": _under("bundle[].", _VALUE_REF),
    "child_contexts": _under("entries[].", _VALUE_REF) | _under("result.", _VALUE_REF),
}

_HISTORIES: dict[str, set[str]] = {
    "records": _under("value_ref.", _VALUE_REF),
    "accepted_inputs": _under("members[].value_ref.", _VALUE_REF),
    "region_aggregates": _under("members[].value_ref.", _VALUE_REF),
    "authority_decisions": set(),
    "trace": set(),
}

_RESULT_BINDING = {"reference", "members[].binding@ResultBinding"}
_ARTIFACT = {"reference", "largest"}

_MODELS: dict[type[BaseModel], set[str]] = {
    PersistedTask: {"record.internal.result_reference"}
    | _under("record.internal.dispatched_artifacts[].", _ARTIFACT)
    | _under("record.internal.runtime.artifacts[].", _ARTIFACT)
    | _under("record.internal.runtime.unwritten_artifacts[].", _ARTIFACT),
    TaskHistory: _under("result.", _RESULT_BINDING),
    OutputHistory: _under("publication.value_ref.", _VALUE_REF)
    | _under("value.", _RESULT_BINDING),
    OutcomeManifest: {"content"},
    ArtifactBinding: _ARTIFACT,
}

# What one level of a recursive model holds, filled where a path refers back to it.
_RECURSION: dict[str, set[str]] = {
    "ValueRef": {"content"},
    "ResultBinding": {"reference"},
}

_CASES: list[tuple[str, type[BaseModel], set[str]]] = [
    *((f"keyed:{name}", KEYED[name][0], paths) for name, paths in _KEYED.items()),
    *(
        (f"history:{name}", HISTORIES[name], paths)
        for name, paths in _HISTORIES.items()
    ),
    *((model.__name__, model, paths) for model, paths in _MODELS.items()),
]


def test_every_ledger_collection_is_classified() -> None:
    assert set(_KEYED) == set(KEYED)
    assert set(_HISTORIES) == set(HISTORIES)


@pytest.mark.parametrize(
    ("model", "expected"),
    [(model, paths) for _, model, paths in _CASES],
    ids=[name for name, _, _ in _CASES],
)
def test_reference_paths_match_the_classified_inventory(
    model: type[BaseModel], expected: set[str]
) -> None:
    assert reference_paths(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [(model, paths) for _, model, paths in _CASES if paths],
    ids=[name for name, _, paths in _CASES if paths],
)
def test_every_reference_at_a_classified_path_is_found(
    model: type[BaseModel], expected: set[str]
) -> None:
    minted = _Mint()
    value = _build(model, expected, minted)
    found = list(references_of(value))
    assert {r.identity for r in found} == {r.identity for r in minted.references}
    assert len({r.identity for r in found}) == len(minted.references) == len(found)


def test_a_model_reached_back_through_a_cycle_bears_what_the_cycle_reaches() -> None:
    class Inner(BaseModel):
        outer: "Outer | None" = None

    class Outer(BaseModel):
        inner: Inner | None = None
        reference: ContentReference | None = None

    class Holder(BaseModel):
        outer: Outer | None = None

    class Wrapper(BaseModel):
        inner: Inner | None = None

    Inner.model_rebuild()
    reference = reference_for("org", b"x")

    # Deciding Outer first reaches Inner while Outer is still being decided.
    assert list(references_of(Holder(outer=Outer(reference=reference)))) == [reference]
    nested = Wrapper(inner=Inner(outer=Outer(reference=reference)))
    assert list(references_of(nested)) == [reference]
    assert reference_paths(Wrapper) == {
        "inner.outer.reference",
        "inner.outer.inner@Inner",
    }


class _Mint:
    """Distinct references, one per path filled."""

    def __init__(self) -> None:
        self._counter = itertools.count()
        self.references: list[ContentReference] = []

    def __call__(self) -> ContentReference:
        reference = reference_for("org", f"ref-{next(self._counter)}".encode())
        self.references.append(reference)
        return reference


def _build(model: type[BaseModel], paths: set[str], mint: _Mint) -> BaseModel:
    """Construct ``model`` with a distinct reference at every one of ``paths``,
    leaving every other field unset."""
    grouped: dict[str, set[str]] = {}
    for path in paths:
        head, _, rest = path.partition(".")
        grouped.setdefault(head, set())
        if rest:
            grouped[head].add(rest)
    fields: dict[str, Any] = {}
    for head, rests in grouped.items():
        segment, _, recursive = head.partition("@")
        name = segment.removesuffix("[]" * segment.count("[]"))
        containers, nested = _shape(model.model_fields[name].annotation)
        assert len(containers) == segment.count("[]"), head
        if recursive:
            assert nested is not None and nested.__name__ == recursive, head
            value: Any = _build(nested, _RECURSION[recursive], mint)
        elif rests:
            assert nested is not None, head
            value = _build(nested, rests, mint)
        else:
            assert nested is None, head
            value = mint()
        for container in reversed(containers):
            value = {"k": value} if container == "map" else [value]
        fields[name] = value
    return model.model_construct(**fields)


def _shape(annotation: Any) -> tuple[list[str], type[BaseModel] | None]:
    """The containers ``annotation`` nests its value in, outermost first, and the
    model the value is, or None for a reference."""
    containers: list[str] = []
    while True:
        annotation = _plain(annotation)
        origin = get_origin(annotation)
        if origin is None:
            assert isinstance(annotation, type), annotation
            if issubclass(annotation, ContentReference):
                return containers, None
            assert issubclass(annotation, BaseModel), annotation
            return containers, annotation
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        if isinstance(origin, type) and issubclass(origin, Mapping):
            containers.append("map")
            annotation = args[1]
        else:
            containers.append("seq")
            annotation = args[0]


def _plain(annotation: Any) -> Any:
    while True:
        if isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is typing.Annotated:
            annotation = get_args(annotation)[0]
        elif get_origin(annotation) in (Union, types.UnionType):
            (annotation,) = [a for a in get_args(annotation) if a is not type(None)]
        else:
            return annotation
