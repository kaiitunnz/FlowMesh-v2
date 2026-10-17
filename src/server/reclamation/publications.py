"""Holding what a publication names before the commit that binds it.

A publication names the objects its worker stored. Before any of them is bound, each is
held against retirement under one acquisition: an object that is being retired refuses,
as does one retired before whose current backing the store does not confirm. A refused
publication holds nothing, and its worker stores the named objects again from what it
kept. The holds are released once the commit binding the objects is durable, when the
protection that commit adds keeps them instead.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.artifact_upload import ARTIFACTS_PENDING
from shared.tasks.publication import (
    CHECKPOINT_FIELD,
    PREPARED_FIELD,
    PublicationDisposition,
)

from .pins import AcquireMode, Nonces, PinStore, Verdict

# How many times one publication re-verifies objects whose retirement revision moved
# while it verified them.
VERIFY_ROUNDS = 3

type Backed = Callable[[ContentReference], bool | None]


@dataclass
class Assertion:
    """What a publication names, by scope and by how its backing is held."""

    bind: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """Objects the worker stored before reporting them."""
    pending: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """Objects the worker uploads after the report accepts them."""
    manifests: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """Tree manifests whose members are protected once the tree is expanded."""
    references: dict[tuple[str, str], ContentReference] = field(default_factory=dict)

    def add(self, reference: ContentReference, mode: AcquireMode) -> None:
        scope, digest = reference.authorization_scope, reference.content_digest
        self.references[(scope, digest)] = reference
        (self.bind if mode is AcquireMode.BIND else self.pending)[scope].add(digest)

    def add_binding(self, binding: ArtifactBinding, staged: bool) -> None:
        if binding.kind is ArtifactKind.FILE:
            self.add(
                binding.reference, AcquireMode.PENDING if staged else AcquireMode.BIND
            )
            return
        # A tree's manifest is stored before its report even when its files upload
        # after it.
        self.add(binding.reference, AcquireMode.BIND)
        reference = binding.reference
        self.manifests[reference.authorization_scope].add(reference.content_digest)

    def __bool__(self) -> bool:
        return bool(self.references)


def _reference(raw: Any) -> ContentReference | None:
    try:
        return ContentReference.model_validate(raw) if raw is not None else None
    except ValidationError:
        return None


def _bindings(raw: Any) -> list[ArtifactBinding]:
    if not isinstance(raw, list):
        return []
    bindings: list[ArtifactBinding] = []
    for item in raw:
        try:
            bindings.append(ArtifactBinding.model_validate(item))
        except ValidationError:
            continue
    return bindings


def assertion_of(event_type: str, payload: dict[str, Any]) -> Assertion:
    """Return what a task event's publication names."""
    assertion = Assertion()
    if event_type == "TASK_UPDATE":
        for binding in _bindings(payload.get(CHECKPOINT_FIELD)):
            assertion.add_binding(binding, staged=False)
        return assertion
    if event_type != "TASK_SUCCEEDED":
        return assertion
    if isinstance(prepared := payload.get(PREPARED_FIELD), dict):
        if (reference := _reference(prepared.get("reference"))) is not None:
            assertion.add(reference, AcquireMode.BIND)
        return assertion
    staged = bool(payload.get(ARTIFACTS_PENDING))
    if (reference := _reference(payload.get("result_reference"))) is not None:
        assertion.add(reference, AcquireMode.BIND)
    children = payload.get("child_result_references")
    for raw in children.values() if isinstance(children, dict) else ():
        if (reference := _reference(raw)) is not None:
            assertion.add(reference, AcquireMode.BIND)
    bindings = _bindings(payload.get("artifacts"))
    child_artifacts = payload.get("child_artifacts")
    for raw in child_artifacts.values() if isinstance(child_artifacts, dict) else ():
        bindings.extend(_bindings(raw))
    for binding in bindings:
        assertion.add_binding(binding, staged)
    return assertion


@dataclass(frozen=True)
class Holds:
    """The holds one acquisition took, which only its own release ends."""

    nonce: str
    held: dict[str, tuple[str, ...]]

    def __bool__(self) -> bool:
        return any(self.held.values())


@dataclass(frozen=True)
class Preparation:
    """The outcome of holding a publication's objects: its holds, or why it is
    refused."""

    holds: Holds | None
    refusal: PublicationDisposition | None = None
    reason: str | None = None
    digests: tuple[str, ...] = ()


class PublicationHolds:
    """Holds a publication's objects for the commit binding them."""

    def __init__(
        self,
        pins: PinStore,
        nonces: Callable[[], Nonces | None],
        backed: Backed,
        logger: logging.Logger | None = None,
    ) -> None:
        self._pins = pins
        self._nonces = nonces
        self._backed = backed
        self._logger = logger or logging.getLogger("publication-holds")

    def acquire(self, assertion: Assertion, locator: dict[str, Any]) -> Preparation:
        """Hold every object ``assertion`` names, or none of them."""
        nonces = self._nonces()
        if nonces is None:
            return Preparation(
                None, PublicationDisposition.RETRY, "this root holds no epoch yet"
            )
        holds = Holds(nonces.next(), {})
        try:
            refused = self._acquire(assertion, locator, holds)
        except Exception as exc:
            self._logger.warning("Holding a publication's objects failed: %s", exc)
            self.release(holds)
            return Preparation(
                None, PublicationDisposition.RETRY, f"the control store failed: {exc}"
            )
        if refused is not None:
            self.release(holds)
            return refused
        return Preparation(holds)

    def _acquire(
        self, assertion: Assertion, locator: dict[str, Any], holds: Holds
    ) -> Preparation | None:
        scopes = sorted(set(assertion.bind) | set(assertion.pending))
        for scope in scopes:
            for mode, digests in (
                (AcquireMode.BIND, assertion.bind.get(scope, ())),
                (AcquireMode.PENDING, assertion.pending.get(scope, ())),
            ):
                if not digests:
                    continue
                manifests = (
                    sorted(assertion.manifests.get(scope, ()))
                    if mode is AcquireMode.BIND
                    else []
                )
                if (
                    refused := self._acquire_scope(
                        assertion,
                        scope,
                        sorted(digests),
                        mode,
                        manifests,
                        locator,
                        holds,
                    )
                ) is not None:
                    return refused
        return None

    def _acquire_scope(
        self,
        assertion: Assertion,
        scope: str,
        digests: list[str],
        mode: AcquireMode,
        manifests: list[str],
        locator: dict[str, Any],
        holds: Holds,
    ) -> Preparation | None:
        verified: dict[str, str] = {}
        for _ in range(VERIFY_ROUNDS):
            acquisition = self._pins.acquire(
                scope, digests, holds.nonce, locator, mode, verified, manifests
            )
            match acquisition.verdict:
                case Verdict.ACQUIRED:
                    holds.held[scope] = holds.held.get(scope, ()) + acquisition.held
                    return None
                case Verdict.RETIRING:
                    return Preparation(
                        None,
                        PublicationDisposition.RETRY_RETIREMENT,
                        "a named object is being removed from the store",
                        acquisition.retiring,
                    )
            missing: list[str] = []
            for digest, revision in acquisition.unverified.items():
                match self._backed(assertion.references[(scope, digest)]):
                    case True:
                        verified[digest] = revision
                    case False:
                        missing.append(digest)
                    case None:
                        return Preparation(
                            None,
                            PublicationDisposition.RETRY,
                            "the store cannot confirm a named object now",
                        )
            if missing:
                return Preparation(
                    None,
                    PublicationDisposition.REMATERIALIZE,
                    "a named object is not in the store",
                    tuple(missing),
                )
        return Preparation(
            None,
            PublicationDisposition.RETRY,
            "a named object's removal kept moving while it was verified",
        )

    def release(self, holds: Holds) -> None:
        """Release exactly ``holds``; raises while the control store is unreachable."""
        for scope, digests in holds.held.items():
            if digests:
                self._pins.release(scope, digests, holds.nonce)


__all__ = [
    "Assertion",
    "Holds",
    "Preparation",
    "PublicationHolds",
    "assertion_of",
]
