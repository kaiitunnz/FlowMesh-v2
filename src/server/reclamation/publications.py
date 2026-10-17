"""Holding what a publication names before the commit that binds it.

A publication names the objects its worker stored. Before any of them is bound, each is
held against retirement under one acquisition: an object that is being retired refuses,
as does one retired before whose current backing the store does not confirm. A refused
publication holds nothing, and its worker stores the named objects again from what it
kept. The holds are released once the commit binding the objects is durable, when the
protection that commit adds keeps them instead.
"""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference, StoreRefusal
from shared.content.tree import TreeMember
from shared.tasks.artifact_binding import FINAL_REVISION, ArtifactBinding, ArtifactKind
from shared.tasks.artifact_upload import ARTIFACTS_PENDING
from shared.tasks.publication import (
    CHECKPOINT_FIELD,
    PREPARED_FIELD,
)

from ..clients.redis import REDIS_CONN_ERRORS
from .pins import AcquireMode, Nonces, PinStore, Verdict
from .retirement import Backed
from .trees import TreeExpander, TreeOutcome, TreeStatus

# How many times one publication re-verifies objects whose retirement revision moved
# while it verified them.
VERIFY_ROUNDS = 3


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
    trees: dict[tuple[str, str], bool] = field(default_factory=dict)
    """Each tree manifest, and whether its members are stored before the report
    (write-through) rather than uploaded after it."""

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
        key = (reference.authorization_scope, reference.content_digest)
        self.manifests[key[0]].add(key[1])
        self.trees[key] = self.trees.get(key, False) or not staged

    def tree_references(self, written_through: bool) -> list[ContentReference]:
        return [
            self.references[key]
            for key, through in sorted(self.trees.items())
            if through is written_through
        ]

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


def _reference_in(
    references: dict[tuple[str, str], ContentReference], scope: str, digest: str
) -> ContentReference:
    return references[(scope, digest)]


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
        assertion.add_binding(binding, staged and binding.revision == FINAL_REVISION)
    return assertion


@dataclass(frozen=True)
class Holds:
    """The holds one acquisition took, which only its own release ends."""

    nonce: str
    held: dict[str, tuple[str, ...]]
    trees: list[ContentReference] = field(default_factory=list)
    """Write-through trees whose members this acquisition holds; their holds are
    found again by reading the manifest, so they are not kept in ``held``."""
    staged: list[ContentReference] = field(default_factory=list)
    """Trees whose members are protected once their binding is durable."""

    def __bool__(self) -> bool:
        return any(self.held.values()) or bool(self.trees) or bool(self.staged)


@dataclass(frozen=True)
class Preparation:
    """The outcome of holding a publication's objects: its holds, or why it is
    refused."""

    holds: Holds | None
    refusal: StoreRefusal | None = None
    reason: str | None = None
    digests: tuple[str, ...] = ()
    malformed: bool = False
    """It names a tree whose stored manifest is not a well-formed tree."""


class PublicationHolds:
    """Holds a publication's objects for the commit binding them, and protects the
    members of the trees it binds once that commit is durable."""

    def __init__(
        self,
        pins: PinStore,
        nonces: Callable[[], Nonces | None],
        backed: Backed,
        trees: TreeExpander,
        logger: logging.Logger | None = None,
    ) -> None:
        self._pins = pins
        self._nonces = nonces
        self._backed = backed
        self._logger = logger or logging.getLogger("publication-holds")
        self._trees = trees
        self._owed: list[Holds] = []
        """Holds whose release waits for the store or the control store."""
        self._owed_lock = threading.Lock()

    def acquire(self, assertion: Assertion, locator: dict[str, Any]) -> Preparation:
        """Hold every object ``assertion`` names, write-through trees' members
        included, or none of them."""
        nonces = self._nonces()
        if nonces is None:
            return Preparation(None, StoreRefusal.RETRY, "this root holds no epoch yet")
        holds = Holds(nonces.next(), {})
        try:
            through = self._unexpanded(assertion.tree_references(True))
            holds.staged.extend(assertion.tree_references(False))
            refused = self._acquire(assertion, locator, holds)
            for manifest in through if refused is None else ():
                if (refused := self._hold_tree(manifest, locator, holds)) is not None:
                    break
        except Exception as exc:
            self._logger.warning("Holding a publication's objects failed: %s", exc)
            self._release_quietly(holds)
            return Preparation(
                None, StoreRefusal.RETRY, f"the control store failed: {exc}"
            )
        if refused is not None:
            self._release_quietly(holds)
            return refused
        return Preparation(holds)

    def _unexpanded(self, manifests: list[ContentReference]) -> list[ContentReference]:
        """Return the manifests whose members are not already protected."""
        if not manifests:
            return []
        expanded = self._trees.expanded(manifests)
        return [m for m in manifests if m.content_digest not in expanded]

    def _hold_tree(
        self, manifest: ContentReference, locator: dict[str, Any], holds: Holds
    ) -> Preparation | None:
        holds.trees.append(manifest)
        refusal: list[Preparation] = []

        def hold_page(scope: str, members: list[TreeMember]) -> TreeOutcome | None:
            references = {m.digest: m.reference(scope) for m in members}
            refused = self._acquire_scope(
                references.__getitem__,
                scope,
                list(references),
                AcquireMode.BIND,
                [],
                locator,
                holds,
                record=False,
            )
            if refused is None:
                return None
            refusal.append(refused)
            return TreeOutcome(TreeStatus.REFUSED)

        outcome = self._trees.hold(manifest, hold_page)
        match outcome.status:
            case TreeStatus.DONE:
                return None
            case TreeStatus.REFUSED:
                return refusal[0]
            case TreeStatus.MALFORMED:
                return Preparation(
                    None,
                    None,
                    f"tree {manifest.content_digest} is malformed: {outcome.reason}",
                    (manifest.content_digest,),
                    malformed=True,
                )
        return Preparation(
            None,
            StoreRefusal.RETRY,
            f"tree {manifest.content_digest} cannot be read now: {outcome.reason}",
        )

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
                        partial(_reference_in, assertion.references, scope),
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
        reference_of: Callable[[str], ContentReference],
        scope: str,
        digests: list[str],
        mode: AcquireMode,
        manifests: list[str],
        locator: dict[str, Any],
        holds: Holds,
        record: bool = True,
    ) -> Preparation | None:
        verified: dict[str, str] = {}
        moved = 0
        while True:
            try:
                acquisition = self._pins.acquire(
                    scope, digests, holds.nonce, locator, mode, verified, manifests
                )
            except Exception:
                if record:
                    holds.held[scope] = holds.held.get(scope, ()) + tuple(digests)
                raise
            match acquisition.verdict:
                case Verdict.ACQUIRED:
                    if record:
                        holds.held[scope] = holds.held.get(scope, ()) + acquisition.held
                    return None
                case Verdict.RETIRING:
                    return Preparation(
                        None,
                        StoreRefusal.RETRY_RETIREMENT,
                        "a named object is being removed from the store",
                        acquisition.retiring,
                    )
            if any(d in verified for d in acquisition.unverified):
                moved += 1
                if moved > VERIFY_ROUNDS:
                    return Preparation(
                        None,
                        StoreRefusal.RETRY,
                        "a named object's removal kept moving while it was verified",
                    )
            missing: list[str] = []
            for digest, revision in acquisition.unverified.items():
                match self._backed(reference_of(digest)):
                    case True:
                        verified[digest] = revision
                    case False:
                        missing.append(digest)
                    case None:
                        return Preparation(
                            None,
                            StoreRefusal.RETRY,
                            "the store cannot confirm a named object now",
                        )
            if missing:
                return Preparation(
                    None,
                    StoreRefusal.REMATERIALIZE,
                    "a named object is not in the store",
                    tuple(missing),
                )

    def release(self, holds: Holds) -> None:
        """Release exactly ``holds``; raises while the control store is unreachable.

        A tree whose manifest cannot be read now has its release owed, and repaid by
        a later ``repay``.
        """
        if (rest := self._settle(holds)) is not None:
            self._owe(rest)

    def repay(self) -> int:
        """Release the holds an outage left owed; returns how many are still owed."""
        with self._owed_lock:
            owed, self._owed = self._owed, []
        still: list[Holds] = []
        done = 0
        try:
            for holds in owed:
                try:
                    if (rest := self._settle(holds)) is not None:
                        still.append(rest)
                except REDIS_CONN_ERRORS:
                    still.append(holds)
                done += 1
        finally:
            with self._owed_lock:
                self._owed.extend(still + owed[done:])
        with self._owed_lock:
            return len(self._owed)

    def _settle(self, holds: Holds) -> Holds | None:
        """Release ``holds`` as far as the store reads; returns the tree holds whose
        manifest could not be read."""
        unread = [
            manifest
            for manifest in holds.trees
            if self._trees.release(manifest, holds.nonce, self._pins.release).status
            is TreeStatus.UNAVAILABLE
        ]
        for scope, digests in holds.held.items():
            if digests:
                self._pins.release(scope, digests, holds.nonce)
        return Holds(holds.nonce, {}, unread) if unread else None

    def _owe(self, holds: Holds) -> None:
        with self._owed_lock:
            self._owed.append(holds)

    def _release_quietly(self, holds: Holds) -> None:
        try:
            self.release(holds)
        except REDIS_CONN_ERRORS as exc:
            self._logger.warning("Releasing holds waits for the control store: %s", exc)
            self._owe(holds)
        except Exception as exc:
            # An earlier root's holds are released by the next one; this root's
            # stay until it is replaced.
            self._logger.warning("Could not release holds: %s", exc)

    def finish(
        self, holds: Holds, bound: frozenset[str], locator: dict[str, Any]
    ) -> list[tuple[ContentReference, TreeOutcome]]:
        """Protect the members of each tree the durable commit bound and end every
        hold; returns the trees that did not finish, and the staged trees that did
        with members retired before, which raises nothing.

        A write-through tree's held members are protected; a staged tree is read and
        its members protected now. A tree the commit did not bind has its holds
        released instead.
        """
        unfinished: list[tuple[ContentReference, TreeOutcome]] = []
        self.repay()
        for manifest in holds.trees:
            outcome = (
                self._trees.promote(manifest, holds.nonce)
                if manifest.content_digest in bound
                else self._trees.release(manifest, holds.nonce, self._pins.release)
            )
            if outcome.status is not TreeStatus.DONE:
                unfinished.append((manifest, outcome))
        for manifest in holds.staged:
            if manifest.content_digest in bound:
                outcome = self.expand(manifest, locator)
                if outcome.status is not TreeStatus.DONE or outcome.retired:
                    unfinished.append((manifest, outcome))
        if not any(o.status is TreeStatus.UNAVAILABLE for _, o in unfinished):
            for scope, digests in holds.held.items():
                if digests:
                    self._pins.release(scope, digests, holds.nonce)
        return unfinished

    def expand(
        self, manifest: ContentReference, locator: dict[str, Any], staged: bool = True
    ) -> TreeOutcome:
        """Protect the members a bound tree names: hold them under the expansion's
        own nonce while its manifest verifies, then protect what that held.

        A member being retired stops the expansion: a ``staged`` tree, whose files may
        still be uploading, waits for the retirement to end; any other tree's bytes
        were stored when it was bound, so its scope is un-certified.
        """
        nonces = self._nonces()
        if nonces is None:
            return TreeOutcome(TreeStatus.UNAVAILABLE, "this root holds no epoch yet")
        self.repay()
        nonce = nonces.next()
        retired: list[ContentReference] = []

        def hold_page(scope: str, members: list[TreeMember]) -> TreeOutcome | None:
            acquisition = self._pins.acquire(
                scope, [m.digest for m in members], nonce, locator, AcquireMode.MEMBER
            )
            if acquisition.verdict is Verdict.RETIRING:
                return TreeOutcome(
                    TreeStatus.RETIRING,
                    "a member is being removed from the store",
                    retiring=acquisition.retiring,
                )
            if acquisition.retired:
                by_digest = {m.digest: m for m in members}
                retired.extend(
                    by_digest[d].reference(scope) for d in acquisition.retired
                )
            return None

        outcome = TreeOutcome(TreeStatus.UNAVAILABLE, "the expansion did not finish")
        try:
            outcome = self._trees.hold(manifest, hold_page)
            if outcome.status is TreeStatus.DONE:
                promoted = self._trees.promote(manifest, nonce)
                if promoted.status is TreeStatus.DONE:
                    return TreeOutcome(TreeStatus.DONE, retired=tuple(retired))
                outcome = promoted
            elif outcome.status is TreeStatus.RETIRING and not staged:
                self._trees.uncertify(manifest, outcome.retiring)
        except REDIS_CONN_ERRORS as exc:
            outcome = TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        finally:
            if outcome.status is not TreeStatus.DONE:
                self._release_quietly(Holds(nonce, {}, [manifest]))
        return outcome


__all__ = [
    "Assertion",
    "Holds",
    "Preparation",
    "PublicationHolds",
    "assertion_of",
]
