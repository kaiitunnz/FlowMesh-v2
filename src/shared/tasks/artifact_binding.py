"""Where a task's published artifacts are, by name and revision.

A task publishes each artifact under a normalized relative name and a revision: the
``final`` set when it succeeds, and a ``step-<n>`` for each checkpoint a training run
saves on the way. The first publication of a name at a revision is kept: the same
bytes published again change nothing, and different bytes are refused. An artifact is
one file, a tree of files named by its manifest, or an archive view of such a tree
whose bytes follow from the tree alone.
"""

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from shared.content import ContentReference, is_member_path
from shared.utils.nofollow import is_plain_segment

FINAL_REVISION = "final"
_STEP_GRAMMAR = r"step-(0|[1-9][0-9]*)"
REVISION_PATTERN = rf"^({FINAL_REVISION}|{_STEP_GRAMMAR})$"
_STEP = re.compile(_STEP_GRAMMAR)
_REVISION = re.compile(REVISION_PATTERN)


def revision_step(revision: str) -> int | None:
    """Return the step a ``step-<n>`` revision names, or None for any other revision."""
    return int(match.group(1)) if (match := _STEP.fullmatch(revision)) else None


def latest_revision(bindings: "Iterable[ArtifactBinding] | None") -> str | None:
    """Return the latest ``step-<n>`` revision among ``bindings``, or None."""
    steps = [
        step
        for binding in bindings or ()
        if (step := revision_step(binding.revision)) is not None
    ]
    return step_revision(max(steps)) if steps else None


def step_revision(step: int) -> str:
    """Return the revision a checkpoint saved at training step ``step`` publishes as."""
    return f"step-{step}"


def is_revision(value: str) -> bool:
    """Return whether ``value`` names a revision: ``final`` or ``step-<n>``."""
    return _REVISION.fullmatch(value) is not None


def _revision(value: str) -> str:
    if not is_revision(value):
        raise ValueError(f"{value!r} is not a revision")
    return value


def _name(value: str) -> str:
    if not is_member_path(value):
        raise ValueError(f"{value!r} is not a relative path of plain segments")
    return value


type ArtifactName = Annotated[str, AfterValidator(_name)]
type Revision = Annotated[str, AfterValidator(_revision)]


class ArtifactKind(StrEnum):
    """What an artifact's reference names."""

    FILE = "file"
    """One file's bytes."""
    TREE = "tree"
    """A tree manifest over the files under the name."""
    ARCHIVE = "archive"
    """A stored-block ``.tar.gz`` view of the tree ``of`` names."""


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{value!r} is not a SHA-256 hex digest")
    return value


def _segment(value: str) -> str:
    if not is_plain_segment(value):
        raise ValueError(f"{value!r} is not one plain path segment")
    return value


def _digested(reference: ContentReference) -> ContentReference:
    _sha256(reference.content_digest)
    return reference


type Sha256 = Annotated[str, AfterValidator(_sha256)]
type DigestedReference = Annotated[ContentReference, AfterValidator(_digested)]


class ArchiveFacts(BaseModel):
    """What a whole archive view checks against, computed when it was published."""

    model_config = ConfigDict(frozen=True)

    sha256: Sha256
    crc32: int = Field(ge=0, lt=1 << 32)
    size_bytes: int = Field(ge=0)
    top: Annotated[str, AfterValidator(_segment)]


class ArtifactBinding(BaseModel):
    """One published artifact: its name, revision and the object naming its bytes."""

    model_config = ConfigDict(frozen=True)

    name: ArtifactName
    """The normalized relative name the artifact publishes under."""
    revision: Revision = FINAL_REVISION
    """The ``final`` set or the ``step-<n>`` checkpoint the artifact belongs to."""
    kind: ArtifactKind
    """What ``reference`` names."""
    reference: DigestedReference
    """The object naming the artifact's bytes."""
    size_bytes: int = Field(ge=0)
    """The artifact's size in bytes, counting a tree's members."""
    files: int = Field(default=1, ge=0)
    """The number of files, counting a tree's members."""
    largest: DigestedReference | None = None
    """A tree's largest member, the object a consumer's placement looks for."""
    of: str | None = None
    """The name of the tree an archive view packs."""
    archive: ArchiveFacts | None = None
    """An archive view's facts."""
    dispatch_id: str | None = None
    """The dispatch the control plane stamps when it accepts the publication."""
    published_at: str | None = None
    """The time the control plane stamps when it accepts the publication."""

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if (self.kind is ArtifactKind.ARCHIVE) != (
            self.of is not None and self.archive is not None
        ):
            raise ValueError("an archive and only an archive names its tree")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.revision, self.name

    def same_publication(self, other: "ArtifactBinding") -> bool:
        """Return whether ``other`` publishes the same bytes under the same name."""
        return (
            self.key == other.key
            and self.kind == other.kind
            and self.reference.identity == other.reference.identity
            and self.of == other.of
            and self.archive == other.archive
        )

    def accepted(self, dispatch_id: str | None, published_at: str) -> Self:
        """Return this publication as the control plane records it."""
        return self.model_copy(
            update={"dispatch_id": dispatch_id, "published_at": published_at}
        )


type ArtifactKey = tuple[str, str]
type ArtifactBindings = dict[ArtifactKey, ArtifactBinding]


def final_set(bindings: ArtifactBindings) -> ArtifactBindings:
    """Return the ``final`` revision of a catalog."""
    return {key: b for key, b in bindings.items() if key[0] == FINAL_REVISION}


def resolve_artifact_path(
    bindings: ArtifactBindings, revision: str, path: str
) -> tuple[ArtifactBinding, str] | None:
    """Return the binding a path at a revision falls under, with the path inside it.

    The binding is the one whose name is ``path`` itself or its longest leading
    directory; the remainder is empty for the binding's own name. A path that is not
    a relative path of plain segments falls under none.
    """
    if not is_member_path(path):
        return None
    parts = path.split("/")
    for cut in range(len(parts), 0, -1):
        name = "/".join(parts[:cut])
        if (binding := bindings.get((revision, name))) is not None:
            return binding, "/".join(parts[cut:])
    return None


def resolve_artifact_input(
    bindings: ArtifactBindings, path: str
) -> tuple[ArtifactBinding, str] | None:
    """Return the final artifact a consumer loads for a ref to ``path``, with the path
    inside it. A model archive loads as the tree it archives, which every loader takes
    as a directory."""
    if (found := resolve_artifact_path(bindings, FINAL_REVISION, path)) is None:
        return None
    binding, inner = found
    if binding.kind is ArtifactKind.ARCHIVE and binding.of is not None:
        binding = bindings[(binding.revision, binding.of)]
    return binding, inner


MODEL_ARTIFACT_NAMES = ("final_model", "final_lora")


def model_artifact(
    bindings: ArtifactBindings,
    producer: str,
    choose: str = "name the artifact to load by filename",
) -> ArtifactBinding:
    """Return the model ``producer`` published: its ``final_model`` or ``final_lora``
    tree.

    Raises ``ValueError`` when its final set holds neither or both, saying how to
    ``choose`` one instead.
    """
    found = [
        binding
        for name in MODEL_ARTIFACT_NAMES
        if (binding := bindings.get((FINAL_REVISION, name))) is not None
        and binding.kind is ArtifactKind.TREE
    ]
    if len(found) != 1:
        held = " and ".join(binding.name for binding in found) or "no model"
        raise ValueError(f"task {producer} published {held}; {choose}")
    return found[0]


ARTIFACT_REF_SCHEME = "flowmesh-artifact:"
# The key is bracketed, so no text written right after a token can extend it.
_TOKEN = re.compile(r"flowmesh-artifact:\[([A-Za-z0-9-]+\.[0-9]+)\]")


class UnknownArtifactToken(ValueError):
    """A spec names an artifact token its dispatch carries no artifact for."""


class ArtifactInputs:
    """The published artifacts one dispatch's rendered spec names, each by a token
    its worker replaces with the local path it hydrates the artifact to.

    Each token is keyed under ``scope``, the dispatch's own id, so text that only
    looks like a token is never replaced.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.bindings: dict[str, ArtifactBinding] = {}
        self._keys: dict[tuple[str, ...], str] = {}

    def mark(self) -> int:
        """Return a mark ``rollback`` forgets every later token back to."""
        return len(self._keys)

    def rollback(self, mark: int) -> None:
        """Forget every token made since ``mark``, as for a spec that does not ship."""
        for identity, key in list(self._keys.items())[mark:]:
            del self._keys[identity]
            del self.bindings[key]

    def token(self, binding: ArtifactBinding, inner: str = "") -> str:
        """Return the token naming ``binding``, and ``inner`` inside it when a tree."""
        # A file lands under its own name, so identical bytes under two names are two
        # inputs; a tree lands by its digest alone.
        name = "" if binding.kind is ArtifactKind.TREE else binding.name
        identity = (*binding.reference.identity, binding.kind, name)
        if (key := self._keys.get(identity)) is None:
            key = self._keys[identity] = f"{self.scope}.{len(self._keys)}"
            self.bindings[key] = binding
        return (
            f"{ARTIFACT_REF_SCHEME}[{key}]/{inner}"
            if inner
            else f"{ARTIFACT_REF_SCHEME}[{key}]"
        )


def substitute_artifact_tokens(value: str, paths: dict[str, str], scope: str) -> str:
    """Replace each token of ``scope`` in ``value`` with the path ``paths`` gives it.

    Raises ``UnknownArtifactToken`` for a token of ``scope`` that ``paths`` lacks.
    """

    def path(match: re.Match[str]) -> str:
        key = match.group(1)
        if (found := paths.get(key)) is not None:
            return found
        if key.startswith(f"{scope}."):
            raise UnknownArtifactToken(f"no artifact input {key} was dispatched")
        return match.group(0)

    return _TOKEN.sub(path, value)


def names_artifact_token(value: str) -> bool:
    return ARTIFACT_REF_SCHEME in value


__all__ = [
    "ARTIFACT_REF_SCHEME",
    "ArtifactInputs",
    "UnknownArtifactToken",
    "names_artifact_token",
    "substitute_artifact_tokens",
    "FINAL_REVISION",
    "ArchiveFacts",
    "ArtifactBinding",
    "ArtifactBindings",
    "ArtifactKey",
    "final_set",
    "ArtifactKind",
    "ArtifactName",
    "Revision",
    "is_revision",
    "resolve_artifact_input",
    "resolve_artifact_path",
    "step_revision",
]
