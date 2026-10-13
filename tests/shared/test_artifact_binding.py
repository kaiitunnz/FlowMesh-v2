"""An artifact binding names a revision of one name, and a path resolves to the
binding it falls under."""

import pytest
from pydantic import ValidationError

from shared.content import ContentReference
from shared.tasks.artifact_binding import (
    ArchiveFacts,
    ArtifactBinding,
    ArtifactKind,
    resolve_artifact_path,
    step_revision,
)

_REF = ContentReference(
    authorization_scope="org", content_digest="ab" * 32, size_bytes=3
)


def _binding(name: str, revision: str = "final", **extra: object) -> ArtifactBinding:
    return ArtifactBinding.model_validate(
        {
            "name": name,
            "revision": revision,
            "kind": ArtifactKind.TREE,
            "reference": _REF,
            "size_bytes": 3,
            **extra,
        }
    )


@pytest.mark.parametrize("revision", ["latest", "step-", "step-01", "final2"])
def test_a_bad_revision_is_refused(revision: str) -> None:
    with pytest.raises(ValidationError):
        _binding("a", revision)


@pytest.mark.parametrize("name", ["", "/a", "a/../b", "a//b"])
def test_a_name_that_is_not_plain_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        _binding(name)


def test_an_archive_names_its_tree() -> None:
    with pytest.raises(ValidationError):
        _binding("a.tar.gz", kind=ArtifactKind.ARCHIVE)
    archive = _binding(
        "a.tar.gz",
        kind=ArtifactKind.ARCHIVE,
        of="a",
        archive=ArchiveFacts(sha256="cd" * 32, crc32=1, size_bytes=9, top="a"),
    )
    assert archive.of == "a"


def test_a_path_resolves_to_the_longest_bound_name() -> None:
    outer = _binding("checkpoints")
    inner = _binding("checkpoints/final_model")
    step = _binding("checkpoints", step_revision(100))
    bindings = {b.key: b for b in (outer, inner, step)}

    assert resolve_artifact_path(bindings, "final", "checkpoints/final_model/w") == (
        inner,
        "w",
    )
    assert resolve_artifact_path(bindings, "final", "checkpoints/x") == (outer, "x")
    assert resolve_artifact_path(bindings, "step-100", "checkpoints/c/w") == (
        step,
        "c/w",
    )
    assert resolve_artifact_path(bindings, "step-7", "checkpoints/c") is None


def test_identical_publications_match_and_stamps_do_not_count() -> None:
    first = _binding("a").accepted("dsp-1", "2026-10-14T00:00:00Z")

    assert first.same_publication(_binding("a"))
    assert not first.same_publication(
        _binding("a", reference=_REF.model_copy(update={"content_digest": "cd" * 32}))
    )
