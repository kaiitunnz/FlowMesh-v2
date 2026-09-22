import pytest

from shared.content import (
    ContentHydrationError,
    ContentReference,
    reference_for,
    verify_content,
)


def test_verified_bytes_are_returned_as_they_are() -> None:
    data = b"verified bytes"
    assert verify_content(reference_for("org-a", data), data) == data


def test_bytes_that_digest_differently_are_refused() -> None:
    reference = reference_for("org-a", b"one object")
    with pytest.raises(ContentHydrationError, match="digest mismatch"):
        verify_content(reference, b"two object")


def test_a_reference_naming_an_unknown_algorithm_is_refused_not_trusted() -> None:
    data = b"bytes under an algorithm this build cannot compute"
    known = reference_for("org-a", data)
    unknown = ContentReference.model_construct(
        **{**known.model_dump(), "digest_algorithm": "blake3"}
    )
    with pytest.raises(ContentHydrationError, match="unsupported digest algorithm"):
        verify_content(unknown, data)
