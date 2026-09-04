"""Manifest/carrier schema and content-store find-or-finalize/hydrate behavior."""

import pytest

from shared.outcome import (
    InlineControl,
    ManifestRef,
    OutcomeHydrationError,
    OutcomeManifest,
    content_digest,
)

from .outcome_helpers import InMemoryContentStore


def test_manifest_carries_no_url_or_path() -> None:
    manifest = OutcomeManifest(
        content_digest=content_digest(b"hi"), size_bytes=2, media_type="text/plain"
    )
    fields = set(OutcomeManifest.model_fields)
    assert "url" not in fields and "path" not in fields
    assert "http" not in str(manifest.model_dump())


def test_carrier_round_trips_inline_and_ref() -> None:
    manifest = OutcomeManifest(
        content_digest=content_digest(b"x"), size_bytes=1, media_type="application/json"
    )
    ref = ManifestRef(manifest=manifest)
    assert ManifestRef.model_validate_json(ref.model_dump_json()).manifest == manifest
    inline = InlineControl(value="unavailable")
    assert InlineControl.model_validate_json(inline.model_dump_json()) == inline


def test_materialize_is_idempotent_under_idempotency_key() -> None:
    store = InMemoryContentStore()
    first = store.materialize("idm-1", b"payload", media_type="application/json")
    second = store.materialize("idm-1", b"payload", media_type="application/json")
    assert first == second
    assert store.write_count == 1  # the second call found the first, no second write


def test_hydrate_verifies_digest() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize("idm-2", b"body", media_type="application/json")
    assert store.hydrate(manifest) == b"body"


def test_hydrate_raises_on_missing_content() -> None:
    store = InMemoryContentStore()
    absent = OutcomeManifest(
        content_digest=content_digest(b"other"), size_bytes=5, media_type="text/plain"
    )
    with pytest.raises(OutcomeHydrationError):
        store.hydrate(absent)


def test_hydrate_raises_on_digest_mismatch() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize("idm-4", b"body", media_type="application/json")
    tampered = manifest.model_copy(update={"content_digest": content_digest(b"other")})
    with pytest.raises(OutcomeHydrationError):
        store.hydrate(tampered)
