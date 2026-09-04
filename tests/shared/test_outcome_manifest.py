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
    dumped = manifest.model_dump()
    assert "http" not in str(dumped)


def test_carrier_discriminates_inline_and_ref() -> None:
    manifest = OutcomeManifest(
        content_digest=content_digest(b"x"), size_bytes=1, media_type="application/json"
    )
    inline = InlineControl(value="unavailable")
    ref = ManifestRef(manifest=manifest)
    assert inline.kind == "inline" and ref.kind == "manifest"
    assert ManifestRef.model_validate_json(ref.model_dump_json()).manifest == manifest


def test_materialize_is_idempotent_under_idempotency_key() -> None:
    store = InMemoryContentStore()
    first = store.materialize("t1", "idm-1", b"payload", media_type="application/json")
    second = store.materialize("t1", "idm-1", b"payload", media_type="application/json")
    assert first == second
    assert store.write_count == 1  # the second call found the first, no second write


def test_hydrate_verifies_digest() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize("t1", "idm-2", b"body", media_type="application/json")
    assert store.hydrate(manifest) == b"body"


def test_hydrate_denies_wrong_tenant() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize("t1", "idm-3", b"body", media_type="application/json")
    other = manifest.model_copy(
        update={"access": manifest.access.model_copy(update={"tenant": "t2"})}
    )
    with pytest.raises(OutcomeHydrationError):
        store.hydrate(other)


def test_hydrate_raises_on_digest_mismatch() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize("t1", "idm-4", b"body", media_type="application/json")
    tampered = manifest.model_copy(update={"content_digest": content_digest(b"other")})
    with pytest.raises(OutcomeHydrationError):
        store.hydrate(tampered)
