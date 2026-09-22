"""Manifest/carrier schema and content-store find-or-finalize/hydrate behavior."""

import pytest

from shared.content import ContentHydrationError, reference_for
from shared.outcome import (
    InlineControl,
    ManifestRef,
    OutcomeManifest,
)

from .outcome_helpers import InMemoryContentStore


def test_manifest_carries_no_url_or_path() -> None:
    manifest = OutcomeManifest(
        content=reference_for("local", b"hi", media_type="text/plain")
    )
    fields = set(OutcomeManifest.model_fields) | set(
        manifest.content.__class__.model_fields
    )
    assert "url" not in fields and "path" not in fields
    assert "http" not in str(manifest.model_dump())


def test_carrier_round_trips_inline_and_ref() -> None:
    manifest = OutcomeManifest(
        content=reference_for("local", b"x", media_type="application/json")
    )
    ref = ManifestRef(manifest=manifest)
    assert ManifestRef.model_validate_json(ref.model_dump_json()).manifest == manifest
    inline = InlineControl(value="unavailable")
    assert InlineControl.model_validate_json(inline.model_dump_json()) == inline


def test_materialize_is_idempotent_under_idempotency_key() -> None:
    store = InMemoryContentStore()
    first = store.materialize(
        "local", "idm-1", b"payload", media_type="application/json"
    )
    second = store.materialize(
        "local", "idm-1", b"payload", media_type="application/json"
    )
    assert first == second
    assert store.write_count == 1  # the second call found the first, no second write


def test_a_scope_never_resolves_another_scopes_key() -> None:
    store = InMemoryContentStore()
    store.materialize("local", "idm-1", b"payload", media_type="application/json")
    assert store.find("other", "idm-1") is None


def test_hydrate_verifies_digest() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize(
        "local", "idm-2", b"body", media_type="application/json"
    )
    assert store.hydrate(manifest.content) == b"body"


def test_hydrate_raises_on_missing_content() -> None:
    store = InMemoryContentStore()
    with pytest.raises(ContentHydrationError):
        store.hydrate(reference_for("local", b"other", media_type="text/plain"))


def test_hydrate_raises_on_digest_mismatch() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize(
        "local", "idm-4", b"body", media_type="application/json"
    )
    tampered = manifest.content.model_copy(
        update={"content_digest": reference_for("local", b"other").content_digest}
    )
    with pytest.raises(ContentHydrationError):
        store.hydrate(tampered)


def test_hydrate_raises_on_size_mismatch() -> None:
    store = InMemoryContentStore()
    manifest = store.materialize(
        "local", "idm-5", b"body", media_type="application/json"
    )
    with pytest.raises(ContentHydrationError):
        store.hydrate(manifest.content.model_copy(update={"size_bytes": 999}))


def test_a_reference_names_one_object_per_scope() -> None:
    store = InMemoryContentStore()
    here = store.write("local", b"same", media_type="text/plain")
    there = store.write("other", b"same", media_type="text/plain")
    assert here.content_digest == there.content_digest
    assert here.identity != there.identity
    assert store.write_count == 2  # equal bytes in two scopes are two objects
