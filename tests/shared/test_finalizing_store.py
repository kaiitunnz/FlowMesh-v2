"""Outcome finalization split across the shared store and the control-plane index."""

import pytest

from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    ContentStoreError,
    FabricObjectStore,
    StoreRefusal,
    reference_for,
)
from shared.outcome import FinalizingContentStore, OutcomeManifest
from shared.outcome.finalizing_store import FinalizationRefusedError


class _SharedStore(FabricObjectStore):
    """The durable store, holding bytes for every scope."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.writes = 0

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        self.objects[(scope, reference.content_digest)] = data
        self.writes += 1
        return reference

    def fetch(self, reference: ContentReference) -> bytes:
        key = (reference.authorization_scope, reference.content_digest)
        if key not in self.objects:
            raise ContentHydrationError(f"no content for {reference.content_digest}")
        return self.objects[key]

    def for_scope(self, scope: str) -> FabricObjectStore:
        return self


class _Index:
    """The control plane's finalization bindings."""

    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], OutcomeManifest] = {}

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        return self.bindings.get((scope, idempotency_key))

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        key = (scope, idempotency_key)
        if key not in self.bindings:
            self.bindings[key] = OutcomeManifest(
                content=content, idempotency_key=idempotency_key
            )
        return self.bindings[key]


def _store() -> tuple[FinalizingContentStore, _SharedStore, _Index]:
    shared, index = _SharedStore(), _Index()
    return FinalizingContentStore(shared, index), shared, index  # type: ignore[arg-type]


def test_a_materialized_outcome_is_in_the_shared_store_before_it_is_bound() -> None:
    store, shared, index = _store()
    manifest = store.materialize(
        "local", "idm-1", b"result", media_type="application/json"
    )
    assert shared.objects[("local", manifest.content.content_digest)] == b"result"
    assert index.find("local", "idm-1") == manifest


def test_a_redrive_settles_as_the_first_materialization() -> None:
    store, shared, _ = _store()
    first = store.materialize("local", "idm-2", b"sampled", media_type="text/plain")
    second = store.materialize("local", "idm-2", b"resampled", media_type="text/plain")
    assert first == second
    assert shared.writes == 1  # the second drive never reached the store


def test_an_outcome_hydrates_from_the_shared_store() -> None:
    store, _, _ = _store()
    manifest = store.materialize(
        "local", "idm-3", b"result", media_type="application/json"
    )
    assert store.hydrate(manifest.content) == b"result"


def test_a_scope_never_resolves_another_scopes_key() -> None:
    store, _, _ = _store()
    store.materialize("local", "idm-4", b"result", media_type="application/json")
    assert store.find("other", "idm-4") is None


def test_content_the_shared_store_does_not_have_fails_to_hydrate() -> None:
    store, _, _ = _store()
    with pytest.raises(ContentStoreError):
        store.hydrate(reference_for("local", b"never written"))


class _RefusingIndex(_Index):
    """An index that refuses each recording with the dispositions given, in order."""

    def __init__(self, *refusals: str) -> None:
        super().__init__()
        self.refusals = list(refusals)
        self.records = 0

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        self.records += 1
        if self.refusals:
            raise FinalizationRefusedError(StoreRefusal(self.refusals.pop(0)))
        return super().record(scope, idempotency_key, content)


def test_a_refused_recording_records_the_same_bytes_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shared.outcome.finalizing_store.time.sleep", lambda _: None)
    shared = _SharedStore()
    index = _RefusingIndex("retry_retirement", "rematerialize", "retry")
    store = FinalizingContentStore(shared, index)  # type: ignore[arg-type]

    manifest = store.materialize("local", "idm-9", b"once", media_type="text/plain")

    assert manifest.content == reference_for("local", b"once", media_type="text/plain")
    assert index.records == 4
    # Written once, and again only for the refusal that found its copy gone.
    assert shared.writes == 2
