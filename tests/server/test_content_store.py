"""Filesystem content store: content-addressing, idempotency, tenant isolation."""

import pytest

from server.services.content_store import ServerContentStore, default_content_root
from shared.outcome import ContentStoreError, OutcomeHydrationError, content_digest


def _store(tmp_path) -> ServerContentStore:
    return ServerContentStore(tmp_path / "content")


def test_materialize_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.materialize("t1", "idm-1", b"payload", media_type="application/json")
    second = store.materialize("t1", "idm-1", b"payload", media_type="application/json")
    assert first == second
    assert first.content_digest == content_digest(b"payload")
    assert store.find("t1", "idm-1") == first


def test_read_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    manifest = store.materialize("t1", "idm-2", b"body", media_type="application/json")
    assert store.read("t1", manifest.content_digest) == b"body"


def test_tenant_isolation_on_read(tmp_path) -> None:
    store = _store(tmp_path)
    manifest = store.materialize("t1", "idm-3", b"body", media_type="application/json")
    with pytest.raises(OutcomeHydrationError):
        store.read("t2", manifest.content_digest)


def test_missing_digest_raises(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(OutcomeHydrationError):
        store.read("t1", content_digest(b"absent"))


def test_unsafe_segment_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ContentStoreError):
        store.find("t1", "../escape")


def test_default_content_root_is_under_data_dir() -> None:
    assert (
        default_content_root("/var/lib/flowmesh")
        .as_posix()
        .endswith("flowmesh/content")
    )
