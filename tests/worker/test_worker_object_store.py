"""The worker's held objects: scope isolation and the conservative orphan sweep."""

import os
import time

import pytest

from shared.content import ContentHydrationError
from worker.content import WorkerObjectStore


def _store(tmp_path, grace: float = 60.0) -> WorkerObjectStore:
    return WorkerObjectStore(tmp_path / "content", orphan_grace_sec=grace)


def test_a_written_object_hydrates_by_its_reference(tmp_path) -> None:
    store = _store(tmp_path)
    reference = store.write("local", b"body", media_type="application/json")
    assert store.holds(reference)
    assert store.hydrate(reference) == b"body"


def test_another_scope_does_not_reach_the_object(tmp_path) -> None:
    store = _store(tmp_path)
    reference = store.write("tenant-a", b"body")
    elsewhere = reference.model_copy(update={"authorization_scope": "tenant-b"})
    assert not store.holds(elsewhere)
    with pytest.raises(ContentHydrationError):
        store.hydrate(elsewhere)


def test_a_bound_object_survives_the_sweep(tmp_path) -> None:
    store = _store(tmp_path, grace=0.0)
    reference = store.write("local", b"body")
    store.bind(reference)
    assert store.reclaim_orphans() == 0
    assert store.hydrate(reference) == b"body"


def test_a_binding_survives_a_restart(tmp_path) -> None:
    reference = _store(tmp_path, grace=0.0).write("local", b"body")
    _store(tmp_path, grace=0.0).bind(reference)
    assert _store(tmp_path, grace=0.0).reclaim_orphans() == 0


def test_an_unbound_write_is_reclaimed_after_the_grace(tmp_path) -> None:
    store = _store(tmp_path, grace=0.0)
    reference = store.write("local", b"body")
    assert store.reclaim_orphans() == 1
    assert not store.holds(reference)


def test_an_unbound_write_inside_the_grace_stays(tmp_path) -> None:
    store = _store(tmp_path, grace=3600.0)
    reference = store.write("local", b"body")
    assert store.reclaim_orphans() == 0
    assert store.holds(reference)


def test_an_unbound_write_being_transferred_stays(tmp_path) -> None:
    store = _store(tmp_path, grace=0.0)
    reference = store.write("local", b"body")
    assert store.reclaim_orphans(in_transfer=frozenset({reference.content_digest})) == 0
    assert store.holds(reference)


def test_the_sweep_reads_the_write_time_not_the_call_time(tmp_path) -> None:
    store = _store(tmp_path, grace=30.0)
    reference = store.write("local", b"body")
    path = store._objects.object_path("local", reference.content_digest)
    stale = time.time() - 120
    os.utime(path, (stale, stale))
    assert store.reclaim_orphans() == 1
