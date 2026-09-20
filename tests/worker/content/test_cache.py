"""The worker's content cache: scope isolation and age-based eviction."""

import os
import time

import pytest

from shared.content import ContentHydrationError
from worker.content import WorkerContentCache


def _cache(tmp_path, retain: float = 60.0) -> WorkerContentCache:
    return WorkerContentCache(tmp_path / "content", retain_sec=retain)


def test_a_written_object_hydrates_by_its_reference(tmp_path) -> None:
    cache = _cache(tmp_path)
    reference = cache.write("local", b"body", media_type="application/json")
    assert cache.holds(reference)
    assert cache.hydrate(reference) == b"body"


def test_another_scope_does_not_reach_the_object(tmp_path) -> None:
    cache = _cache(tmp_path)
    reference = cache.write("tenant-a", b"body")
    elsewhere = reference.model_copy(update={"authorization_scope": "tenant-b"})
    assert not cache.holds(elsewhere)
    with pytest.raises(ContentHydrationError):
        cache.hydrate(elsewhere)


def test_a_copy_past_the_retention_window_is_evicted(tmp_path) -> None:
    cache = _cache(tmp_path, retain=0.0)
    reference = cache.write("local", b"body")
    assert cache.evict_aged() == 1
    assert not cache.holds(reference)


def test_a_copy_inside_the_retention_window_stays(tmp_path) -> None:
    cache = _cache(tmp_path, retain=3600.0)
    reference = cache.write("local", b"body")
    assert cache.evict_aged() == 0
    assert cache.holds(reference)


def test_a_copy_being_transferred_stays(tmp_path) -> None:
    cache = _cache(tmp_path, retain=0.0)
    reference = cache.write("local", b"body")
    assert cache.evict_aged(in_transfer=frozenset({reference.content_digest})) == 0
    assert cache.holds(reference)


def test_eviction_reads_the_cached_time_not_the_call_time(tmp_path) -> None:
    cache = _cache(tmp_path, retain=30.0)
    reference = cache.write("local", b"body")
    path = cache._objects.object_path("local", reference.content_digest)
    stale = time.time() - 120
    os.utime(path, (stale, stale))
    assert cache.evict_aged() == 1
