"""The worker's content cache: scope isolation and its age and disk bounds."""

import os
import time
from pathlib import Path

import pytest

from shared.content import ContentHydrationError, ContentReference
from worker.content import WorkerContentCache
from worker.content.custody import ArtifactCustody


def _cache(tmp_path, **bounds) -> WorkerContentCache:
    return WorkerContentCache(tmp_path / "content", **bounds)


def _used(cache: WorkerContentCache, reference: ContentReference, ago: float) -> None:
    """Make a copy look as though it was last used ``ago`` seconds back."""
    path = cache._objects.object_path(
        reference.authorization_scope, reference.content_digest
    )
    at = time.time() - ago
    os.utime(path, (at, at))


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


def test_an_unbounded_cache_keeps_every_copy_indefinitely(tmp_path) -> None:
    cache = _cache(tmp_path)
    reference = cache.write("local", b"body")
    _used(cache, reference, ago=10 * 365 * 86400)
    assert cache.evict() == 0
    assert cache.holds(reference)


def test_a_copy_unused_past_the_retention_window_is_evicted(tmp_path) -> None:
    cache = _cache(tmp_path, retain_sec=30.0)
    reference = cache.write("local", b"body")
    _used(cache, reference, ago=120)
    assert cache.evict() == 1
    assert not cache.holds(reference)


def test_a_copy_inside_the_retention_window_stays(tmp_path) -> None:
    cache = _cache(tmp_path, retain_sec=3600.0)
    reference = cache.write("local", b"body")
    assert cache.evict() == 0
    assert cache.holds(reference)


def test_reading_a_copy_renews_it_against_the_retention_window(tmp_path) -> None:
    cache = _cache(tmp_path, retain_sec=30.0)
    reference = cache.write("local", b"body")
    _used(cache, reference, ago=120)
    assert cache.hydrate(reference) == b"body"
    assert cache.evict() == 0
    assert cache.holds(reference)


def test_past_the_disk_budget_the_least_recently_used_copy_goes(tmp_path) -> None:
    cache = _cache(tmp_path, max_bytes=10)
    oldest = cache.write("local", b"aaaaaa")
    newest = cache.write("local", b"bbbbbb")
    _used(cache, oldest, ago=60)
    _used(cache, newest, ago=30)
    assert cache.evict() == 1
    assert not cache.holds(oldest)
    assert cache.holds(newest)


def test_the_disk_budget_ranks_by_last_use_not_by_write(tmp_path) -> None:
    cache = _cache(tmp_path, max_bytes=10)
    written_first = cache.write("local", b"aaaaaa")
    written_second = cache.write("local", b"bbbbbb")
    _used(cache, written_first, ago=60)
    _used(cache, written_second, ago=30)
    cache.hydrate(written_first)
    assert cache.evict() == 1
    assert cache.holds(written_first)
    assert not cache.holds(written_second)


def test_a_cache_within_its_disk_budget_keeps_everything(tmp_path) -> None:
    cache = _cache(tmp_path, max_bytes=100)
    first = cache.write("local", b"aaaaaa")
    second = cache.write("local", b"bbbbbb")
    assert cache.evict() == 0
    assert cache.holds(first) and cache.holds(second)


def test_a_copy_being_transferred_stays_under_either_bound(tmp_path) -> None:
    cache = _cache(tmp_path, retain_sec=30.0, max_bytes=1)
    reference = cache.write("local", b"body")
    _used(cache, reference, ago=120)
    assert cache.evict(in_transfer=frozenset({reference.content_digest})) == 0
    assert cache.holds(reference)


def _object(cache: WorkerContentCache, reference: ContentReference) -> Path:
    return cache._objects.object_path(
        reference.authorization_scope, reference.content_digest
    )


def test_a_copy_a_task_output_also_links_is_evicted_past_the_disk_budget(
    tmp_path,
) -> None:
    cache = _cache(tmp_path, max_bytes=1)
    reference = cache.write("local", b"oversized")
    output = tmp_path / "task-output"
    os.link(_object(cache, reference), output)

    assert cache.evict() == 1
    assert not cache.holds(reference)
    assert output.read_bytes() == b"oversized"


def test_a_copy_custody_holds_stays_whichever_inode_custody_staged(
    tmp_path,
) -> None:
    cache = _cache(tmp_path, max_bytes=1)
    reference = cache.write("local", b"oversized")
    staged = (
        ArtifactCustody(tmp_path / "content").publication_dir("local", "dsp-2")
        / reference.content_digest
    )
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"oversized")

    assert cache.evict() == 0
    assert cache.holds(reference)

    staged.unlink()
    assert cache.evict() == 1


def test_a_copy_replaced_since_the_walk_is_not_evicted(tmp_path, monkeypatch) -> None:
    cache = _cache(tmp_path, max_bytes=1)
    reference = cache.write("local", b"oversized")
    path = _object(cache, reference)
    walked = path.stat()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"oversized")
    os.utime(replacement, ns=(walked.st_atime_ns, walked.st_mtime_ns))
    real_stat = cache._objects.stat
    calls = 0

    def stat_then_replace(scope: str, digest: str) -> os.stat_result | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.replace(replacement, path)
        return real_stat(scope, digest)

    monkeypatch.setattr(cache._objects, "stat", stat_then_replace)

    assert cache.evict() == 0
    assert cache.holds(reference)
