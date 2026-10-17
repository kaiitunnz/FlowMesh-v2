"""The mark: a scope's Bloom filter never loses a digest added to it and stays under its
target false-positive rate, at its capacity hint and after growing past it."""

import hashlib

import pytest

import server.reclamation.mark as mark_module

from server.reclamation.mark import (
    FALSE_POSITIVE_RATE,
    MIN_CAPACITY,
    DigestSet,
    Mark,
    ScopeMark,
    digest_words,
    is_content_digest,
)


def _digests(prefix: str, count: int) -> list[str]:
    return [hashlib.sha256(f"{prefix}-{i}".encode()).hexdigest() for i in range(count)]


def _false_positive_rate(mark: ScopeMark, probes: int = 100_000) -> float:
    return sum(d in mark for d in _digests("absent", probes)) / probes


def test_every_digest_added_is_found_again() -> None:
    mark = ScopeMark(10_000)
    digests = _digests("present", 100_000)
    for digest in digests:
        mark.add(digest)

    assert all(digest in mark for digest in digests)


def test_the_false_positive_rate_at_its_hint_is_under_the_target() -> None:
    mark = ScopeMark(20_000)
    for digest in _digests("present", 20_000):
        mark.add(digest)

    assert _false_positive_rate(mark) < FALSE_POSITIVE_RATE


def test_a_mark_grown_far_past_a_tiny_hint_stays_near_the_target() -> None:
    mark = ScopeMark(1)
    digests = _digests("present", 60 * MIN_CAPACITY)
    for digest in digests:
        mark.add(digest)

    assert all(digest in mark for digest in digests)
    assert _false_positive_rate(mark) < 2 * FALSE_POSITIVE_RATE


@pytest.mark.parametrize("digest", ["not-hex", "ab", "A" * 64, "a" * 62, "a" * 66])
def test_only_a_lowercase_hex_sha256_digest_is_a_content_digest(digest: str) -> None:
    assert not is_content_digest(digest)
    with pytest.raises(ValueError):
        ScopeMark().add(digest)
    with pytest.raises(ValueError):
        DigestSet().add(digest)


def test_words_and_digests_name_the_same_entry() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    mark = ScopeMark()
    mark.add_words(digest_words(digest))

    assert digest in mark and digest_words(digest) in mark


def test_a_scope_absent_from_the_mark_binds_nothing() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    mark = Mark()
    mark.add("org-a", digest)

    assert mark.binds("org-a", digest)
    assert not mark.binds("org-b", digest)
    assert mark.scopes() == {"org-a"}


def test_counts_and_size_cover_every_scope() -> None:
    mark = Mark({"org-a": 4000})
    for digest in _digests("a", 3):
        mark.add("org-a", digest)
    mark.add("org-a", _digests("a", 1)[0])
    mark.add_all("org-b", [digest_words(d) for d in _digests("b", 2)])

    assert mark.counts() == {"org-a": 3, "org-b": 2}
    assert mark.nbytes == mark.scope("org-a").nbytes + mark.scope("org-b").nbytes
    assert mark.scope("org-a").nbytes > mark.scope("org-b").nbytes


def test_a_reserved_batch_lands_in_one_layer_sized_for_it() -> None:
    mark = ScopeMark(1)
    words = [digest_words(f"{i:064x}") for i in range(50_000)]

    mark.reserve(len(words))
    for pair in words:
        mark.add_words(pair)

    assert all(pair in mark for pair in words)
    assert mark.nbytes * 8 / len(words) < 14


def test_a_digest_set_holds_exactly_what_was_added_across_compactions() -> None:
    digests = _digests("d", 2000)
    absent = _digests("x", 2000)
    held = DigestSet()
    for digest in digests[:1000]:
        held.add(digest)
    held.compact()
    for digest in digests[1000:]:
        held.add(digest)
    held.add(digests[0])

    assert all(digest in held for digest in digests)
    assert not any(digest in held for digest in absent)
    held.compact()
    assert all(digest in held for digest in digests)
    assert len(held) == 2000 and held.nbytes == 2000 * 32


def test_a_digest_set_fills_at_its_limit_and_compacts_with_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mark_module, "PENDING_LIMIT", 3)
    held = DigestSet()
    digests = _digests("d", 10_000)
    for digest in digests[:2]:
        held.add(digest)
    assert not held.full
    held.add(digests[2])
    assert held.full
    held.compact()
    for digest in digests[3:]:
        held.add(digest)
    ticks: list[None] = []

    held.compact(lambda: ticks.append(None))

    assert ticks and not held.full
    assert all(digest in held for digest in digests) and len(held) == 10_000
