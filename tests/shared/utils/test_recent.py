"""A RecentSet and a RecentMap keep what was added most recently, up to their
capacity."""

import pytest

from shared.utils.recent import RecentMap, RecentSet


def test_the_oldest_item_is_forgotten_first() -> None:
    recent: RecentSet[str] = RecentSet(2)
    for item in ("a", "b", "c"):
        recent.add(item)

    assert list(recent) == ["b", "c"]
    assert "a" not in recent


def test_adding_an_item_again_makes_it_the_newest() -> None:
    recent: RecentSet[str] = RecentSet(2)
    recent.add("a")
    recent.add("b")
    recent.add("a")
    recent.add("c")

    assert list(recent) == ["a", "c"]


def test_a_discarded_item_is_gone() -> None:
    recent: RecentSet[str] = RecentSet(2)
    recent.add("a")
    recent.discard("a")
    recent.discard("missing")

    assert len(recent) == 0


def test_a_recent_set_holds_at_least_one_item() -> None:
    with pytest.raises(ValueError):
        RecentSet(0)


def test_the_key_set_least_recently_is_forgotten_first() -> None:
    recent: RecentMap[str, int] = RecentMap(2)
    recent["a"] = 1
    recent["b"] = 2
    recent["a"] = 3
    recent["c"] = 4

    assert (recent.get("a"), recent.get("b"), recent.get("c")) == (3, None, 4)
    assert len(recent) == 2


def test_a_recent_map_holds_at_least_one_item() -> None:
    with pytest.raises(ValueError):
        RecentMap(0)
