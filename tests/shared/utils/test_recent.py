"""A RecentSet keeps the items added most recently, up to its capacity."""

import pytest

from shared.utils.recent import RecentSet


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
