"""A tracked collection tells its observer of every change to an entry, and forgets an
entry without journaling it only once the entry has nothing left to write."""

import pytest

from server.utils.journal import ChangeJournal, JournaledModel, TrackedDict


class _Entity(JournaledModel):
    state: str


def _observed() -> tuple[ChangeJournal, TrackedDict[str, _Entity], list[str]]:
    journal = ChangeJournal()
    seen: list[str] = []
    return journal, TrackedDict(journal, "e", on_change=seen.append), seen


def test_the_observer_hears_each_set_assignment_and_removal_after_it_lands() -> None:
    journal, tracked, seen = _observed()
    states: list[str | None] = []

    def heard(key: str) -> None:
        seen.append(key)
        states.append(e.state if (e := tracked.get(key)) is not None else None)

    tracked.on_change = heard

    tracked["a"] = _Entity(state="new")
    tracked["a"].state = "held"
    tracked["a"].state = "held"  # an equal value changes nothing
    del tracked["a"]

    assert seen == ["a", "a", "a"]
    assert states == ["new", "held", None]
    assert ("e", "a") in journal.pending


def test_a_restored_entry_is_heard_without_being_journaled() -> None:
    journal = ChangeJournal()
    seen: list[str] = []
    tracked = TrackedDict[str, _Entity](
        journal, "e", restoring={"a": 3}, on_change=seen.append
    )
    tracked["a"] = _Entity(state="stored")

    assert seen == ["a"]
    assert journal.pending == {}


def test_forget_drops_an_entry_without_journaling_it() -> None:
    journal, tracked, seen = _observed()
    entity = _Entity(state="done")
    tracked["a"] = entity
    journal.clear(journal.captured())
    seen.clear()

    tracked.forget("a")

    assert "a" not in tracked and seen == ["a"]
    assert journal.pending == {}
    entity.state = "later"  # a forgotten entity marks nothing
    assert journal.pending == {}


def test_forget_refuses_an_entry_with_a_change_still_to_write() -> None:
    journal, tracked, _seen = _observed()
    tracked["a"] = _Entity(state="unwritten")

    with pytest.raises(ValueError, match="pending"):
        tracked.forget("a")
    assert "a" in tracked
