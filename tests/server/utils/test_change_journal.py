"""A tracked collection journals each change to an entry and tells its observer of it,
an entity marks only the slot holding it, and an entry is forgotten without journaling
only once it has nothing left to write."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import pytest

from server.utils.journal import (
    AppendOnlyList,
    ChangeJournal,
    JournaledModel,
    TrackedDict,
    TrackedSet,
)


class _Status(StrEnum):
    READY = "ready"
    DONE = "done"


class _Entity(JournaledModel):
    state: str = ""
    status: _Status = _Status.READY
    ids: tuple[str, ...] = ()


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


def test_a_tracked_dict_journals_each_entry_it_sets_or_removes_in_order() -> None:
    journal = ChangeJournal()
    tracked = TrackedDict[str, str](journal, "c", [("a", "1"), ("b", "2")])
    tracked["c"] = "3"
    first = dict(tracked.ordinals)

    tracked["b"] = "4"
    del tracked["a"]
    tracked["a"] = "5"
    tracked.setdefault("d", "6")
    tracked.update({"e": "7"})
    tracked.pop("e")

    assert list(tracked) == ["b", "c", "a", "d"]
    assert tracked.ordinals["b"] == first["b"]
    assert tracked.ordinals["a"] > first["c"]
    assert sorted(tracked, key=tracked.ordinals.__getitem__) == list(tracked)
    assert set(journal.pending) == {("c", key) for key in "abcde"}


def test_a_tracked_set_journals_each_member_it_adds_or_removes() -> None:
    journal = ChangeJournal()
    members = TrackedSet(journal, "s", ["a"])
    members.add("a")
    assert not journal.pending

    members |= {"b"}
    members -= {"a"}
    members.discard("z")

    assert set(members) == {"b"}
    assert set(journal.pending) == {("s", "a"), ("s", "b")}


_EDITS: dict[str, Callable[[list[Any]], Any]] = {
    "set": lambda h: h.__setitem__(0, "x"),
    "delete": lambda h: h.__delitem__(0),
    "insert": lambda h: h.insert(0, "x"),
    "pop": lambda h: h.pop(),
    "remove": lambda h: h.remove("a"),
    "clear": lambda h: h.clear(),
    "sort": lambda h: h.sort(),
    "reverse": lambda h: h.reverse(),
    "repeat": lambda h: h.__imul__(2),
}


@pytest.mark.parametrize("edit", _EDITS)
def test_a_history_refuses_every_change_but_an_append(edit: str) -> None:
    history = AppendOnlyList(["a", "b"])

    with pytest.raises(TypeError):
        _EDITS[edit](history)
    history.append("c")
    history += ["d"]
    assert history == ["a", "b", "c", "d"]


def test_an_entity_marks_only_the_collection_slot_holding_it() -> None:
    journal = ChangeJournal()
    items = TrackedDict[str, _Entity](journal, "items")
    entity = _Entity()
    items["a"] = entity
    journal.pending.clear()

    entity.state = "held"
    assert journal.pending.keys() == {("items", "a")}

    journal.pending.clear()
    del items["a"]
    journal.pending.clear()
    entity.state = "gone"
    assert not journal.pending


@pytest.mark.parametrize(
    ("field", "stored", "assigned", "marks"),
    [
        ("status", _Status.READY, _Status.READY, False),
        ("status", _Status.READY, _Status.DONE, True),
        ("status", _Status.READY, "ready", True),
        ("ids", ("a", "b"), ["a", "b"], False),
    ],
    ids=["equal", "different", "equal-other-type", "list-equal-to-tuple"],
)
def test_an_entity_marks_an_assignment_only_when_it_changes_the_value(
    field: str, stored: Any, assigned: Any, marks: bool
) -> None:
    journal = ChangeJournal()
    items = TrackedDict[str, _Entity](journal, "items")
    entity = _Entity()
    setattr(entity, field, stored)
    items["a"] = entity
    journal.pending.clear()

    setattr(entity, field, assigned)

    assert bool(journal.pending) is marks
    assert getattr(entity, field) == (assigned if marks else stored)


def test_an_entity_held_in_one_slot_is_refused_another() -> None:
    journal = ChangeJournal()
    items = TrackedDict[str, _Entity](journal, "items")
    others = TrackedDict[str, _Entity](journal, "others")
    entity = _Entity()
    items["a"] = entity

    with pytest.raises(ValueError, match="already held"):
        others["a"] = entity
    with pytest.raises(ValueError, match="already held"):
        items["b"] = entity
    others["a"] = entity.model_copy()


def test_a_refused_replacement_leaves_the_entry_journaling_its_entity() -> None:
    journal = ChangeJournal()
    items = TrackedDict[str, _Entity](journal, "items")
    held, other = _Entity(), _Entity()
    items["a"], items["b"] = held, other

    with pytest.raises(ValueError, match="already held"):
        items["a"] = other
    journal.reset()
    held.state = "changed"

    assert items["a"] is held
    assert set(journal.pending) == {("items", "a")}


def test_a_clear_keeps_a_change_made_after_it_was_captured() -> None:
    journal = ChangeJournal()
    journal.mark(("c", "a"))
    journal.mark(("c", "b"))
    captured = journal.captured()
    journal.mark(("c", "b"))
    journal.mark(("c", "d"))

    journal.clear(captured)

    assert set(journal.pending) == {("c", "b"), ("c", "d")}
