"""A ledger journals each change its collections take, and stores one field per
entity in a layout a restore reads back in order or refuses."""

import json
import typing
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from pydantic import AfterValidator, BaseModel

from server.orchestration import LedgerSnapshot
from server.orchestration.journal import (
    AppendOnlyList,
    JournaledModel,
    LedgerJournal,
    TrackedDict,
    TrackedSet,
    freeze_mapping,
)
from server.orchestration.ledger_layout import (
    FOUNDATION,
    HISTORIES,
    KEYED,
    KEYED_TEXTS,
    SCALARS,
    SETS,
    LedgerLayoutError,
    StoredLedger,
    decode_ledger,
    encode_ledger,
    field_name,
)
from server.orchestration.state import (
    Continuation,
    IterationKind,
    IterationResolution,
    OrchestrationEvent,
    ProgressAxis,
    WorkItem,
)
from tests.server.orchestration.helpers import chain_bundle, engine


def _work_item(work_item_id: str = "wki-1") -> WorkItem:
    return WorkItem(
        work_item_id=work_item_id,
        activation_id="act-1",
        operator_id="op",
        legacy_task_id="op",
    )


def test_a_tracked_dict_journals_each_entry_it_sets_or_removes_in_order() -> None:
    journal = LedgerJournal()
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
    journal = LedgerJournal()
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
    journal = LedgerJournal()
    items = TrackedDict[str, WorkItem](journal, "work_items")
    wi = _work_item()
    items[wi.work_item_id] = wi
    journal.pending.clear()

    wi.invocation_id = "inv-1"
    assert journal.pending.keys() == {("work_items", "wki-1")}

    journal.pending.clear()
    del items["wki-1"]
    journal.pending.clear()
    wi.invocation_id = "inv-2"
    assert not journal.pending


def test_an_entity_held_in_one_slot_is_refused_another() -> None:
    journal = LedgerJournal()
    items = TrackedDict[str, WorkItem](journal, "work_items")
    others = TrackedDict[str, WorkItem](journal, "others")
    wi = _work_item()
    items["wki-1"] = wi

    with pytest.raises(ValueError, match="already held"):
        others["wki-1"] = wi
    with pytest.raises(ValueError, match="already held"):
        items["wki-2"] = wi
    others["wki-1"] = wi.model_copy()


def test_a_refused_replacement_leaves_the_entry_journaling_its_entity() -> None:
    journal = LedgerJournal()
    items = TrackedDict[str, WorkItem](journal, "work_items")
    held, other = _work_item("wki-1"), _work_item("wki-2")
    items["wki-1"], items["wki-2"] = held, other

    with pytest.raises(ValueError, match="already held"):
        items["wki-1"] = other
    journal.reset()
    held.invocation_id = "inv-1"

    assert items["wki-1"] is held
    assert set(journal.pending) == {("work_items", "wki-1")}


def test_an_entity_changes_its_containers_only_by_replacing_them() -> None:
    wi = _work_item()
    wi.attempt_ids = [*wi.attempt_ids, "att-1"]  # type: ignore[assignment]
    continuation = Continuation(work_item_id="wki-1", waiting_on=frozenset({"a"}))
    event = OrchestrationEvent(seq=1, kind="k", detail={"a": "b"})
    resolution = IterationResolution(
        loop="scp-1", iteration=0, kind=IterationKind.FEEDBACK, bundle={}
    )

    assert wi.attempt_ids == ("att-1",)
    assert continuation.waiting_on == frozenset({"a"})
    with pytest.raises(TypeError):
        event.detail["a"] = "c"  # type: ignore[index]
    with pytest.raises(TypeError):
        resolution.bundle.clear()  # type: ignore[attr-defined]


def _immutability_breaches(
    model: type[BaseModel], seen: set[type[BaseModel]], top: bool
) -> list[str]:
    if model in seen:
        return []
    seen.add(model)
    breaches = []
    journaled = issubclass(model, JournaledModel)
    if journaled and not top:
        breaches.append(f"{model.__name__} is journaled inside another entity")
    if not journaled and not model.model_config.get("frozen"):
        breaches.append(f"{model.__name__} is neither frozen nor journaled")
    for name, info in model.model_fields.items():
        breaches += _annotation_breaches(
            info.annotation, f"{model.__name__}.{name}", seen, frozen=False
        )
    return breaches


def _annotation_breaches(
    annotation: Any, where: str, seen: set[type[BaseModel]], frozen: bool
) -> list[str]:
    if isinstance(annotation, typing.TypeAliasType):
        return _annotation_breaches(annotation.__value__, where, seen, frozen)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        frozen = any(
            isinstance(meta, AfterValidator) and meta.func is freeze_mapping
            for meta in annotation.__metadata__
        )
        return _annotation_breaches(args[0], where, seen, frozen)
    breaches = []
    if origin in (list, set, dict) or annotation in (list, set, dict):
        breaches.append(f"{where} is a mutable {annotation}")
    if origin is Mapping and not frozen:
        breaches.append(f"{where} is a mapping it does not freeze")
    for arg in args:
        breaches += _annotation_breaches(arg, where, seen, False)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        breaches += _immutability_breaches(annotation, seen, top=False)
    return breaches


def test_every_value_a_ledger_stores_changes_only_through_its_collection() -> None:
    seen: set[type[BaseModel]] = set()
    breaches = []
    for name in (*KEYED, *HISTORIES):
        model = KEYED[name][0] if name in KEYED else HISTORIES[name]
        breaches += _immutability_breaches(model, seen, top=True)
    for name in FOUNDATION:
        annotation = LedgerSnapshot.model_fields[name].annotation
        breaches += _annotation_breaches(annotation, name, seen, frozen=False)

    assert not breaches


def test_the_stored_layout_places_every_snapshot_field_once() -> None:
    groups = [set(KEYED), set(KEYED_TEXTS), set(HISTORIES), set(SETS)]
    groups += [set(FOUNDATION), set(SCALARS)]

    assert set().union(*groups) == set(LedgerSnapshot.model_fields)
    assert sum(map(len, groups)) == len(LedgerSnapshot.model_fields)


def test_a_clear_keeps_a_change_made_after_it_was_captured() -> None:
    journal = LedgerJournal()
    journal.mark(("c", "a"))
    journal.mark(("c", "b"))
    captured = journal.captured()
    journal.mark(("c", "b"))
    journal.mark(("c", "d"))

    journal.clear(captured)

    assert set(journal.pending) == {("c", "b"), ("c", "d")}


def test_each_key_names_one_field() -> None:
    names = {
        field_name("c", key)
        for key in [
            ("a", 1),
            ("a", "1"),
            "a,1",
            ("a", ProgressAxis.LOOP_TIME),
            ("a", "loop_time", ""),
            '["a",1]',
        ]
    }

    assert len(names) == 6
    assert field_name("c", ("a", ProgressAxis.LOOP_TIME)) == 'c:["a","loop_time"]'


@pytest.mark.parametrize(
    "key",
    [
        "wki-1",
        'quo"te',
        "back\\slash",
        "tab\there",
        "\x7f",
        "naïve",
        "",
        7,
        True,
        ("a", 1, ProgressAxis.LOOP_TIME),
    ],
)
def test_a_field_name_is_its_key_as_compact_json(key: Any) -> None:
    parts = key if isinstance(key, tuple) else (key,)
    values = [p.value if isinstance(p, ProgressAxis) else p for p in parts]

    assert field_name("c", key) == "c:" + json.dumps(values, separators=(",", ":"))


def _stored() -> StoredLedger:
    eng = engine(chain_bundle())
    eng.on_dispatched("A", "w1")
    eng.on_failed("A", "boom", retryable=False)
    image = eng._codec.image()
    return decode_ledger(image)


def test_a_stored_ledger_reads_back_as_it_was_written() -> None:
    eng = engine(chain_bundle())
    eng.on_dispatched("A", "w1")
    eng.on_failed("A", "boom", retryable=False)
    image = eng._codec.image()

    stored = decode_ledger(image)

    assert stored.snapshot == eng.to_snapshot()
    assert encode_ledger(stored) == image


def _named(image: dict[str, str], collection: str) -> list[str]:
    return sorted(name for name in image if name.startswith(f"{collection}:"))


def test_a_restore_reads_keyed_entries_in_their_insertion_order() -> None:
    image = encode_ledger(_stored())
    ordinal = {n: int(image[n].partition(":")[0]) for n in _named(image, "work_items")}
    first, second = sorted(ordinal, key=ordinal.__getitem__)
    image[first] = f"{ordinal[second]}:{image[first].partition(':')[2]}"
    image[second] = f"{ordinal[first]}:{image[second].partition(':')[2]}"

    stored = decode_ledger(image)

    assert [wi.work_item_id for wi in stored.snapshot.work_items] == [
        name.removeprefix('work_items:["').removesuffix('"]')
        for name in (second, first)
    ]


def _swap_entity(image: dict[str, str]) -> None:
    first, second = _named(image, "work_items")
    image[first] = image[second]


def _drop_ordinal(image: dict[str, str]) -> None:
    first = _named(image, "work_items")[0]
    image[first] = image[first].partition(":")[2]


def _repeat_ordinal(image: dict[str, str]) -> None:
    first, second = _named(image, "work_items")
    ordinal = image[first].partition(":")[0]
    image[second] = f"{ordinal}:{image[second].partition(':')[2]}"


def _respell(image: dict[str, str], name: str, spelling: str) -> None:
    image[spelling] = image.pop(name)


def _foundation_with(image: dict[str, str], **changes: Any) -> None:
    foundation = json.loads(image["meta:foundation"])
    foundation.update(changes)
    image["meta:foundation"] = json.dumps(
        {k: v for k, v in foundation.items() if v is not _ABSENT}
    )


_ABSENT = object()

_CORRUPTIONS: dict[str, Callable[[dict[str, str]], Any]] = {
    "unknown layout": lambda f: f.update({"meta:layout": "0"}),
    "no foundation": lambda f: f.pop("meta:foundation"),
    "unknown meta field": lambda f: f.update({"meta:other": "1"}),
    "unknown collection": lambda f: f.update({'others:["a"]': "0:{}"}),
    "field without a key": lambda f: f.update({"scopes": "x"}),
    "malformed key": lambda f: f.update({"scopes:[a": "x"}),
    "entity under another key": _swap_entity,
    "entry without an ordinal": _drop_ordinal,
    "conflicting ordinals": _repeat_ordinal,
    "history with a gap": lambda f: f.pop(_named(f, "trace")[0]),
    "malformed set member": lambda f: f.update({"failed_scopes:[1]": "1"}),
    "missing required field": lambda f: f.update(
        {"meta:foundation": '{"instance": null}'}
    ),
    "missing next_seq": lambda f: f.pop("meta:next_seq"),
    "missing instance_failure": lambda f: f.pop("meta:instance_failure"),
    "missing control_failure": lambda f: f.pop("meta:control_failure"),
    "missing instance_cancelled": lambda f: f.pop("meta:instance_cancelled"),
    "foundation with a foreign field": lambda f: _foundation_with(f, next_seq=99),
    "foundation missing a field": lambda f: _foundation_with(
        f, max_loop_iterations=_ABSENT
    ),
    "malformed foundation": lambda f: f.update({"meta:foundation": "{not json"}),
    "foundation not an object": lambda f: f.update({"meta:foundation": "[]"}),
    "malformed scalar": lambda f: f.update({"meta:next_seq": "{bad"}),
    "malformed entity": lambda f: f.update({_named(f, "work_items")[0]: "0:{not json"}),
    "boolean history position": lambda f: _respell(f, "trace:[1]", "trace:[true]"),
    "padded history position": lambda f: _respell(f, "trace:[1]", "trace:[ 1]"),
    "fractional history position": lambda f: _respell(f, "trace:[1]", "trace:[1.0]"),
    "one history position twice": lambda f: f.update({"trace:[ 0]": f["trace:[1]"]}),
    "respelled failure reason": lambda f: _respell(
        f, _named(f, "failure_reasons")[0], 'failure_reasons:[ "A"]'
    ),
    "respelled set member": lambda f: f.update({'released_scopes:[ "scp-x"]': "1"}),
}


@pytest.mark.parametrize("corruption", _CORRUPTIONS)
def test_a_stored_ledger_that_cannot_be_placed_is_refused(corruption: str) -> None:
    image = encode_ledger(_stored())
    _CORRUPTIONS[corruption](image)

    with pytest.raises(LedgerLayoutError):
        decode_ledger(image)


def test_an_unplaceable_field_is_a_layout_error() -> None:
    image = encode_ledger(_stored())
    image['others:["a"]'] = "0:{}"

    with pytest.raises(LedgerLayoutError, match="unknown ledger collection"):
        decode_ledger(image)


def test_every_empty_frozen_mapping_is_one_shared_map() -> None:
    events = [
        OrchestrationEvent(seq=1, kind="k"),
        OrchestrationEvent(seq=2, kind="k", detail={}),
        OrchestrationEvent.model_validate_json('{"seq":3,"kind":"k","detail":{}}'),
    ]

    assert len({id(event.detail) for event in events}) == 1
    assert freeze_mapping({}) is events[0].detail
