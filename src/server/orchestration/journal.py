"""Change tracking over the persisted collections of one orchestration ledger.

The ledger's collections record each change they take in a ``LedgerJournal``, so a
durable write carries only what changed since the last one. A mutable entity reports
an assignment to the one collection slot that holds it, and keeps its container fields
immutable so it can only change by assignment. An append-only history refuses any
other change.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, NoReturn, Self

from pydantic import BaseModel, PrivateAttr

type JournalKey = tuple[str, Hashable]


class LedgerJournal:
    """What of a ledger changed since its last durable write: each changed entry of a
    keyed collection or set, by collection and key, with the version it last changed
    at.

    Versions let a write clear only the changes it captured, so a change made while
    the write is in flight stays pending.
    """

    def __init__(self) -> None:
        self.pending: dict[JournalKey, int] = {}
        self._version = 0
        self._next_ordinal = 0

    def mark(self, key: JournalKey) -> None:
        self._version += 1
        self.pending[key] = self._version

    def ordinal(self) -> int:
        """The next insertion ordinal, which orders a keyed collection on restore."""
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        return ordinal

    def seen_ordinal(self, ordinal: int) -> None:
        self._next_ordinal = max(self._next_ordinal, ordinal + 1)

    def captured(self) -> dict[JournalKey, int]:
        return dict(self.pending)

    def clear(self, captured: Mapping[JournalKey, int]) -> None:
        """Clear the captured changes not changed again since."""
        for key, version in captured.items():
            if self.pending.get(key) == version:
                del self.pending[key]


class JournaledModel(BaseModel):
    """A mutable ledger entity: each assignment marks the collection slot holding it,
    and a container assigned to it is stored frozen."""

    _slot: tuple["TrackedDict[Any, Any]", Hashable] | None = PrivateAttr(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        super().__setattr__(name, frozen(value))
        if (slot := self._slot) is not None:
            slot[0].touch(slot[1])

    @property
    def held(self) -> bool:
        return self._slot is not None

    def claim(self, owner: "TrackedDict[Any, Any]", key: Hashable) -> None:
        if (slot := self._slot) is not None and (
            slot[0] is not owner or slot[1] != key
        ):
            raise ValueError(
                f"{type(self).__name__} {key!r} is already held at {slot[1]!r}"
            )
        self._slot = (owner, key)

    def release(self, owner: "TrackedDict[Any, Any]") -> None:
        if (slot := self._slot) is not None and slot[0] is owner:
            self._slot = None

    def __eq__(self, other: Any) -> bool:
        # The slot holding an entity is not part of its value.
        if not isinstance(other, BaseModel):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.__dict__ == other.__dict__
            and self.__pydantic_extra__ == other.__pydantic_extra__
        )

    def __copy__(self) -> Self:
        copied = super().__copy__()
        copied._slot = None
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = super().__deepcopy__(memo)
        copied._slot = None
        return copied


def _refusal(message: str) -> Callable[..., NoReturn]:
    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(message)

    return refuse


class FrozenMap[K, V](dict[K, V]):
    """A mapping that refuses every change; it copies as itself."""

    _refuse = _refusal("a frozen mapping is immutable")
    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse
    clear = _refuse

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (dict(self),))


def frozen(value: Any) -> Any:
    """A container as its immutable counterpart; any other value as it is."""
    match value:
        case FrozenMap():
            return value
        case dict():
            return FrozenMap(value)
        case set():
            return frozenset(value)
        case list():
            return tuple(value)
    return value


def freeze_mapping[K, V](value: Mapping[K, V]) -> FrozenMap[K, V]:
    return value if isinstance(value, FrozenMap) else FrozenMap(value)


class TrackedDict[K: Hashable, V](dict[K, V]):
    """A keyed ledger collection that journals each entry it sets or removes, keeps
    each key's insertion ordinal, and holds each mutable entity it stores."""

    def __init__(
        self,
        journal: LedgerJournal,
        name: str,
        items: Iterable[tuple[K, V]] = (),
    ) -> None:
        super().__init__()
        self.journal = journal
        self.name = name
        self.ordinals: dict[K, int] = {}
        for key, value in items:
            self[key] = value

    def touch(self, key: Hashable) -> None:
        self.journal.mark((self.name, key))

    def __setitem__(self, key: K, value: V) -> None:
        if (old := self.get(key)) is not value and isinstance(old, JournaledModel):
            old.release(self)
        if isinstance(value, JournaledModel):
            value.claim(self, key)
        super().__setitem__(key, value)
        if key not in self.ordinals:
            self.ordinals[key] = self.journal.ordinal()
        self.touch(key)

    def __delitem__(self, key: K) -> None:
        if isinstance(old := self[key], JournaledModel):
            old.release(self)
        super().__delitem__(key)
        del self.ordinals[key]
        self.touch(key)

    def pop(self, key: K, *default: Any) -> Any:
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self) -> tuple[K, V]:
        key = next(reversed(self))
        value = self[key]
        del self[key]
        return key, value

    def setdefault(self, key: K, default: Any = None) -> V:
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        items: dict[Any, Any] = dict(*args, **kwargs)
        for key, value in items.items():
            self[key] = value

    def clear(self) -> None:
        for key in list(self):
            del self[key]

    def _updated(self, other: Any) -> Any:
        self.update(other)
        return self

    __ior__ = _updated


class TrackedSet[T: Hashable](set[T]):
    """A ledger set that journals each member it adds or removes."""

    def __init__(
        self, journal: LedgerJournal, name: str, members: Iterable[T] = ()
    ) -> None:
        super().__init__(members)
        self.journal = journal
        self.name = name

    def add(self, member: T) -> None:
        if member not in self:
            super().add(member)
            self.journal.mark((self.name, member))

    def discard(self, member: Any) -> None:
        if member in self:
            super().discard(member)
            self.journal.mark((self.name, member))

    def remove(self, member: Any) -> None:
        if member not in self:
            raise KeyError(member)
        self.discard(member)

    def pop(self) -> T:
        member = next(iter(self))
        self.discard(member)
        return member

    def clear(self) -> None:
        for member in list(self):
            self.discard(member)

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            for member in other:
                self.add(member)

    def difference_update(self, *others: Iterable[Any]) -> None:
        for other in others:
            for member in list(other):
                self.discard(member)

    def intersection_update(self, *others: Iterable[Any]) -> None:
        keep = set(self).intersection(*others)
        for member in list(self):
            if member not in keep:
                self.discard(member)

    def symmetric_difference_update(self, other: Iterable[T]) -> None:
        for member in set(other):
            if member in self:
                self.discard(member)
            else:
                self.add(member)

    def _updated(self, other: Any) -> Any:
        self.update(other)
        return self

    def _intersected(self, other: Any) -> Any:
        self.intersection_update(other)
        return self

    def _subtracted(self, other: Any) -> Any:
        self.difference_update(other)
        return self

    def _xored(self, other: Any) -> Any:
        self.symmetric_difference_update(other)
        return self

    __ior__ = _updated
    __iand__ = _intersected
    __isub__ = _subtracted
    __ixor__ = _xored


class AppendOnlyList[T](list[T]):
    """A ledger history: entries are only ever appended, so what a durable write took
    is the prefix up to the length it saw."""

    _refuse = _refusal("a ledger history only appends")
    __setitem__ = _refuse
    __delitem__ = _refuse
    __imul__ = _refuse
    insert = _refuse
    pop = _refuse
    remove = _refuse
    clear = _refuse
    sort = _refuse
    reverse = _refuse
