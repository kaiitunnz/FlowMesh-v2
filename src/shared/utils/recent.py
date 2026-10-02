"""Bounded memories of the items seen most recently."""

from collections import OrderedDict
from collections.abc import Hashable, Iterator


class RecentSet[T: Hashable]:
    """The last ``capacity`` distinct items added; the oldest is forgotten first.

    It is not synchronized: a caller sharing one across threads holds its own lock.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a RecentSet holds at least one item")
        self._capacity = capacity
        self._items: OrderedDict[T, None] = OrderedDict()

    def add(self, item: T) -> None:
        self._items[item] = None
        self._items.move_to_end(item)
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def discard(self, item: T) -> None:
        self._items.pop(item, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class RecentMap[K: Hashable, V]:
    """The values of the last ``capacity`` distinct keys set; the key set least
    recently is forgotten first.

    It is not synchronized: a caller sharing one across threads holds its own lock.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a RecentMap holds at least one item")
        self._capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()

    def __setitem__(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def get(self, key: K) -> V | None:
        return self._items.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)
