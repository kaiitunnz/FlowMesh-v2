"""A compact, conservative record of which content digests a scope binds.

Each scope keeps a scalable Bloom filter: a digest added is always found again, and a
digest never added is found with probability below the target false-positive rate. A
false positive counts an unbound object as bound, so an audit reading the mark can only
undercount what is unbound. A content digest is already a uniform hash, so its bit
positions come from its own bytes, two 64-bit words combined per position.

A filter's first layer is sized for a capacity hint; once that many digests are added
it grows a layer of twice the capacity at half the error rate, so the whole stays below
the target.
"""

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# The target false-positive rate, and the least capacity a scope's first layer holds.
FALSE_POSITIVE_RATE = 0.01
MIN_CAPACITY = 1024

_LN2_SQUARED = math.log(2) ** 2


def digest_words(digest: str) -> tuple[int, int]:
    """The two words a digest's bit positions derive from.

    A hex SHA-256 digest supplies them directly; anything else is hashed first.
    """
    try:
        raw = bytes.fromhex(digest)
    except ValueError:
        raw = b""
    if len(raw) < 16:
        raw = hashlib.sha256(digest.encode()).digest()
    return int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:16], "little") | 1


class _Layer:
    __slots__ = ("bits", "capacity", "count", "hashes", "size")

    def __init__(self, capacity: int, error: float) -> None:
        self.capacity = capacity
        self.size = max(8, math.ceil(-capacity * math.log(error) / _LN2_SQUARED))
        self.hashes = max(1, round(self.size / capacity * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)
        self.count = 0

    def add(self, h1: int, h2: int) -> None:
        bits, size = self.bits, self.size
        for i in range(self.hashes):
            position = (h1 + i * h2) % size
            bits[position >> 3] |= 1 << (position & 7)
        self.count += 1

    def __contains__(self, words: tuple[int, int]) -> bool:
        h1, h2 = words
        bits, size = self.bits, self.size
        for i in range(self.hashes):
            position = (h1 + i * h2) % size
            if not bits[position >> 3] & (1 << (position & 7)):
                return False
        return True


class ScopeMark:
    """The digests one scope binds, as a scalable Bloom filter."""

    def __init__(
        self, capacity: int = MIN_CAPACITY, error: float = FALSE_POSITIVE_RATE
    ) -> None:
        self._error = error / 2
        self._layers = [_Layer(max(MIN_CAPACITY, capacity), self._error)]

    def add(self, digest: str) -> None:
        self.add_words(digest_words(digest))

    def add_words(self, words: tuple[int, int]) -> None:
        if words in self:
            return
        layer = self._layers[-1]
        if layer.count >= layer.capacity:
            layer = _Layer(layer.capacity * 2, self._error / 2 ** len(self._layers))
            self._layers.append(layer)
        layer.add(*words)

    def __contains__(self, item: str | tuple[int, int]) -> bool:
        words = digest_words(item) if isinstance(item, str) else item
        return any(words in layer for layer in self._layers)

    @property
    def count(self) -> int:
        """Distinct digests added, short of any a false positive absorbed."""
        return sum(layer.count for layer in self._layers)

    @property
    def nbytes(self) -> int:
        return sum(len(layer.bits) for layer in self._layers)


@dataclass(frozen=True)
class MarkSize:
    scopes: int
    digests: int
    bytes: int


class Mark:
    """The digests each scope binds; a scope absent from it binds nothing."""

    def __init__(self, capacities: Mapping[str, int] | None = None) -> None:
        self._capacities = dict(capacities or {})
        self._scopes: dict[str, ScopeMark] = {}

    def add(self, scope: str, digest: str) -> None:
        self.scope(scope).add(digest)

    def add_all(self, scope: str, words: Iterable[tuple[int, int]]) -> None:
        mark = self.scope(scope)
        for pair in words:
            mark.add_words(pair)

    def scope(self, scope: str) -> ScopeMark:
        if (mark := self._scopes.get(scope)) is None:
            # Headroom over the hint keeps a steady scope in one layer.
            hint = self._capacities.get(scope, 0) * 5 // 4
            mark = self._scopes[scope] = ScopeMark(hint)
        return mark

    def binds(self, scope: str, digest: str) -> bool:
        mark = self._scopes.get(scope)
        return mark is not None and digest in mark

    def scopes(self) -> frozenset[str]:
        return frozenset(self._scopes)

    def counts(self) -> dict[str, int]:
        return {scope: mark.count for scope, mark in self._scopes.items()}

    def size(self) -> MarkSize:
        return MarkSize(
            len(self._scopes),
            sum(mark.count for mark in self._scopes.values()),
            sum(mark.nbytes for mark in self._scopes.values()),
        )


__all__ = [
    "FALSE_POSITIVE_RATE",
    "MIN_CAPACITY",
    "Mark",
    "MarkSize",
    "ScopeMark",
    "digest_words",
]
