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

import heapq
import math
import re
from collections.abc import Callable, Iterable, Mapping

# The target false-positive rate, and the least capacity a scope's first layer holds.
FALSE_POSITIVE_RATE = 0.01
MIN_CAPACITY = 1024

_LN2_SQUARED = math.log(2) ** 2


_CONTENT_DIGEST = re.compile(r"[0-9a-f]{64}")


def is_content_digest(digest: str) -> bool:
    """Whether ``digest`` is a SHA-256 content digest in lowercase hex."""
    return _CONTENT_DIGEST.fullmatch(digest) is not None


def digest_bytes(digest: str) -> bytes:
    """The 32 bytes of a content digest; raises ``ValueError`` for anything else."""
    if not is_content_digest(digest):
        raise ValueError(f"not a content digest: {digest!r}")
    return bytes.fromhex(digest)


def digest_words(digest: str) -> tuple[int, int]:
    """The two words a content digest's bit positions derive from."""
    raw = digest_bytes(digest)
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

    def reserve(self, count: int) -> None:
        """Make room for ``count`` more digests in one layer, so a batch of known size
        never grows the filter a layer at a time."""
        layer = self._layers[-1]
        if layer.capacity - layer.count >= count:
            return
        capacity = max(layer.capacity * 2, count)
        self._layers.append(_Layer(capacity, self._error / 2 ** len(self._layers)))

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


class Mark:
    """The digests each scope binds; a scope absent from it binds nothing."""

    def __init__(self, capacities: Mapping[str, int] | None = None) -> None:
        self._capacities = dict(capacities or {})
        self._scopes: dict[str, ScopeMark] = {}

    def add(self, scope: str, digest: str) -> None:
        self.scope(scope).add(digest)

    def add_all(
        self, scope: str, words: Iterable[tuple[int, int]], count: int = 0
    ) -> None:
        """Add every digest of ``words``, ``count`` of them when known."""
        mark = self.scope(scope)
        mark.reserve(count)
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

    @property
    def nbytes(self) -> int:
        return sum(mark.nbytes for mark in self._scopes.values())


_DIGEST_BYTES = 32
# The most digests a set holds unpacked before a caller should compact it, and the
# records merged between ticks of a compaction's pacer.
PENDING_LIMIT = 65536
_MERGED_PER_TICK = 4096


class DigestSet:
    """An exact set of content digests, packed 32 bytes each.

    Digests added since the last ``compact`` wait in a small set; ``compact`` merges
    them into one sorted run that lookups search, and ``full`` says when enough wait
    that it should.
    """

    def __init__(self) -> None:
        self._sorted = bytearray()
        self._pending: set[bytes] = set()

    def add(self, digest: str) -> None:
        if (raw := digest_bytes(digest)) not in self:
            self._pending.add(raw)

    def __contains__(self, item: str | bytes) -> bool:
        raw = item if isinstance(item, bytes) else digest_bytes(item)
        if raw in self._pending:
            return True
        low, high = 0, len(self._sorted) // _DIGEST_BYTES
        while low < high:
            middle = (low + high) // 2
            at = middle * _DIGEST_BYTES
            found = self._sorted[at : at + _DIGEST_BYTES]
            if found == raw:
                return True
            if found < raw:
                low = middle + 1
            else:
                high = middle
        return False

    @property
    def full(self) -> bool:
        return len(self._pending) >= PENDING_LIMIT

    def compact(self, tick: Callable[[], object] | None = None) -> None:
        """Merge the waiting digests into the sorted run, calling ``tick`` between
        stretches of the merge."""
        if not self._pending:
            return
        packed = self._sorted
        merged = bytearray()
        for count, record in enumerate(
            heapq.merge(
                (
                    packed[at : at + _DIGEST_BYTES]
                    for at in range(0, len(packed), _DIGEST_BYTES)
                ),
                sorted(self._pending),
            ),
            1,
        ):
            merged += record
            if tick is not None and count % _MERGED_PER_TICK == 0:
                tick()
        self._sorted = merged
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._sorted) // _DIGEST_BYTES + len(self._pending)

    @property
    def nbytes(self) -> int:
        return len(self._sorted) + len(self._pending) * _DIGEST_BYTES


__all__ = [
    "FALSE_POSITIVE_RATE",
    "MIN_CAPACITY",
    "PENDING_LIMIT",
    "DigestSet",
    "Mark",
    "ScopeMark",
    "digest_bytes",
    "digest_words",
    "is_content_digest",
]
