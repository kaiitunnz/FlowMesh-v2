"""Paged scans over lexically ordered indexes, shared by the listings that read one.

A scan reads an index from an exclusive start member in growing steps, so a page that
a filter thins out reads further without reading the whole index for a dense one.
"""

import heapq
import itertools
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)

# The most index entries one filtered scan step reads, unless a page asks for more.
SCAN_CHUNK_MAX = 1000

# Reads up to ``count`` members of an index between two lexical bounds, descending
# when ``reverse``.
type LexRange = Callable[[str, str, str, int, bool], list[str]]
type LexRangeAsync = Callable[[str, str, str, int, bool], Awaitable[list[str]]]


def scan_sizes(first: int, filtered: bool, cap: int = SCAN_CHUNK_MAX) -> Iterator[int]:
    """Yield the entries each scan step reads: ``first``, then doubling up to the
    larger of ``first`` and ``cap`` while a filter rejects entries."""
    size, top = first, max(first, cap) if filtered else first
    while True:
        yield size
        size = min(size * 2, top)


def lex_bounds(start: str | None, forward: bool) -> tuple[str, str]:
    """The lexical bounds of a scan from the exclusive member ``start``, ascending
    when ``forward``, else descending; from the first or last member without one."""
    if forward:
        return f"({start}" if start is not None else "-", "+"
    return f"({start}" if start is not None else "+", "-"


def lex_scan(
    read: LexRange,
    key: str,
    start: str | None,
    forward: bool,
    sizes: Iterator[int],
) -> Generator[list[str]]:
    """Yield an index's members past ``start`` in scan order, a step at a time."""
    for size in sizes:
        members = read(key, *lex_bounds(start, forward), size, not forward)
        if members:
            yield members
        if len(members) < size:
            return
        start = members[-1]


async def lex_scan_async(
    read: LexRangeAsync,
    key: str,
    start: str | None,
    forward: bool,
    sizes: Iterator[int],
) -> AsyncGenerator[list[str]]:
    for size in sizes:
        members = await read(key, *lex_bounds(start, forward), size, not forward)
        if members:
            yield members
        if len(members) < size:
            return
        start = members[-1]


def chunked[T](items: Iterable[T], sizes: Iterator[int]) -> Generator[list[T]]:
    """Yield ``items`` in order, a step of ``sizes`` at a time."""
    remaining = iter(items)
    for size in sizes:
        if not (chunk := list(itertools.islice(remaining, size))):
            return
        yield chunk


def merged(scans: Sequence[Iterator[list[str]]], forward: bool) -> Iterator[str]:
    """Merge scans of disjoint indexes into one scan order."""
    return heapq.merge(
        *((member for step in scan for member in step) for scan in scans),
        reverse=not forward,
    )


async def merged_async(
    scans: Sequence[AsyncIterator[list[str]]], forward: bool
) -> AsyncGenerator[str]:
    """Merge scans of disjoint indexes into one scan order, as ``merged`` does."""
    buffers: list[list[str]] = [[] for _ in scans]
    live = list(range(len(scans)))

    async def head(index: int) -> str | None:
        buffer = buffers[index]
        while not buffer:
            try:
                buffer.extend(reversed(await anext(scans[index])))
            except StopAsyncIteration:
                return None
        return buffer[-1]

    while live:
        heads = [(member, index) for index in live if (member := await head(index))]
        live = [index for _, index in heads]
        if not heads:
            return
        _, index = (min if forward else max)(heads)
        yield buffers[index].pop()


async def chunked_async(
    items: AsyncIterator[str], sizes: Iterator[int]
) -> AsyncGenerator[list[str]]:
    """Yield an async scan's members a step of ``sizes`` at a time."""
    for size in sizes:
        chunk: list[str] = []
        async for member in items:
            chunk.append(member)
            if len(chunk) == size:
                break
        if chunk:
            yield chunk
        if len(chunk) < size:
            return


__all__ = [
    "SCAN_CHUNK_MAX",
    "LexRange",
    "LexRangeAsync",
    "chunked",
    "chunked_async",
    "lex_bounds",
    "lex_scan",
    "lex_scan_async",
    "merged",
    "merged_async",
    "scan_sizes",
]
