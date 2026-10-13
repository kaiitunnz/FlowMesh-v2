"""A tree served as a ``.tar.gz`` whose bytes follow from the tree alone.

The tar is deterministic — members in manifest order under one top directory, every
header fixed but for path and size, pax records only where ustar cannot hold a value —
and the gzip around it stores its deflate blocks uncompressed, framed at a fixed size.
Every byte of the archive is therefore a pure function of the manifest and the members'
bytes: a producer hashes the archive once while it hashes the members, and a reader
serves any byte range by arithmetic over the manifest, reading only the members that
range overlaps.
"""

import bisect
import hashlib
import struct
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .tree import TreeManifest, TreeMember

ARCHIVE_MEDIA_TYPE = "application/gzip"

_BLOCK = 512
# A tar ends on a whole record of twenty blocks, as tar writes it by default.
_RECORD = 20 * _BLOCK
_USTAR_NAME_BYTES = 100
_USTAR_SIZE_LIMIT = 8**11
# The most one stored deflate block holds.
STORED_BLOCK_BYTES = 0xFFFF
_GZIP_HEADER = bytes((0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF))
_BLOCK_HEADER_BYTES = 5
_TRAILER_BYTES = 8
_STRIDE = STORED_BLOCK_BYTES + _BLOCK_HEADER_BYTES

# Reads ``length`` bytes of a member from ``start``.
type MemberReader = Callable[[TreeMember, int, int], Iterator[bytes]]


def _octal(value: int, width: int) -> bytes:
    return f"{value:0{width - 1}o}".encode() + b"\0"


def _ustar_header(name: bytes, size: int, typeflag: bytes) -> bytes:
    header = bytearray(_BLOCK)
    header[0 : len(name)] = name
    header[100:108] = _octal(0o644, 8)
    header[108:116] = _octal(0, 8)
    header[116:124] = _octal(0, 8)
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(0, 12)
    header[148:156] = b" " * 8
    header[156:157] = typeflag
    header[257:265] = b"ustar\x0000"
    header[329:337] = _octal(0, 8)
    header[337:345] = _octal(0, 8)
    header[148:156] = f"{sum(header):06o}".encode() + b"\0 "
    return bytes(header)


def _pax_record(key: str, value: bytes) -> bytes:
    body = f" {key}=".encode() + value + b"\n"
    length = len(body) + 1
    while len(str(length).encode()) + len(body) != length:
        length = len(str(length).encode()) + len(body)
    return str(length).encode() + body


def _padding(size: int) -> int:
    return -size % _BLOCK


def member_headers(path: str, size: int) -> bytes:
    """The header blocks a member's data follows: a pax header where the path is too
    long or the size too large for ustar, then the ustar header."""
    name = path.encode()
    records = b""
    if len(name) > _USTAR_NAME_BYTES:
        records += _pax_record("path", name)
    if size >= _USTAR_SIZE_LIMIT:
        records += _pax_record("size", str(size).encode())
    ustar = _ustar_header(
        name[:_USTAR_NAME_BYTES], size if size < _USTAR_SIZE_LIMIT else 0, b"0"
    )
    if not records:
        return ustar
    pax = _ustar_header(b"././@PaxHeader", len(records), b"x")
    return pax + records + bytes(_padding(len(records))) + ustar


@dataclass(frozen=True, slots=True)
class _Span:
    """A run of the tar stream: fixed bytes, or a member's data."""

    offset: int
    length: int
    data: bytes | None = None
    member: TreeMember | None = None


class TreeArchive:
    """The stored-block ``.tar.gz`` of a tree, with its members under ``top``."""

    def __init__(self, manifest: TreeManifest, top: str) -> None:
        spans: list[_Span] = []
        offset = 0
        for member in manifest.members:
            headers = member_headers(f"{top}/{member.path}", member.size_bytes)
            spans.append(_Span(offset, len(headers), data=headers))
            offset += len(headers)
            if member.size_bytes:
                spans.append(_Span(offset, member.size_bytes, member=member))
                offset += member.size_bytes
            if pad := _padding(member.size_bytes):
                spans.append(_Span(offset, pad, data=bytes(pad)))
                offset += pad
        end = offset + 2 * _BLOCK
        end += -end % _RECORD
        spans.append(_Span(offset, end - offset, data=bytes(end - offset)))
        self._spans = spans
        self._starts = [span.offset for span in spans]
        self.tar_size = end
        self._blocks = -(-end // STORED_BLOCK_BYTES)

    @property
    def size_bytes(self) -> int:
        return (
            len(_GZIP_HEADER)
            + self.tar_size
            + _BLOCK_HEADER_BYTES * self._blocks
            + _TRAILER_BYTES
        )

    def _tar(self, start: int, end: int, read: MemberReader) -> Iterator[bytes]:
        index = bisect.bisect_right(self._starts, start) - 1
        while start < end:
            span = self._spans[index]
            stop = min(end, span.offset + span.length)
            skip = start - span.offset
            if span.member is not None:
                yield from read(span.member, skip, stop - start)
            else:
                assert span.data is not None
                yield span.data[skip : skip + stop - start]
            start = stop
            index += 1

    def _block_length(self, block: int) -> int:
        return min(STORED_BLOCK_BYTES, self.tar_size - block * STORED_BLOCK_BYTES)

    def _block_header(self, block: int) -> bytes:
        length = self._block_length(block)
        final = 1 if block == self._blocks - 1 else 0
        return struct.pack("<BHH", final, length, length ^ 0xFFFF)

    def _tar_position(self, position: int) -> int:
        """The tar offset of the first tar byte at or after archive ``position``,
        for a position within the deflate body."""
        block, within = divmod(position - len(_GZIP_HEADER), _STRIDE)
        return min(
            self.tar_size,
            block * STORED_BLOCK_BYTES + max(0, within - _BLOCK_HEADER_BYTES),
        )

    def chunks(
        self,
        read: MemberReader,
        *,
        crc32: int,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[bytes]:
        """The archive's bytes ``[start, end)``, its members read through ``read``.

        ``crc32`` is the tar stream's CRC-32, which the trailer carries.
        """
        size = self.size_bytes
        end = size if end is None else end
        if not 0 <= start <= end <= size:
            raise ValueError(f"bytes {start}-{end} are outside the archive")
        head = len(_GZIP_HEADER)
        body_end = size - _TRAILER_BYTES
        if start < head:
            yield _GZIP_HEADER[start : min(end, head)]
        low, high = max(start, head), min(end, body_end)
        if low < high:
            tar = _Rechunk(
                self._tar(self._tar_position(low), self._tar_position(high), read)
            )
            first = (low - head) // _STRIDE
            last = (high - 1 - head) // _STRIDE
            for block in range(first, last + 1):
                header_start = head + block * _STRIDE
                data_start = header_start + _BLOCK_HEADER_BYTES
                data_end = data_start + self._block_length(block)
                if low < data_start and header_start < high:
                    yield self._block_header(block)[
                        max(low, header_start)
                        - header_start : min(high, data_start)
                        - header_start
                    ]
                if (want := min(high, data_end) - max(low, data_start)) > 0:
                    yield from tar.take(want)
        if end > body_end:
            trailer = struct.pack("<II", crc32 & 0xFFFFFFFF, self.tar_size & 0xFFFFFFFF)
            yield trailer[max(0, start - body_end) : end - body_end]

    def tar_chunks(self, read: MemberReader) -> Iterator[bytes]:
        """The whole tar stream the archive frames."""
        return self._tar(0, self.tar_size, read)


class _Rechunk:
    """Takes exact byte counts off an iterator of arbitrary chunks."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._held = b""

    def take(self, count: int) -> Iterator[bytes]:
        while count:
            if not self._held:
                self._held = next(self._chunks)
                continue
            piece, self._held = self._held[:count], self._held[count:]
            count -= len(piece)
            yield piece


@dataclass(frozen=True, slots=True)
class ArchiveDigest:
    """What a reader checks a whole archive against: its SHA-256 and size, and the
    tar stream's CRC-32 its trailer carries."""

    sha256: str
    crc32: int
    size_bytes: int


def digest_archive(archive: TreeArchive, read: MemberReader) -> ArchiveDigest:
    """Hash the whole archive in one pass over its members."""
    sha = hashlib.sha256(_GZIP_HEADER)
    crc = 0
    block = 0
    filled = 0
    for chunk in archive.tar_chunks(read):
        view = memoryview(chunk)
        while view:
            if not filled:
                sha.update(archive._block_header(block))
            piece = view[: STORED_BLOCK_BYTES - filled]
            sha.update(piece)
            crc = zlib.crc32(piece, crc)
            filled += len(piece)
            view = view[len(piece) :]
            if filled == STORED_BLOCK_BYTES:
                block, filled = block + 1, 0
    sha.update(struct.pack("<II", crc, archive.tar_size & 0xFFFFFFFF))
    return ArchiveDigest(sha.hexdigest(), crc, archive.size_bytes)


__all__ = [
    "ARCHIVE_MEDIA_TYPE",
    "ArchiveDigest",
    "MemberReader",
    "STORED_BLOCK_BYTES",
    "TreeArchive",
    "member_headers",
    "digest_archive",
]
