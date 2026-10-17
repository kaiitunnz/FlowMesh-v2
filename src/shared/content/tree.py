"""A tree of files named as one object: a manifest listing each member's path,
digest and size, stored like any other object.

A tree's members are objects of their own in the same scope, so a reader fetches and
verifies each member by the reference its manifest line names, and a tree never holds
bytes a member does not. The manifest is canonical — members sorted by path, encoded
one way — so the same files always name the same tree.
"""

import bisect
import codecs
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import attrgetter
from typing import Self

from shared.utils.nofollow import is_plain_segment

from .reference import OCTET_STREAM, ContentReference, DigestAlgorithm

TREE_MEDIA_TYPE = "application/vnd.flowmesh.tree+json"
_SCHEMA = 1
_HEX = frozenset("0123456789abcdef")


class TreeManifestError(ValueError):
    """Bytes that are not a well-formed tree manifest."""


def is_member_path(path: str) -> bool:
    """Return whether ``path`` is a relative path of plain segments."""
    return bool(path) and all(is_plain_segment(part) for part in path.split("/"))


@dataclass(frozen=True, slots=True)
class TreeMember:
    """One file of a tree: where it sits and the object holding its bytes."""

    path: str
    digest: str
    size_bytes: int

    def reference(self, scope: str) -> ContentReference:
        """Return the member's object in the tree's scope."""
        return ContentReference(
            authorization_scope=scope,
            digest_algorithm=DigestAlgorithm.SHA256,
            content_digest=self.digest,
            size_bytes=self.size_bytes,
            media_type=OCTET_STREAM,
        )


@dataclass(frozen=True, slots=True)
class TreeManifest:
    """A tree's members, sorted by path."""

    members: tuple[TreeMember, ...]

    @classmethod
    def of(cls, members: Iterable[TreeMember]) -> Self:
        """Build the canonical manifest of ``members``, refusing a bad or repeated
        path."""
        ordered = tuple(sorted(members, key=lambda member: member.path))
        for index, member in enumerate(ordered):
            _check_member(member)
            if index and ordered[index - 1].path == member.path:
                raise TreeManifestError(f"tree member {member.path!r} is repeated")
        return cls(ordered)

    @property
    def size_bytes(self) -> int:
        return sum(member.size_bytes for member in self.members)

    def largest(self) -> TreeMember | None:
        """Return the member holding the most bytes, or None for an empty tree."""
        return max(self.members, key=lambda member: member.size_bytes, default=None)

    def member(self, path: str) -> TreeMember | None:
        """Return the member at ``path``, or None."""
        index = bisect.bisect_left(self.members, path, key=attrgetter("path"))
        if index < len(self.members) and self.members[index].path == path:
            return self.members[index]
        return None

    def encode(self) -> bytes:
        return json.dumps(
            {
                "schema": _SCHEMA,
                "members": [
                    [member.path, member.digest, member.size_bytes]
                    for member in self.members
                ],
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()

    @classmethod
    def decode(cls, data: bytes) -> Self:
        try:
            document = json.loads(data)
        except (ValueError, RecursionError) as exc:
            raise TreeManifestError(f"tree manifest is not JSON: {exc}") from exc
        if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
            raise TreeManifestError("tree manifest has an unknown schema")
        entries = document.get("members")
        if not isinstance(entries, list):
            raise TreeManifestError("tree manifest has no member list")
        members: list[TreeMember] = []
        for entry in entries:
            match entry:
                case [str() as path, str() as digest, int() as size]:
                    members.append(TreeMember(path, digest, size))
                case _:
                    raise TreeManifestError(f"malformed tree member {entry!r}")
        manifest = cls.of(members)
        if manifest.members != tuple(members):
            raise TreeManifestError("tree manifest members are not in path order")
        return manifest


_PREFIX = f'{{"schema":{_SCHEMA},"members":['
_DECODER = json.JSONDecoder()
# Past any path a published tree's walk reaches, so a member line stays bounded.
_MAX_PATH_CHARS = 1 << 16
_MAX_SIZE_BYTES = (1 << 63) - 1


def _entry_text(member: TreeMember) -> str:
    return json.dumps(
        [member.path, member.digest, member.size_bytes],
        separators=(",", ":"),
        ensure_ascii=False,
    )


_ENTRY_OPEN = '["'
# The longest a valid member encodes to: the longest path, each character escaped.
_MAX_ENTRY_CHARS = len(
    _entry_text(TreeMember("\0" * _MAX_PATH_CHARS, "0" * 64, _MAX_SIZE_BYTES))
)


def iter_members(chunks: Iterable[bytes]) -> Iterator[TreeMember]:
    """Yield the members of a manifest's canonical encoding as its bytes stream in,
    refusing what ``TreeManifest.decode`` refuses, holding one window of text and one
    member at a time.

    Only the encoding ``TreeManifest.encode`` writes is read, so text that cannot
    begin it, a path past any a published walk reaches, a member line longer than
    such a member encodes to, or a byte past the closing brace is refused once it is
    read. A caller verifying the bytes by
    digest learns of a corrupt manifest only once the stream ends, so it acts on no
    member as settled before then.
    """
    reader = _MemberReader()
    for chunk in chunks:
        yield from reader.feed(chunk)
    yield from reader.finish()


class _MemberReader:
    """Reads members from a canonical manifest's text, window by window.

    A member a window cuts off is decoded again only once as much text again has
    arrived, so text is copied and scanned a bounded number of times however small
    the windows.
    """

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._pos = 0
        self._arrived: list[str] = []
        self._arrived_chars = 0
        self._wait = 0
        self._started = self._ended = False
        self._previous: str | None = None

    def feed(self, chunk: bytes) -> Iterator[TreeMember]:
        text = self._decode(chunk, final=False)
        self._arrived.append(text)
        self._arrived_chars += len(text)
        if self._arrived_chars < self._wait:
            return
        yield from self._scan()

    def finish(self) -> Iterator[TreeMember]:
        self._arrived.append(self._decode(b"", final=True))
        yield from self._scan()
        if not self._started or not self._ended or self._rest() != "}":
            raise TreeManifestError("tree manifest is truncated or has trailing bytes")

    def _rest(self) -> str:
        return self._buffer[self._pos :]

    def _decode(self, data: bytes, final: bool) -> str:
        try:
            return self._utf8.decode(data, final)
        except UnicodeDecodeError as exc:
            raise TreeManifestError(f"tree manifest is not UTF-8: {exc}") from exc

    def _scan(self) -> Iterator[TreeMember]:
        buffer = self._buffer = "".join([self._rest(), *self._arrived])
        self._arrived.clear()
        self._arrived_chars = self._wait = self._pos = 0
        if not self._started:
            if not _PREFIX.startswith(buffer[: len(_PREFIX)]):
                raise TreeManifestError("tree manifest has an unknown schema")
            if len(buffer) < len(_PREFIX):
                return
            self._pos, self._started = len(_PREFIX), True
        while not self._ended and self._pos < len(buffer):
            pos = self._pos
            if buffer[pos] == "]":
                self._ended = True
                self._pos = pos + 1
                break
            if self._previous is not None:
                if buffer[pos] != ",":
                    raise TreeManifestError("malformed tree member list")
                pos += 1
            if not _ENTRY_OPEN.startswith(buffer[pos : pos + len(_ENTRY_OPEN)]):
                raise TreeManifestError("malformed tree member list")
            try:
                entry, end = _DECODER.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if len(buffer) - pos > _MAX_ENTRY_CHARS:
                    raise TreeManifestError("tree member is too long") from None
                self._wait = len(buffer) - self._pos
                return
            except (ValueError, RecursionError) as exc:
                raise TreeManifestError(f"malformed tree member: {exc}") from exc
            match entry:
                case [str() as path, str() as digest, int() as size]:
                    member = TreeMember(path, digest, size)
                case _:
                    raise TreeManifestError(f"malformed tree member {entry!r}")
            _check_member(member)
            _check_bounds(member)
            if self._previous is not None and path <= self._previous:
                raise TreeManifestError(
                    f"tree member {path!r} is repeated or out of path order"
                )
            self._previous = path
            self._pos = end
            yield member
        if self._ended and not "}".startswith(self._rest()):
            raise TreeManifestError("tree manifest has trailing bytes")


def _check_member(member: TreeMember) -> None:
    if not is_member_path(member.path):
        raise TreeManifestError(f"tree member path {member.path!r} is not plain")
    if len(member.digest) != 64 or not set(member.digest) <= _HEX:
        raise TreeManifestError(f"tree member {member.path!r} has a bad digest")
    if member.size_bytes < 0:
        raise TreeManifestError(f"tree member {member.path!r} has a negative size")


def _check_bounds(member: TreeMember) -> None:
    if len(member.path) > _MAX_PATH_CHARS:
        raise TreeManifestError(f"tree member path {member.path[:64]!r}... is too long")
    if member.size_bytes > _MAX_SIZE_BYTES:
        raise TreeManifestError(f"tree member {member.path!r} has a bad size")


__all__ = [
    "TREE_MEDIA_TYPE",
    "TreeManifest",
    "TreeManifestError",
    "TreeMember",
    "is_member_path",
    "iter_members",
]
