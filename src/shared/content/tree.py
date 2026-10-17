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
        except ValueError as exc:
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


def iter_members(chunks: Iterable[bytes]) -> Iterator[TreeMember]:
    """Yield the members of a manifest's canonical encoding as its bytes stream in,
    refusing what ``TreeManifest.decode`` refuses, holding one window of text and one
    member at a time.

    Only the encoding ``TreeManifest.encode`` writes is read; a caller verifying the
    bytes by digest learns of a corrupt manifest only once the stream ends, so it acts
    on no member as settled before then.
    """
    text = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    pos = 0
    started = ended = False
    previous: str | None = None
    for chunk in chunks:
        buffer = buffer[pos:] + text.decode(chunk)
        pos = 0
        if not started:
            if len(buffer) < len(_PREFIX):
                continue
            if not buffer.startswith(_PREFIX):
                raise TreeManifestError("tree manifest has an unknown schema")
            pos, started = len(_PREFIX), True
        while not ended:
            if pos >= len(buffer):
                break
            if buffer[pos] == "]":
                if previous is None and buffer[pos - 1] == ",":
                    raise TreeManifestError("malformed tree member list")
                ended = True
                pos += 1
                break
            if previous is not None:
                if buffer[pos] != ",":
                    raise TreeManifestError("malformed tree member list")
                if pos + 1 >= len(buffer):
                    break
                start = pos + 1
            else:
                start = pos
            try:
                entry, end = _DECODER.raw_decode(buffer, start)
            except json.JSONDecodeError:
                # An entry the window cuts off completes with the next chunk.
                break
            match entry:
                case [str() as path, str() as digest, int() as size]:
                    member = TreeMember(path, digest, size)
                case _:
                    raise TreeManifestError(f"malformed tree member {entry!r}")
            _check_member(member)
            if previous is not None and path <= previous:
                raise TreeManifestError(
                    f"tree member {path!r} is repeated or out of path order"
                )
            previous = path
            pos = end
            yield member
    buffer = buffer[pos:] + text.decode(b"", final=True)
    if not started or not ended or buffer != "}":
        raise TreeManifestError("tree manifest is truncated or has trailing bytes")


def _check_member(member: TreeMember) -> None:
    if not is_member_path(member.path):
        raise TreeManifestError(f"tree member path {member.path!r} is not plain")
    if len(member.digest) != 64 or not set(member.digest) <= _HEX:
        raise TreeManifestError(f"tree member {member.path!r} has a bad digest")
    if member.size_bytes < 0:
        raise TreeManifestError(f"tree member {member.path!r} has a negative size")


__all__ = [
    "TREE_MEDIA_TYPE",
    "TreeManifest",
    "TreeManifestError",
    "TreeMember",
    "is_member_path",
    "iter_members",
]
