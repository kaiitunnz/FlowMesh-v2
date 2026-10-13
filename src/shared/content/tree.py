"""A tree of files named as one object: a manifest listing each member's path,
digest and size, stored like any other object.

A tree's members are objects of their own in the same scope, so a reader fetches and
verifies each member by the reference its manifest line names, and a tree never holds
bytes a member does not. The manifest is canonical — members sorted by path, encoded
one way — so the same files always name the same tree.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from shared.utils.nofollow import is_plain_segment

from .reference import OCTET_STREAM, ContentReference, DigestAlgorithm

TREE_MEDIA_TYPE = "application/vnd.flowmesh.tree+json"
_SCHEMA = 1
_HEX = frozenset("0123456789abcdef")


class TreeManifestError(ValueError):
    """Bytes that are not a well-formed tree manifest."""


def is_member_path(path: str) -> bool:
    """Whether ``path`` is a relative path of plain segments."""
    return bool(path) and all(is_plain_segment(part) for part in path.split("/"))


@dataclass(frozen=True, slots=True)
class TreeMember:
    """One file of a tree: where it sits and the object holding its bytes."""

    path: str
    digest: str
    size_bytes: int

    def reference(self, scope: str) -> ContentReference:
        """The member's object in the tree's scope."""
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
    def of(cls, members: Iterable[TreeMember]) -> "TreeManifest":
        """The canonical manifest of ``members``, refusing a bad or repeated path."""
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
        """The member holding the most bytes, or None for an empty tree."""
        return max(self.members, key=lambda member: member.size_bytes, default=None)

    def member(self, path: str) -> TreeMember | None:
        """The member at ``path``, or None."""
        low, high = 0, len(self.members)
        while low < high:
            mid = (low + high) // 2
            if self.members[mid].path < path:
                low = mid + 1
            else:
                high = mid
        if low < len(self.members) and self.members[low].path == path:
            return self.members[low]
        return None

    def under(self, prefix: str) -> Iterator[TreeMember]:
        """The members at or below directory ``prefix``; every member for ``""``."""
        if not prefix:
            yield from self.members
            return
        for member in self.members:
            if member.path.startswith(prefix + "/"):
                yield member

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
    def decode(cls, data: bytes) -> "TreeManifest":
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
]
