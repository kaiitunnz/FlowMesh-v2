"""A tree manifest names the same files the same way, and refuses a path that could
leave the tree."""

import pytest

from shared.content import TreeManifest, TreeManifestError, TreeMember

_D = "ab" * 32


def test_the_same_members_in_any_order_encode_alike() -> None:
    members = [TreeMember("b/c", _D, 2), TreeMember("a", _D, 1)]

    first = TreeManifest.of(members)
    second = TreeManifest.of(reversed(members))

    assert first.encode() == second.encode()
    assert TreeManifest.decode(first.encode()) == first
    assert [member.path for member in first.members] == ["a", "b/c"]


@pytest.mark.parametrize("path", ["", "/a", "a/../b", "a//b", "./a", "a\0b", "a/"])
def test_a_path_that_is_not_plain_is_refused(path: str) -> None:
    with pytest.raises(TreeManifestError):
        TreeManifest.of([TreeMember(path, _D, 1)])


def test_a_repeated_path_is_refused() -> None:
    with pytest.raises(TreeManifestError):
        TreeManifest.of([TreeMember("a", _D, 1), TreeMember("a", _D, 2)])


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b'{"schema": 2, "members": []}',
        b'{"schema": 1, "members": [["b", "'
        + _D.encode()
        + b'", 1], ["a", "'
        + _D.encode()
        + b'", 1]]}',
        b'{"schema": 1, "members": [["a", "short", 1]]}',
    ],
)
def test_malformed_manifest_bytes_are_refused(data: bytes) -> None:
    with pytest.raises(TreeManifestError):
        TreeManifest.decode(data)


def test_lookup_finds_a_member_and_a_directory() -> None:
    manifest = TreeManifest.of(
        [TreeMember("a/x", _D, 1), TreeMember("a/y", _D, 5), TreeMember("ab", _D, 2)]
    )

    assert manifest.member("a/y") == TreeMember("a/y", _D, 5)
    assert manifest.member("a") is None
    assert manifest.largest() == TreeMember("a/y", _D, 5)
    assert manifest.size_bytes == 8


def test_a_large_tree_encodes_compactly() -> None:
    manifest = TreeManifest.of(
        TreeMember(f"shard-{index:06d}.bin", _D, index) for index in range(100_000)
    )

    assert len(manifest.encode()) < 100 * 100_000
