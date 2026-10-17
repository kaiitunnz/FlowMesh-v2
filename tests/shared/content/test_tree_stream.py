"""Reading a tree manifest as it streams: the members ``TreeManifest.encode`` wrote,
in order, whatever windows the bytes arrive in, and nothing else."""

import pytest

from shared.content.tree import (
    TreeManifest,
    TreeManifestError,
    TreeMember,
    iter_members,
)


def _digest(n: int) -> str:
    return f"{n:064x}"


TREES = [
    TreeManifest.of([]),
    TreeManifest.of([TreeMember("a.bin", _digest(1), 0)]),
    TreeManifest.of(
        [
            TreeMember("é/漢字.txt", _digest(2), 7),
            TreeMember('q"uote\\back', _digest(3), 12345678901),
            TreeMember("z/" + "deep/" * 20 + "f", _digest(4), 1),
            TreeMember("tab\tnewline\nx", _digest(5), 2),
        ]
    ),
    TreeManifest.of([TreeMember(f"m/{i:05d}", _digest(i), i) for i in range(300)]),
]


def _windows(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)] or [b""]


@pytest.mark.parametrize("tree", TREES, ids=lambda t: f"{len(t.members)}")
@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 1 << 20])
def test_every_window_reads_the_members_encode_wrote(
    tree: TreeManifest, size: int
) -> None:
    assert tuple(iter_members(_windows(tree.encode(), size))) == tree.members


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b'{"schema":2,"members":[]}',
        b'{"schema":1,"members":[]',
        b'{"schema":1,"members":[]}x',
        b'{"schema":1,"members":[,]}',
        b'{"schema":1,"members":[["a","' + b"0" * 64 + b'",1],]}',
        b'{"schema":1,"members":[["a","' + b"0" * 63 + b'",1]]}',
        b'{"schema":1,"members":[["../a","' + b"0" * 64 + b'",1]]}',
        b'{"schema":1,"members":[["a","' + b"0" * 64 + b'",-1]]}',
        b'{"schema":1,"members":[["b","'
        + b"0" * 64
        + b'",1],["a","'
        + b"0" * 64
        + b'",1]]}',
        b'{"schema":1,"members":[["a","'
        + b"0" * 64
        + b'",1],["a","'
        + b"0" * 64
        + b'",1]]}',
        b'{"schema":1, "members":[]}',
        b'{"schema":1,"members":[["a","' + b"0" * 64 + b'"]]}',
    ],
)
def test_a_manifest_not_written_canonically_is_refused(data: bytes) -> None:
    with pytest.raises(TreeManifestError):
        list(iter_members(_windows(data, 5)))
