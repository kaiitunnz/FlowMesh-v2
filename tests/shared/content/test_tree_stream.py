"""Reading a tree manifest as it streams: the members ``TreeManifest.encode`` wrote,
in order, whatever windows the bytes arrive in, and nothing else."""

import tracemalloc
from collections import deque
from collections.abc import Iterator

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
        b'{"schema":1,"members":[[["a","' + b"0" * 64 + b'",1]]]}',
        b'{"schema":1,"members":[["a","' + b"0" * 64 + b'",' + b"9" * 5000 + b"]]}",
        b'{"schema":1,"members":[["a","' + b"0" * 64 + b'",' + b"9" * 20 + b"]]}",
        b'{"schema":1,"members":[["\xff","' + b"0" * 64 + b'",1]]}',
        b'{"schema":1,"members":[["\xc3',
    ],
)
def test_a_manifest_not_written_canonically_is_refused(data: bytes) -> None:
    with pytest.raises(TreeManifestError):
        list(iter_members(_windows(data, 5)))


_MANIFEST = TreeManifest.of(
    [TreeMember(f"m/{i:07d}", _digest(i), i) for i in range(20_000)]
).encode()
_PADDING = 16 << 20


class _Counted:
    """Windows of ``data``, counting the bytes handed out."""

    def __init__(self, data: bytes, size: int) -> None:
        self.data = data
        self.size = size
        self.read = 0

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.size):
            window = self.data[start : start + self.size]
            self.read += len(window)
            yield window


@pytest.mark.parametrize(
    "data",
    [
        _MANIFEST.replace(b'",5]', b'",5' + b"0x" * (_PADDING // 2) + b"]", 1),
        _MANIFEST[:200] + b'["' + b"a" * _PADDING,
        _MANIFEST + b"x" * _PADDING,
    ],
    ids=["bad-digit", "unterminated-path", "trailing"],
)
@pytest.mark.parametrize("size", [7, 64 * 1024])
def test_a_malformed_manifest_is_refused_early_in_bounded_memory(
    data: bytes, size: int
) -> None:
    windows = _Counted(data, size)

    tracemalloc.start()
    try:
        with pytest.raises(TreeManifestError):
            deque(iter_members(windows), maxlen=0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 6 << 20
    assert windows.read < len(_MANIFEST) + (1 << 20)


def test_a_path_longer_than_any_tree_holds_is_refused() -> None:
    with pytest.raises(TreeManifestError):
        TreeManifest.of([TreeMember("a" * (1 << 17), _digest(1), 1)])


@pytest.mark.parametrize("at", [0, 500, len(_MANIFEST) - 1])
def test_bytes_that_are_not_utf8_are_refused_as_a_malformed_manifest(at: int) -> None:
    data = _MANIFEST[:at] + b"\xff" + _MANIFEST[at:]

    with pytest.raises(TreeManifestError):
        list(iter_members(_windows(data, 4096)))
