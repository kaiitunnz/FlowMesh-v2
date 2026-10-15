"""A tree's archive view is a genuine .tar.gz whose bytes follow from the tree alone,
and any range of it is the same bytes whether read whole or in part."""

import gzip
import hashlib
import io
import os
import random
import shutil
import subprocess  # nosec B404 — the system gzip and tar check the archive
import tarfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from shared.content import TreeManifest, TreeMember
from shared.content.archive import (
    STORED_BLOCK_BYTES,
    MemberReader,
    TreeArchive,
    digest_archive,
    member_headers,
)

_FILES = {
    "config.json": b"{}",
    "model.bin": os.urandom(3 * STORED_BLOCK_BYTES + 11),
    "nested/" + "n" * 140 + ".bin": os.urandom(700),
    "empty": b"",
}


def _tree(files: dict[str, bytes]) -> tuple[TreeArchive, dict[str, bytes]]:
    manifest = TreeManifest.of(
        TreeMember(path, hashlib.sha256(data).hexdigest(), len(data))
        for path, data in files.items()
    )
    return TreeArchive(manifest, "final_model"), {
        hashlib.sha256(data).hexdigest(): data for data in files.values()
    }


def _reader(by_digest: dict[str, bytes]) -> MemberReader:
    def read(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        data = by_digest[member.digest][start : start + length]
        for offset in range(0, len(data), 997):
            yield data[offset : offset + 997]

    return read


def _whole(files: dict[str, bytes]) -> tuple[bytes, TreeArchive, int]:
    archive, by_digest = _tree(files)
    read = _reader(by_digest)
    digest = digest_archive(archive, read)
    data = b"".join(archive.chunks(read, crc32=digest.crc32))
    assert hashlib.sha256(data).hexdigest() == digest.sha256
    assert len(data) == digest.size_bytes == archive.size_bytes
    return data, archive, digest.crc32


def test_the_archive_is_a_gzip_tar_of_the_tree() -> None:
    data, _, crc = _whole(_FILES)

    raw = gzip.decompress(data)
    assert zlib.crc32(raw) == crc
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        found = {}
        for member in tar.getmembers():
            assert member.mtime == 0 and member.uid == 0 and member.gid == 0
            assert member.mode == 0o644 and member.uname == "" == member.gname
            handle = tar.extractfile(member)
            assert handle is not None
            found[member.name] = handle.read()
    assert found == {f"final_model/{path}": body for path, body in _FILES.items()}
    assert len(raw) % (20 * 512) == 0


def test_the_same_tree_always_gives_the_same_archive() -> None:
    first, _, _ = _whole(_FILES)
    second, _, _ = _whole(dict(reversed(list(_FILES.items()))))

    assert first == second


@pytest.mark.skipif(
    shutil.which("gzip") is None or shutil.which("tar") is None,
    reason="needs the system gzip and tar",
)
def test_the_system_gzip_and_tar_accept_the_archive(tmp_path: Path) -> None:
    data, _, _ = _whole(_FILES)
    path = tmp_path / "final_model.tar.gz"
    path.write_bytes(data)
    gzip_bin, tar_bin = shutil.which("gzip"), shutil.which("tar")
    assert gzip_bin is not None and tar_bin is not None

    subprocess.run([gzip_bin, "-t", str(path)], check=True)  # nosec B603 — argv list
    subprocess.run(  # nosec B603 — argv list, absolute path via shutil.which()
        [tar_bin, "xzf", str(path), "-C", str(tmp_path)], check=True
    )

    for name, body in _FILES.items():
        assert (tmp_path / "final_model" / name).read_bytes() == body


def test_every_range_reads_the_bytes_the_whole_holds() -> None:
    data, archive, crc = _whole(_FILES)
    archive_, by_digest = _tree(_FILES)
    read = _reader(by_digest)
    rng = random.Random(7)
    block = 10 + STORED_BLOCK_BYTES
    edges = [0, 9, 10, 14, 15, 512, 1024, block, block + 4, block + 5, len(data) - 8]
    spans = [(start, end) for start in edges for end in (start + 1, start + 600)]
    spans += [
        (start, rng.randrange(start, len(data) + 1))
        for start in (rng.randrange(len(data)) for _ in range(300))
    ]

    for start, end in spans:
        end = min(end, len(data))
        got = b"".join(archive.chunks(read, crc32=crc, start=start, end=end))
        assert got == data[start:end], (start, end)


def test_a_member_too_large_for_ustar_gets_a_pax_size() -> None:
    headers = member_headers("final_model/huge.bin", 8**11)

    assert b" size=8589934592\n" in headers
    assert len(headers) % 512 == 0


def test_a_member_that_fails_its_read_stops_the_archive() -> None:
    archive, by_digest = _tree(_FILES)
    read = _reader(by_digest)
    crc = digest_archive(archive, read).crc32

    def failing(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        if member.size_bytes > 1000:
            raise RuntimeError("digest mismatch")
        yield from read(member, start, length)

    produced = []
    with pytest.raises(RuntimeError):
        for chunk in archive.chunks(failing, crc32=crc):
            produced.append(chunk)
    assert sum(map(len, produced)) < archive.size_bytes - 8
