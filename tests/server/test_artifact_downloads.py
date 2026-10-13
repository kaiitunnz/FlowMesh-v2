"""The root serves a task's published artifacts from the shared store: listed, whole
and verified, by range, as an archive view, and bundled."""

import base64
import gzip
import hashlib
import io
import logging
import tarfile
import threading
import tracemalloc
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException, status

from server.routers.v1 import results as results_router
from server.task import artifact_reads
from server.task.results import ResultReader, ResultUnreadable
from shared.content import (
    ContentReference,
    ContentUnavailable,
    SharedFilesystemObjectStore,
)
from shared.content.archive import STORED_BLOCK_BYTES
from shared.tasks.artifact_binding import ArtifactBinding
from tests.server.asgi_responses import ClientGone, serve
from worker.content.artifacts import ArtifactPublisher

_LOGGER = logging.getLogger("test.artifact_downloads")
_PUBLISHED_AT = "2026-10-14T01:02:03+00:00"
_HTTP_DATE = "Wed, 14 Oct 2026 01:02:03 GMT"
_WEIGHTS = bytes(range(256)) * 600
_FILES = {
    "config.json": b'{"hidden": 8}',
    "weights/model.safetensors": _WEIGHTS,
    "tokenizer.json": b"t" * 1000,
}


class _Runtime:
    """Serves one task's bindings over a filesystem store."""

    def __init__(self, root: Path) -> None:
        self.store = SharedFilesystemObjectStore(root / "cas")
        self.result_reader = ResultReader(self.store)
        self.bindings: dict[str, dict[tuple[str, str], ArtifactBinding]] = {}
        self.results: dict[str, bytes] = {}

    def artifact_bindings(
        self, task_id: str
    ) -> dict[tuple[str, str], ArtifactBinding] | None:
        return self.bindings.get(task_id)

    def read_result_bytes(self, task_id: str) -> bytes | None:
        return self.results.get(task_id)

    def get_record(self, task_id: str) -> None:
        return None


def _publish(tmp_path: Path, runtime: _Runtime) -> dict[str, ArtifactBinding]:
    artifacts = tmp_path / "out" / "artifacts"
    for path, data in _FILES.items():
        (artifacts / "final_model" / path).parent.mkdir(parents=True, exist_ok=True)
        (artifacts / "final_model" / path).write_bytes(data)
    (artifacts / "notes.txt").write_bytes(b"trained for 4 steps\n")
    checkpoint = tmp_path / "spool" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_bytes(b'{"step": 2}')
    publisher = ArtifactPublisher(runtime.store, "org")
    published = publisher.publish_final(
        artifacts, {"final_model.tar.gz": "final_model"}
    )
    step, _ = publisher.publish_tree("checkpoints", tmp_path / "spool", "step-2")
    accepted = {
        binding.key: binding.accepted("dsp-1", _PUBLISHED_AT)
        for binding in [*published, step]
    }
    runtime.bindings["tsk-1"] = accepted
    runtime.results["tsk-1"] = b'{"task_id": "tsk-1", "result": {}}'
    return {name: binding for (_, name), binding in accepted.items()}


@pytest.fixture
def runtime(tmp_path: Path) -> _Runtime:
    return _Runtime(tmp_path)


@pytest.fixture
def published(tmp_path: Path, runtime: _Runtime) -> dict[str, ArtifactBinding]:
    return _publish(tmp_path, runtime)


async def _download(
    runtime: _Runtime, filename: str, revision: str = "final"
) -> results_router.Response:
    return await results_router.download_result_file(
        task_id="tsk-1",
        filename=filename,
        revision=revision,
        principal=cast(Any, None),
        runtime=cast(Any, runtime),
        logger=_LOGGER,
    )


def _repr_digest(data: bytes) -> str:
    return f"sha-256=:{base64.b64encode(hashlib.sha256(data).digest()).decode()}:"


@pytest.mark.anyio
async def test_the_listing_names_every_revision_or_one(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    async def listed(revision: str | None) -> list[tuple[str, str, str]]:
        page = await results_router.list_result_artifacts(
            task_id="tsk-1",
            revision=revision,
            principal=cast(Any, None),
            runtime=cast(Any, runtime),
            logger=_LOGGER,
        )
        return [(a.revision, a.name, a.kind) for a in page.artifacts]

    assert await listed(None) == [
        ("step-2", "checkpoints", "tree"),
        ("final", "final_model", "tree"),
        ("final", "final_model.tar.gz", "archive"),
        ("final", "notes.txt", "file"),
    ]
    assert await listed("step-2") == [("step-2", "checkpoints", "tree")]
    page = await results_router.list_result_artifacts(
        task_id="tsk-1",
        revision="final",
        principal=cast(Any, None),
        runtime=cast(Any, runtime),
        logger=_LOGGER,
    )
    by_name = {artifact.name: artifact for artifact in page.artifacts}
    archive = published["final_model.tar.gz"].archive
    assert archive is not None
    assert by_name["final_model.tar.gz"].sha256 == archive.sha256
    assert by_name["final_model.tar.gz"].archive_of == "final_model"
    assert by_name["notes.txt"].sha256 == hashlib.sha256(_notes()).hexdigest()
    assert by_name["final_model"].files == len(_FILES)
    assert by_name["final_model"].published_at == _PUBLISHED_AT


def _notes() -> bytes:
    return b"trained for 4 steps\n"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("task_id", "revision", "code"),
    [("tsk-none", None, 404), ("tsk-1", "latest", 400)],
)
async def test_the_listing_refuses_an_unknown_task_or_revision(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    task_id: str,
    revision: str | None,
    code: int,
) -> None:
    with pytest.raises(HTTPException) as raised:
        await results_router.list_result_artifacts(
            task_id=task_id,
            revision=revision,
            principal=cast(Any, None),
            runtime=cast(Any, runtime),
            logger=_LOGGER,
        )
    assert raised.value.status_code == code


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("filename", "revision", "data"),
    [
        ("notes.txt", "final", _notes()),
        ("final_model/weights/model.safetensors", "final", _WEIGHTS),
        ("final_model/config.json", "final", _FILES["config.json"]),
        ("checkpoints/checkpoint-2/trainer_state.json", "step-2", b'{"step": 2}'),
    ],
)
async def test_a_published_file_downloads_whole_with_its_digest(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    filename: str,
    revision: str,
    data: bytes,
) -> None:
    served = await serve(await _download(runtime, filename, revision))

    assert served.status == 200
    assert served.body == data
    assert served.headers["etag"] == f'"sha256:{hashlib.sha256(data).hexdigest()}"'
    assert served.headers["repr-digest"] == _repr_digest(data)
    assert served.headers["last-modified"] == _HTTP_DATE
    assert served.headers["content-length"] == str(len(data))
    assert served.headers["accept-ranges"] == "bytes"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("filename", "revision"),
    [
        ("final_model", "final"),
        ("final_model/missing.bin", "final"),
        ("notes.txt/inner", "final"),
        ("final_model.tar.gz/x", "final"),
        ("unknown.bin", "final"),
        ("notes.txt", "step-2"),
    ],
)
async def test_a_name_no_published_file_has_is_not_found(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    filename: str,
    revision: str,
) -> None:
    with pytest.raises(HTTPException) as raised:
        await _download(runtime, filename, revision)
    assert raised.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("header", "start", "end"),
    [
        ("bytes=10-19", 10, 20),
        ("bytes=1000-", 1000, len(_WEIGHTS)),
        ("bytes=-300", len(_WEIGHTS) - 300, len(_WEIGHTS)),
        ("bytes=0-99999999", 0, len(_WEIGHTS)),
    ],
)
async def test_a_range_is_the_slice_it_names_under_the_whole_files_digest(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    header: str,
    start: int,
    end: int,
) -> None:
    served = await serve(
        await _download(runtime, "final_model/weights/model.safetensors"),
        {"Range": header},
    )

    assert served.status == 206
    assert served.body == _WEIGHTS[start:end]
    assert served.headers["content-range"] == f"bytes {start}-{end - 1}/{len(_WEIGHTS)}"
    assert served.headers["repr-digest"] == _repr_digest(_WEIGHTS)
    assert "content-digest" not in served.headers


@pytest.mark.anyio
async def test_conditions_and_several_ranges_are_answered_as_a_file_is(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    name = "final_model/weights/model.safetensors"
    etag = f'"sha256:{hashlib.sha256(_WEIGHTS).hexdigest()}"'
    cases = [
        ({"Range": "bytes=0-9", "If-Range": etag}, 206, _WEIGHTS[:10]),
        ({"Range": "bytes=0-9", "If-Range": _HTTP_DATE}, 206, _WEIGHTS[:10]),
        ({"Range": "bytes=0-9", "If-Range": '"stale"'}, 200, _WEIGHTS),
        ({"Range": "bytes=0-9,20-29"}, 200, _WEIGHTS),
        ({"Range": f"bytes={len(_WEIGHTS)}-"}, 416, b""),
        ({"Range": "bytes=abc"}, 400, None),
        ({"Range": "items=0-1"}, 400, None),
    ]
    for request, code, body in cases:
        served = await serve(await _download(runtime, name), request)
        assert served.status == code, request
        if body is not None:
            assert served.body == body, request
        if code == 416:
            assert served.headers["content-range"] == f"bytes */{len(_WEIGHTS)}"


@pytest.mark.anyio
async def test_the_archive_view_is_a_tar_gz_of_the_tree_under_its_digest(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    facts = published["final_model.tar.gz"].archive
    assert facts is not None

    served = await serve(await _download(runtime, "final_model.tar.gz"))

    assert served.status == 200
    assert served.headers["content-type"] == "application/gzip"
    assert len(served.body) == facts.size_bytes
    assert hashlib.sha256(served.body).hexdigest() == facts.sha256
    assert served.headers["repr-digest"] == _repr_digest(served.body)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(served.body))) as tar:
        members = {
            m.name: cast(Any, tar.extractfile(m)).read()
            for m in tar.getmembers()
            if m.isfile()
        }
    assert members == {f"final_model/{path}": data for path, data in _FILES.items()}


@pytest.mark.anyio
async def test_an_archive_range_is_the_slice_of_the_whole_archive(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    whole = (await serve(await _download(runtime, "final_model.tar.gz"))).body
    header = 10
    block = header + STORED_BLOCK_BYTES + 5
    spans = [
        (0, 12),
        (header + 5 + 500, header + 5 + 530),
        (header + 5 + 512 + len(_FILES["config.json"]) - 3, header + 5 + 1100),
        (block - 20, block + 20),
        (len(whole) - 12, len(whole)),
        (100, len(whole) - 100),
    ]
    for start, end in spans:
        served = await serve(
            await _download(runtime, "final_model.tar.gz"),
            {"Range": f"bytes={start}-{end - 1}"},
        )
        assert served.status == 206
        assert served.body == whole[start:end], (start, end)


def _corrupt(runtime: _Runtime, reference: ContentReference) -> None:
    path = runtime.store._objects.object_path(
        reference.authorization_scope, reference.content_digest
    )
    path.chmod(0o644)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


def _member(binding: ArtifactBinding, runtime: _Runtime, path: str) -> Any:
    manifest = runtime.result_reader.read_manifest(binding.reference)
    member = manifest.member(path)
    assert member is not None
    return member.reference(binding.reference.authorization_scope)


@pytest.mark.anyio
async def test_a_corrupt_member_ends_the_archive_before_its_last_bytes(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    facts = published["final_model.tar.gz"].archive
    assert facts is not None
    _corrupt(
        runtime,
        _member(published["final_model"], runtime, "weights/model.safetensors"),
    )
    sent: list[bytes] = []
    response = await _download(runtime, "final_model.tar.gz")
    with pytest.raises(ResultUnreadable):
        sent.append((await serve(response)).body)
    assert not sent


@pytest.mark.anyio
async def test_a_corrupt_file_answers_500_before_any_byte(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    _corrupt(runtime, published["notes.txt"].reference)

    served = await serve(await _download(runtime, "notes.txt"))

    assert served.status == 500


class _DownStore(SharedFilesystemObjectStore):
    def open(self, reference: ContentReference, **kwargs: Any) -> Any:
        raise ContentUnavailable("store down")


@pytest.mark.anyio
async def test_an_unreachable_store_answers_503_with_a_retry(
    tmp_path: Path, runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    runtime.result_reader = ResultReader(_DownStore(tmp_path / "cas"))

    served = await serve(await _download(runtime, "notes.txt"))

    assert served.status == 503
    assert served.headers["retry-after"] == "5"


@pytest.mark.anyio
async def test_archive_views_and_bundles_past_the_cap_are_refused_until_one_ends(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    held = [
        await _download(runtime, "final_model.tar.gz")
        for _ in range(results_router._STREAM_SLOTS)
    ]

    with pytest.raises(HTTPException) as raised:
        await _download(runtime, "final_model.tar.gz")
    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert raised.value.headers == {"Retry-After": "5"}
    with pytest.raises(HTTPException):
        await _bundle(runtime, [])
    assert (await serve(await _download(runtime, "notes.txt"))).status == 200

    with pytest.raises(ClientGone):
        await serve(held.pop(), abort_after_chunks=0)
    assert (await serve(await _download(runtime, "final_model.tar.gz"))).status == 200
    for response in held:
        await serve(response)
    assert results_router._STREAMS._held == 0


async def _bundle(
    runtime: _Runtime,
    include: list[str],
    results_dir: Path | None = None,
    revision: str = "final",
) -> Any:
    return await results_router.download_result_bundle(
        task_id="tsk-1",
        include=include,
        revision=revision,
        principal=cast(Any, None),
        runtime=cast(Any, runtime),
        results_dir=results_dir or Path("/nonexistent"),
        logger=_LOGGER,
    )


def _bundled(body: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(body))) as tar:
        return {
            m.name: cast(Any, tar.extractfile(m)).read()
            for m in tar.getmembers()
            if m.isfile()
        }


@pytest.mark.anyio
async def test_a_bundle_holds_the_result_the_revisions_files_and_the_logs(
    tmp_path: Path, runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    logs = tmp_path / "results" / "tsk-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_bytes(b'{"message": "hi"}\n')

    served = await serve(await _bundle(runtime, ["all"], tmp_path / "results"))

    assert served.headers["content-encoding"] == "gzip"
    assert served.headers["content-disposition"] == (
        'attachment; filename="tsk-1.tar.gz"'
    )
    assert _bundled(served.body) == {
        "tsk-1/results.json": runtime.results["tsk-1"],
        "tsk-1/artifacts/notes.txt": _notes(),
        **{f"tsk-1/artifacts/final_model/{p}": d for p, d in _FILES.items()},
        "tsk-1/logs/logs.jsonl": b'{"message": "hi"}\n',
    }
    step = await serve(await _bundle(runtime, ["artifacts"], revision="step-2"))
    assert _bundled(step.body) == {
        "tsk-1/artifacts/checkpoints/checkpoint-2/trainer_state.json": b'{"step": 2}'
    }


@pytest.mark.anyio
async def test_a_bundle_over_a_corrupt_member_is_refused_before_its_end(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corrupt(
        runtime,
        _member(published["final_model"], runtime, "weights/model.safetensors"),
    )

    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, ["artifacts"])
    assert raised.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    monkeypatch.setattr(artifact_reads, "_BUNDLE_CHUNK_BYTES", 64)
    response = await _bundle(runtime, ["artifacts"])
    with pytest.raises(ResultUnreadable):
        await serve(response)
    assert results_router._STREAMS._held == 0


@pytest.mark.anyio
async def test_a_bundle_whose_store_is_down_answers_503(
    tmp_path: Path, runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    runtime.result_reader = ResultReader(_DownStore(tmp_path / "cas"))

    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, ["artifacts"])

    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.anyio
async def test_a_bundle_is_written_off_the_event_loop(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: set[int] = set()
    opened = runtime.result_reader.open

    def recording(reference: ContentReference) -> Any:
        threads.add(threading.get_ident())
        return opened(reference)

    monkeypatch.setattr(runtime.result_reader, "open", recording)

    await serve(await _bundle(runtime, ["artifacts"]))

    assert threads and threading.get_ident() not in threads


@pytest.mark.anyio
async def test_an_archive_streams_in_bounded_memory(
    tmp_path: Path, runtime: _Runtime
) -> None:
    big = tmp_path / "big" / "artifacts"
    (big / "model").mkdir(parents=True)
    (big / "model" / "weights.bin").write_bytes(b"\x01" * (48 << 20))
    publisher = ArtifactPublisher(runtime.store, "org")
    bindings = publisher.publish_final(big, {"model.tar.gz": "model"})
    runtime.bindings["tsk-1"] = {b.key: b for b in bindings}

    tracemalloc.start()
    try:
        response = await _download(runtime, "model.tar.gz")
        sha = hashlib.sha256()
        received = 0

        async def send(message: Any) -> None:
            nonlocal received
            if message["type"] == "http.response.body":
                sha.update(message["body"])
                received += len(message["body"])

        async def receive() -> Any:
            return {"type": "http.request"}

        await response({"type": "http", "method": "GET", "headers": []}, receive, send)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert received > 48 << 20
    assert peak < 16 << 20
