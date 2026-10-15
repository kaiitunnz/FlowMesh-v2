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

import anyio
import pytest
from fastapi import FastAPI, HTTPException, status

from server.routers.v1 import results as results_router
from server.task import artifact_reads
from server.task.artifact_durability import (
    ArtifactLoss,
    ArtifactUpload,
    UploadStateUnavailable,
)
from server.task.results import ResultReader, ResultUnavailable, ResultUnreadable
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
        self.uploads: dict[str, ArtifactUpload] = {}
        self.upload_error: Exception | None = None

    def artifact_bindings(
        self, task_id: str, revision: str | None = None
    ) -> dict[tuple[str, str], ArtifactBinding] | None:
        if (bindings := self.bindings.get(task_id)) is None:
            return None
        return {
            key: binding
            for key, binding in bindings.items()
            if revision is None or key[0] == revision
        }

    def artifact_upload_of(self, task_id: str) -> ArtifactUpload | None:
        if self.upload_error is not None:
            raise self.upload_error
        return self.uploads.get(task_id)

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
async def test_a_corrupt_member_in_the_first_window_answers_500(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    _corrupt(
        runtime,
        _member(published["final_model"], runtime, "weights/model.safetensors"),
    )

    served = await serve(await _download(runtime, "final_model.tar.gz"))

    assert served.status == 500


@pytest.mark.anyio
async def test_an_archive_whose_members_are_unreachable_answers_503(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await _download(runtime, "final_model.tar.gz")

    def down(*_args: Any, **_kwargs: Any) -> Any:
        raise ContentUnavailable("store down")

    monkeypatch.setattr(runtime.store, "open", down)
    served = await serve(response)

    assert served.status == 503
    assert served.headers["retry-after"]


@pytest.mark.anyio
async def test_a_range_spanning_the_whole_object_is_verified(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    weights = _member(published["final_model"], runtime, "weights/model.safetensors")
    _corrupt(runtime, weights)

    served = await serve(
        await _download(runtime, "final_model/weights/model.safetensors"),
        {"Range": "bytes=0-"},
    )

    assert served.status == 500


def test_the_bundle_reader_fills_what_it_is_asked_for() -> None:
    reader = artifact_reads._ChunkReader(iter([b"ab", b"", b"cd", b"e"]))
    buffer = bytearray(4)

    assert reader.readinto(buffer) == 4 and buffer == b"abcd"
    assert reader.readinto(buffer) == 1 and buffer[:1] == b"e"
    assert reader.readinto(buffer) == 0


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
async def test_bundles_past_the_cap_are_refused_until_one_ends(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    held = [await _bundle(runtime, []) for _ in range(results_router._STREAM_SLOTS)]

    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, [])
    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert raised.value.headers == {"Retry-After": "5"}
    assert (await serve(await _download(runtime, "final_model.tar.gz"))).status == 200
    assert (await serve(await _download(runtime, "notes.txt"))).status == 200

    with pytest.raises(ClientGone):
        await serve(held.pop(), abort_after_chunks=0)
    assert (await serve(await _bundle(runtime, []))).status == 200
    for response in held:
        await serve(response)
    assert results_router._STREAMS._held == 0


@pytest.mark.anyio
async def test_a_bundle_refused_at_the_cap_reads_nothing(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = [await _bundle(runtime, []) for _ in range(results_router._STREAM_SLOTS)]
    built: list[str] = []

    def building(files: Any, task_id: str) -> Any:
        built.append(task_id)
        raise AssertionError("a refused bundle started its stream")

    monkeypatch.setattr(results_router, "BundleStream", building)

    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, [])
    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert built == []
    for response in held:
        await serve(response)
    assert results_router._STREAMS._held == 0


@pytest.mark.anyio
async def test_a_bundle_that_cannot_start_gives_its_slot_back(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(files: Any, task_id: str) -> Any:
        raise RuntimeError("no thread")

    monkeypatch.setattr(results_router, "BundleStream", broken)

    with pytest.raises(RuntimeError):
        await _bundle(runtime, [])
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
async def test_a_bundle_at_a_revision_never_published_is_not_found(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, ["artifacts"], revision="step-9")
    assert raised.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.anyio
async def test_the_result_file_answers_503_while_the_store_is_unreachable(
    runtime: _Runtime,
    published: dict[str, ArtifactBinding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def down(_task_id: str) -> bytes:
        raise ResultUnavailable("store down")

    monkeypatch.setattr(runtime, "read_result_bytes", down)

    with pytest.raises(HTTPException) as raised:
        await _download(runtime, "results.json")
    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert raised.value.headers == {"Retry-After": str(results_router._RETRY_AFTER_SEC)}


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
            await anyio.sleep_forever()

        await response({"type": "http", "method": "GET", "headers": []}, receive, send)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert received > 48 << 20
    assert peak < 16 << 20


@pytest.mark.anyio
async def test_an_archive_stops_when_its_client_goes_away(
    tmp_path: Path, runtime: _Runtime
) -> None:
    big = tmp_path / "big" / "artifacts"
    (big / "model").mkdir(parents=True)
    (big / "model" / "weights.bin").write_bytes(b"\x01" * (48 << 20))
    publisher = ArtifactPublisher(runtime.store, "org")
    bindings = publisher.publish_final(big, {"model.tar.gz": "model"})
    runtime.bindings["tsk-1"] = {b.key: b for b in bindings}

    served = await serve(
        await _download(runtime, "model.tar.gz"),
        abort_after_chunks=1,
        silent_disconnect=True,
    )

    assert served.sends_after_disconnect <= 2
    assert results_router._STREAMS._held == 0


def test_the_download_routes_declare_their_binary_responses() -> None:
    app = FastAPI()
    app.include_router(results_router.router)
    paths = app.openapi()["paths"]
    files = paths["/results/{task_id}/files/{filename}"]["get"]
    bundle = paths["/results/{task_id}/bundle"]["get"]

    binary = {"type": "string", "format": "binary"}
    assert files["responses"]["200"]["content"] == {
        "application/octet-stream": {"schema": binary}
    }
    assert set(files["responses"]) >= {"200", "206", "416", "503"}
    assert "Repr-Digest" in files["responses"]["200"]["headers"]
    assert "Content-Range" in files["responses"]["206"]["headers"]
    assert "Retry-After" in files["responses"]["503"]["headers"]
    listing = paths["/results/{task_id}/artifacts"]["get"]
    assert "Retry-After" in listing["responses"]["503"]["headers"]
    assert bundle["responses"]["200"]["content"] == {
        "application/gzip": {"schema": binary}
    }
    for operation in (files, bundle):
        (revision,) = (p for p in operation["parameters"] if p["name"] == "revision")
        assert revision["schema"]["pattern"] == "^(final|step-(0|[1-9][0-9]*))$"


def _owed(runtime: _Runtime, lost: dict[str, ArtifactLoss] | None = None) -> None:
    upload = ArtifactUpload(
        dispatch_id="dsp-1", dispatch_task_id="tsk-1", owner_id="wkr-1"
    )
    runtime.uploads["tsk-1"] = upload if lost is None else upload.settle(lost, 1.0)


async def _listed(runtime: _Runtime) -> dict[str, Any]:
    page = await results_router.list_result_artifacts(
        task_id="tsk-1",
        revision=None,
        principal=cast(Any, None),
        runtime=cast(Any, runtime),
        logger=_LOGGER,
    )
    return {
        f"{a.revision}/{a.name}": (a.durability.value, a.durability_error)
        for a in page.artifacts
    }


@pytest.mark.anyio
async def test_the_listing_says_where_each_artifact_stands(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    assert {d for d, _ in (await _listed(runtime)).values()} == {"durable"}

    _owed(runtime)
    pending = await _listed(runtime)
    assert pending["step-2/checkpoints"] == ("durable", None)
    assert {pending[k][0] for k in pending if k.startswith("final/")} == {"pending"}

    loss = ArtifactLoss(message="final_model is gone")
    _owed(runtime, {"final\0final_model": loss})
    lost = await _listed(runtime)
    assert lost["final/final_model"][0] == "lost"
    assert lost["final/final_model.tar.gz"][0] == "lost"
    assert lost["final/final_model"][1].code == "artifact_lost"
    assert lost["final/notes.txt"] == ("durable", None)


@pytest.mark.anyio
async def test_a_pending_artifact_answers_503_and_a_lost_one_410(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    _owed(runtime)
    for name in ("final_model/config.json", "final_model.tar.gz", "notes.txt"):
        with pytest.raises(HTTPException) as raised:
            await _download(runtime, name)
        assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert _code(raised.value) == "artifact_pending"
        assert raised.value.headers and "Retry-After" in raised.value.headers
    step = await serve(
        await _download(
            runtime, "checkpoints/checkpoint-2/trainer_state.json", "step-2"
        )
    )
    assert step.status == 200

    _owed(runtime, {"final\0final_model": ArtifactLoss(message="gone")})
    for name in ("final_model/config.json", "final_model.tar.gz"):
        with pytest.raises(HTTPException) as raised:
            await _download(runtime, name)
        assert raised.value.status_code == status.HTTP_410_GONE
        assert _code(raised.value) == "artifact_lost"
    assert (await serve(await _download(runtime, "notes.txt"))).status == 200


@pytest.mark.anyio
async def test_an_unknown_path_under_a_pending_or_lost_artifact_is_not_found(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    for lost in (None, {"final\0final_model": ArtifactLoss(message="gone")}):
        _owed(runtime, lost)
        for name in ("final_model/absent.json", "notes.txt/inner"):
            with pytest.raises(HTTPException) as raised:
                await _download(runtime, name)
            assert raised.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.anyio
async def test_an_unreadable_upload_state_answers_503_never_durable(
    runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    runtime.upload_error = UploadStateUnavailable("control store down")

    for request in (
        _listed(runtime),
        _download(runtime, "notes.txt"),
        _bundle(runtime, ["artifacts"]),
    ):
        with pytest.raises(HTTPException) as raised:
            await request
        assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "control store" in str(raised.value.detail)
    step = await _download(
        runtime, "checkpoints/checkpoint-2/trainer_state.json", "step-2"
    )
    assert (await serve(step)).status == 200


@pytest.mark.anyio
async def test_a_bundle_refuses_pending_or_lost_artifacts_before_any_byte(
    tmp_path: Path, runtime: _Runtime, published: dict[str, ArtifactBinding]
) -> None:
    _owed(runtime)
    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, ["artifacts"])
    assert raised.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    results_only = await serve(await _bundle(runtime, ["results"]))
    assert set(_bundled(results_only.body)) == {"tsk-1/results.json"}
    assert (
        await serve(await _bundle(runtime, ["artifacts"], revision="step-2"))
    ).status == 200

    _owed(runtime, {"final\0final_model": ArtifactLoss(message="gone")})
    with pytest.raises(HTTPException) as raised:
        await _bundle(runtime, ["artifacts"])
    assert raised.value.status_code == status.HTTP_410_GONE


def _code(error: HTTPException) -> str:
    detail: Any = error.detail
    assert isinstance(detail, dict)
    return detail["code"]
