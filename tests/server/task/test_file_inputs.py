"""A file a task published reaches its consumer as that published artifact on every
path its reference travels: a rendered spec names it by a token its worker hydrates, a
worker resolving the ref itself hydrates it from its producer's publications, and an
agent reads its root download path."""

import io
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from server.task.v2 import CompileError
from shared.content import TREE_MEDIA_TYPE, reference_for
from shared.tasks.artifact_binding import (
    ArchiveFacts,
    ArtifactBinding,
    ArtifactKind,
)
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.task.test_agent_region_inputs import _AGENT, _member_values, _ready
from tests.server.task.test_runtime_control_flow import _Run, _workflow
from tests.server.task.test_scoped_inputs import _dispatch
from tests.worker.content.test_artifact_inputs import _CacheLane
from tests.worker.content.test_plane_routing import _Access
from worker.content import WorkerContentPlane
from worker.content.inputs import TaskInputHydrator
from worker.executors.base_executor import ExecutionError
from worker.executors.mixins import data as data_module
from worker.executors.mixins.data import DataMixin
from worker.executors.utils import artifacts as artifacts_module
from worker.executors.utils.artifacts import artifact_ref_source
from worker.executors.utils.expressions import item_path, project_expression_paths

_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _tree(org: str, producer: str) -> ArtifactBinding:
    return ArtifactBinding(
        name="final_lora",
        kind=ArtifactKind.TREE,
        reference=reference_for(org, producer.encode(), media_type=TREE_MEDIA_TYPE),
        size_bytes=4,
    )


def _train(run: _Run, name: str, **extra: Any) -> str:
    """Settle the ready ``name`` as a trainer that published its adapter tree and the
    archive view over it, returning its task id."""
    producer = next(t for t in run.ready if run.name(t) == name)
    record = run.runtime.get_record(producer)
    assert record is not None
    tree = _tree(record.org_id, producer)
    archive = ArtifactBinding(
        name="final_lora.tar.gz",
        kind=ArtifactKind.ARCHIVE,
        reference=tree.reference,
        size_bytes=30,
        of=tree.name,
        archive=ArchiveFacts(sha256="ab" * 32, crc32=1, size_bytes=30, top=tree.name),
    )
    run.run(
        name,
        {
            "final_lora_archive": {"path": "final_lora.tar.gz"},
            "_artifacts": {"task_id": producer},
            **extra,
        },
        [tree, archive],
    )
    return producer


def _token(n: int) -> str:
    return f"flowmesh-artifact:dsp-1.{n}"


def _tokens(message: Any) -> list[ArtifactBinding]:
    return list((message.artifact_inputs or {}).values())


def _reads_tree_of(run: _Run, consumer: str, producer: str) -> bool:
    """Whether the ready ``consumer`` renders one item naming ``producer``'s tree."""
    spec, message = _dispatch(run, consumer)
    return spec["data"]["items"] == [_token(0)] and [
        b.reference for b in _tokens(message)
    ] == [_tree(message.content_scope, producer).reference]


@pytest.mark.anyio
async def test_a_root_input_projecting_a_file_names_its_published_tree() -> None:
    run = await _Run().start(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: use
        dependsOn: [{{node: train, input: archive, project: [final_lora_archive]}}]
        spec: {{taskType: echo, data: {{type: list, items: ["${{archive}}"]}}}}
"""))
    train = _train(run, "train")

    spec, message = _dispatch(run, "use")

    assert spec["data"]["items"] == [_token(0)]
    assert [b.reference for b in _tokens(message)] == [
        _tree(message.content_scope, train).reference
    ]


def test_a_root_whole_read_of_an_unnamed_stage_stays_refused() -> None:
    with pytest.raises(CompileError, match="train"):
        _Run().runtime.validate(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: use
        dependsOn: [train]
        spec: {{taskType: echo, data: {{type: list, items: ["${{train}}"]}}}}
"""))


_CARRY_BODY = """
    templates:
      - name: body
        inputs: [{name: state, role: carried}]
        nodes:
          - name: step
            dependsOn: [{node: $ingress, port: state, input: state}]
            spec: {taskType: echo, data: {type: list, items: ["${state}"]}}
          - name: route
            dependsOn: [{node: step, input: input}]
            region:
              kind: branch
              inputs: [{name: input}]
              outputs: [{name: again}, {name: done}]
              selection: {input: input, field: [route]}
        edges:
          - from: {node: route, port: again}
            to: {node: $feedback, port: state}
            project: [final_lora_archive]
          - from: {node: route, port: done}
            to: {node: $egress, port: state}
"""

_CARRY = f"""
      - name: seed
        spec: {_ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          carried: [{{name: state}}]
"""


@pytest.mark.anyio
async def test_a_fed_back_file_names_its_producers_tree() -> None:
    run = await _Run().start(_workflow(_CARRY, _CARRY_BODY))
    run.run("seed", {"v": 0})
    step = _train(run, "step", route="again")

    spec, message = _dispatch(run, "step")

    assert spec["data"]["items"] == [_token(0)]
    assert [b.reference for b in _tokens(message)] == [
        _tree(message.content_scope, step).reference
    ]


_CONCAT = f"""
      - name: left
        spec: {_ECHO}
      - name: right
        spec: {_ECHO}
      - name: both
        dependsOn: [{{node: left, input: l}}, {{node: right, input: r}}]
        region: {{kind: merge, combination: concat}}
      - name: use
        dependsOn: [{{node: both, input: all}}]
        spec:
          taskType: echo
          data:
            type: list
            items:
              - "${{all.0.value.final_lora_archive}}"
              - "${{all.1.value.final_lora_archive}}"
"""


@pytest.mark.anyio
async def test_each_aggregate_members_file_names_its_own_producers_tree() -> None:
    run = await _Run().start(_workflow(_CONCAT))
    left = _train(run, "left")
    right = _train(run, "right")

    spec, message = _dispatch(run, "use")

    assert spec["data"]["items"] == [_token(0), _token(1)]
    assert [b.reference for b in _tokens(message)] == [
        _tree(message.content_scope, left).reference,
        _tree(message.content_scope, right).reference,
    ]
    upstream = message.task.spec.upstreamResults or {}
    items, paths = project_expression_paths("all.value.final_lora_archive", upstream)
    assert [
        artifact_ref_source(item, upstream, "all", item_path(paths, i))
        for i, item in enumerate(items)
    ] == [(left, "final_lora.tar.gz"), (right, "final_lora.tar.gz")]
    assert DataMixin()._extract_source_data_ids(message.task.spec) == [left, right]


@pytest.mark.anyio
async def test_rendering_an_aggregate_reads_each_member_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(f"""
      - name: plan
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: use
        dependsOn: [{{node: collect, input: all}}]
        spec:
          taskType: echo
          data:
            type: list
            items: ["${{all.0.value.v}}", "${{all.1.value.v}}", "${{all.2.value.v}}"]
"""))
    run.run("plan", {"items": [str(i) for i in range(4)]})
    for i in range(4):
        run.run("kid", {"v": i})
    reads: list[Any] = []
    read = run.runtime.read_binding

    def counting(binding: Any) -> Any:
        reads.append(binding.task_id)
        return read(binding)

    monkeypatch.setattr(run.runtime, "read_binding", counting)

    spec, _ = _dispatch(run, "use")

    assert spec["data"]["items"] == [0, 1, 2]
    assert sorted(reads) == sorted(set(reads)) and len(reads) == 4


@pytest.mark.anyio
async def test_an_agent_reading_a_projected_file_reads_its_download_path() -> None:
    run = await _Run().start(_workflow(f"""
      - name: train
        spec: {_ECHO}
      - name: reader
        dependsOn: [{{node: train, input: x, project: [final_lora_archive]}}]
        spec:{_AGENT}
"""))
    train = _train(run, "train")

    assert _member_values(run, _ready(run, "reader")) == {
        "x": [f"/api/v1/results/{train}/files/final_lora.tar.gz"]
    }


def _png(color: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (1, 1), color).save(out, format="PNG")
    return out.getvalue()


_IMAGES = {
    "left": {"a.png": "red", "b.png": "green"},
    "right": {"c.png": "blue", "d.png": "white"},
}


def _publish_images(run: _Run, name: str, extra: tuple[str, ...] = ()) -> str:
    """Settle the ready ``name`` as a task whose result names its images, each one it
    published into the shared store, returning its task id."""
    producer = next(t for t in run.ready if run.name(t) == name)
    record = run.runtime.get_record(producer)
    assert record is not None
    published = [
        ArtifactBinding(
            name=image,
            kind=ArtifactKind.FILE,
            reference=(reference := run.reader.store.write(record.org_id, _png(color))),
            size_bytes=reference.size_bytes,
        )
        for image, color in _IMAGES[name].items()
    ]
    named = [*_IMAGES[name], *extra]
    run.run(
        name,
        {"images": [{"path": p} for p in named], "_artifacts": {"task_id": producer}},
        published,
    )
    return producer


async def _image_consumer(
    expr: str, extra: tuple[str, ...] = ()
) -> tuple[WorkerTaskMessage, dict[str, str], Any]:
    run = await _Run().start(_workflow(f"""
      - name: left
        spec: {_ECHO}
      - name: right
        spec: {_ECHO}
      - name: both
        dependsOn: [{{node: left, input: l}}, {{node: right, input: r}}]
        region: {{kind: merge, combination: concat}}
      - name: use
        dependsOn: [{{node: both, input: all}}]
        spec: {_ECHO}
"""))
    ids = {
        "left": _publish_images(run, "left", extra),
        "right": _publish_images(run, "right"),
    }
    _, message = _dispatch(run, "use")
    spec = InferenceSpecStrict.model_validate(
        {
            "taskType": "inference",
            "model": {
                "source": {"identifier": "vlm"},
                "transformers": {"mode": "visual-embedding"},
            },
            "data": {"type": "list", "expr": expr},
            "_upstreamResults": message.task.spec.upstreamResults,
        }
    )
    message.task = message.task.model_copy(update={"spec": spec})
    return message, ids, run.reader.store


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("all.value[0].images", ["left/a.png", "left/b.png"]),
        (
            "all.value.images",
            ["left/a.png", "left/b.png", "right/c.png", "right/d.png"],
        ),
    ],
)
async def test_images_over_an_aggregate_hydrate_from_their_own_producer_not_the_root(
    expr: str,
    expected: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message, ids, store = await _image_consumer(expr)

    def no_root(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a worker reads no artifact over HTTP")

    monkeypatch.setattr(artifacts_module.requests, "get", no_root)
    monkeypatch.setattr(data_module.requests, "get", no_root)
    lane = _CacheLane(tmp_path / "cache")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _Access(store)))

    TaskInputHydrator(cast(Any, plane)).install_upstream_artifacts(
        message, tmp_path / "in"
    )
    entry = DataMixin()._collect_prompts_for_spec(
        cast(InferenceSpecStrict, message.task.spec), message.task_id, True
    )

    assert message.upstream_artifacts is not None
    assert set(message.upstream_artifacts) == set(ids.values())
    assert [image.getpixel((0, 0)) for image in entry.images if image] == [
        Image.new("RGB", (1, 1), _IMAGES[name][image]).getpixel((0, 0))
        for name, image in (item.split("/") for item in expected)
    ]
    assert len(lane.peer_reads) == len(expected)
    assert sorted(ref.content_digest for ref in lane.kept) == sorted(
        reference_for(message.content_scope, _png(_IMAGES[name][image])).content_digest
        for name, image in (item.split("/") for item in expected)
    )


@pytest.mark.anyio
async def test_an_image_its_producer_never_published_fails_the_task_unreadable(
    tmp_path: Path,
) -> None:
    message, ids, store = await _image_consumer(
        "all.value[0].images", extra=("missing.png",)
    )

    with pytest.raises(ExecutionError, match="input_unreadable") as raised:
        TaskInputHydrator(
            cast(Any, WorkerContentPlane(None, cast(Any, _Access(store))))
        ).install_upstream_artifacts(message, tmp_path / "in")

    assert raised.value.retryable is False
    assert f"task {ids['left']} published no artifact named 'missing.png'" in str(
        raised.value
    )
