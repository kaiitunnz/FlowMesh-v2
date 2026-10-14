"""The result reader's cache bounds the memory it holds, however small each result."""

import gc
import json
import tracemalloc
from pathlib import Path

from server.task.results import ResultReader
from shared.content import (
    TREE_MEDIA_TYPE,
    ContentReference,
    SharedFilesystemObjectStore,
    TreeManifest,
    TreeMember,
)
from shared.tasks.result_binding import ResultBinding
from shared.utils.ids import new_task_id

_BUDGET = 64 * 1024


def _stored(store: SharedFilesystemObjectStore, count: int) -> list[tuple[str, str]]:
    """Tiny stored results, each named by its task and its reference's JSON."""
    stored = []
    for _ in range(count):
        task_id = new_task_id()
        data = json.dumps({"task_id": task_id, "result": {}}).encode()
        reference = store.write("org", data, media_type="application/json")
        stored.append((task_id, reference.model_dump_json()))
    return stored


def test_many_tiny_results_keep_the_cache_within_its_budget(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    reader = ResultReader(store, cache_max_bytes=_BUDGET)
    stored = _stored(store, 2000)
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for task_id, raw in stored:
            reference = ContentReference.model_validate_json(raw)
            reader.read_bytes(ResultBinding(task_id=task_id, reference=reference))
        gc.collect()
        held = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()

    assert 0 < len(reader._cache) < len(stored)
    # The process allocates beside the cache while it reads; uncounted, the overhead
    # held twenty times the budget.
    assert held <= 2 * _BUDGET


def test_a_cached_result_is_read_from_memory(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    reader = ResultReader(store, cache_max_bytes=_BUDGET)
    ((task_id, raw),) = _stored(store, 1)
    binding = ResultBinding(
        task_id=task_id, reference=ContentReference.model_validate_json(raw)
    )
    first = reader.read_bytes(binding)
    for path in (tmp_path / "cas").rglob("*"):
        if path.is_file():
            path.unlink()

    assert reader.read_bytes(binding) == first


def test_a_trees_members_are_decoded_once(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    reader = ResultReader(store, cache_max_bytes=_BUDGET)
    member = store.write("org", b"weights")
    manifest = TreeManifest.of(
        [TreeMember("w.bin", member.content_digest, member.size_bytes)]
    )
    tree = store.write("org", manifest.encode(), media_type=TREE_MEDIA_TYPE)

    first = reader.tree_members(tree)
    reader._cache.clear()
    for path in (tmp_path / "cas").rglob("*"):
        if path.is_file():
            path.unlink()

    assert reader.tree_members(tree) is first
    assert first == {member.identity}
