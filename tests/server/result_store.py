"""Test helpers for task results kept in the shared content store."""

import tempfile
from pathlib import Path
from typing import Any

from server.task.results import ResultReader
from shared.content import ContentReference, SharedFilesystemObjectStore
from shared.schemas.result import RESULT_MEDIA_TYPE, BaseExecutorResult, ResultEnvelope


class StoreResultReader(ResultReader):
    """A result reader that keeps its store at hand for a test to write into."""

    def __init__(self, store: SharedFilesystemObjectStore) -> None:
        super().__init__(store)
        self.store = store


def make_result_reader(
    store: SharedFilesystemObjectStore | None = None,
) -> StoreResultReader:
    """A result reader over a local shared store, a fresh one by default."""
    return StoreResultReader(
        store or SharedFilesystemObjectStore(Path(tempfile.mkdtemp()) / "content")
    )


def store_result(
    reader: ResultReader,
    task_id: str,
    result: BaseExecutorResult | dict[str, Any],
    scope: str = "",
) -> ContentReference:
    """Store a task's result envelope as its worker would, returning its reference."""
    assert isinstance(reader, StoreResultReader)
    envelope = ResultEnvelope.model_validate({"task_id": task_id, "result": result})
    return reader.store.write(
        scope,
        envelope.model_dump_json(indent=2).encode("utf-8"),
        media_type=RESULT_MEDIA_TYPE,
    )


def result_payload(
    reader: ResultReader,
    task_id: str,
    result: BaseExecutorResult | dict[str, Any],
    scope: str = "",
) -> dict[str, Any]:
    """The success metadata a worker reports for a stored result."""
    reference = store_result(reader, task_id, result, scope)
    return {"result_reference": reference.model_dump(mode="json")}
