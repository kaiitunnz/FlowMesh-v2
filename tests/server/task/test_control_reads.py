"""Reading the stored value a branch routes or a spawn fans out over."""

from typing import Any, cast

from server.orchestration.state import ValueRef
from server.task.results import ResultUnavailable
from server.task.runtime.control_reads import read_control_value
from shared.content import ContentReference
from shared.tasks.result_binding import ResultBinding


class _AwayStore:
    def __init__(self) -> None:
        self.reads = 0

    def read(self, binding: ResultBinding) -> Any:
        self.reads += 1
        raise ResultUnavailable("store away")


def test_a_read_the_store_cannot_answer_is_left_to_a_later_redrive() -> None:
    store = _AwayStore()
    reference = ContentReference(
        authorization_scope="local", content_digest="d" * 64, size_bytes=1
    )
    read = read_control_value(
        cast(Any, store),
        ValueRef(kind="legacy_task_result", legacy_task_id="tsk-1"),
        ResultBinding(task_id="tsk-1", reference=reference),
    )
    assert read.unavailable
    assert store.reads == 1
