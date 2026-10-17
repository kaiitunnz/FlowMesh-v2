"""The tree finisher acts on how each tree's expansion ended: a later root's fence stops
it, a staged tree whose members were not all held is tried again, and a written-through
one is left to reconciliation."""

import logging
from typing import Any, cast
from unittest.mock import Mock

import pytest

from server.reclamation.publications import Holds
from server.reclamation.trees import TreeOutcome, TreeStatus
from server.services.publications import TreeFinisher
from shared.content import ContentReference

_MANIFEST = ContentReference(
    authorization_scope="org", content_digest="a" * 64, size_bytes=10
)


class _Holds:
    def __init__(self, status: TreeStatus) -> None:
        self.status = status

    def finish(self, holds: Holds, bound: frozenset[str], locator: Any) -> list[Any]:
        return [(_MANIFEST, TreeOutcome(self.status))]


def _finisher(status: TreeStatus) -> tuple[TreeFinisher, Mock]:
    runtime = Mock()
    finisher = TreeFinisher(
        cast(Any, runtime),
        cast(Any, _Holds(status)),
        logging.getLogger("test.finisher"),
        clock=lambda: 0.0,
    )
    return finisher, runtime


def _submit(finisher: TreeFinisher, staged: bool) -> None:
    holds = Holds("1:1", {}, staged=[_MANIFEST] if staged else [])
    finisher.submit(
        "tsk-1", holds, {"tsk-1": frozenset({_MANIFEST.content_digest})}, {}
    )


def _run_once(finisher: TreeFinisher) -> None:
    _, _, job = finisher._queue.pop()
    finisher._finish(job)


@pytest.mark.parametrize("staged", [True, False])
def test_a_fenced_expansion_is_left_to_the_later_root(staged: bool) -> None:
    finisher, runtime = _finisher(TreeStatus.FENCED)
    _submit(finisher, staged)

    _run_once(finisher)

    assert finisher._queue == []
    runtime.tree_expanded.assert_not_called()


def test_a_staged_tree_whose_members_were_not_all_held_is_tried_again() -> None:
    finisher, runtime = _finisher(TreeStatus.INCOMPLETE)
    _submit(finisher, staged=True)

    _run_once(finisher)

    assert len(finisher._queue) == 1
    runtime.tree_expanded.assert_not_called()


def test_a_written_through_tree_whose_members_were_not_all_held_is_left_unexpanded() -> (
    None
):
    finisher, runtime = _finisher(TreeStatus.INCOMPLETE)
    _submit(finisher, staged=False)

    _run_once(finisher)

    assert finisher._queue == []
    runtime.tree_expanded.assert_not_called()
