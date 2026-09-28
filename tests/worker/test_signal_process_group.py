"""A process-group signal never reaches beyond a child's own group."""

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from worker.utils.process import signal_process_group


@pytest.mark.parametrize(
    "pgid",
    [0, 1, os.getpgrp(), MagicMock().pid],
    ids=["zero", "init", "own", "mocked"],
)
def test_a_group_beyond_the_child_is_never_signalled(pgid: int) -> None:
    with patch("os.killpg") as killpg:
        signal_process_group(pgid, signal.SIGTERM)

    killpg.assert_not_called()


def test_a_childs_own_group_is_signalled() -> None:
    pgid = os.getpgrp() + 1
    with patch("os.killpg") as killpg:
        signal_process_group(pgid, signal.SIGTERM)

    killpg.assert_called_once_with(pgid, signal.SIGTERM)
