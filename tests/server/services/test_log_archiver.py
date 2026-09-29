import logging
import stat
from pathlib import Path
from unittest.mock import MagicMock

from server.services.log_archiver import TaskLogArchiver


def test_the_logs_directory_is_shared_before_the_archiver_writes(
    tmp_path: Path,
) -> None:
    archiver = TaskLogArchiver(
        redis=MagicMock(),
        runtime=MagicMock(),
        results_dir=tmp_path,
        logger=logging.getLogger("test.log_archiver"),
    )

    logs_path = archiver._logs_path("tsk-1")

    assert not logs_path.exists()
    for directory in (logs_path.parent.parent, logs_path.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o777
