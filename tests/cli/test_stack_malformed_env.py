"""A stack command refuses an env file Docker Compose would refuse, with its error."""

from pathlib import Path

import pytest
from flowmesh_cli.cli import build_cli_app
from typer.testing import CliRunner


@pytest.fixture
def malformed(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text('REDIS_PASSWORD="abc\n')
    return env_file


def test_a_stack_command_reports_a_malformed_env_file_without_a_traceback(
    malformed: Path,
) -> None:
    result = CliRunner().invoke(
        build_cli_app(),
        ["stack", "worker", "pull", "cpu", "--env-file", str(malformed)],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "unterminated quoted value" in result.output
