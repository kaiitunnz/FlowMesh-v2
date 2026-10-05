import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh_cli_stack import stack as stack_module
from flowmesh_stack.env import load_env


def test_a_stack_reloaded_in_one_process_keeps_its_plugin_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FLOWMESH_PLUGIN_DATA_DIR=my_plugin_volume\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(load_env, "_loaded", None, raising=False)
    with patch.dict(os.environ, {}, clear=True):
        stack = stack_module._stack()
        for _ in range(2):
            stack.load_env(env_file)
            assert os.environ["FLOWMESH_PLUGIN_DATA_VOLUME"] == "my_plugin_volume"
            assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == "flowmesh_plugin_data"
