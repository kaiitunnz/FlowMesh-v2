"""The codex app-server launches from a secret-free environment allowlist."""

import os

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

from worker.executors.harness.codex_transport import (  # noqa: E402
    clean_launch_environ,
    sanitized_launch_env,
)

_SECRETS = (
    "AGENT_MODEL_API_KEY",
    "WEB_SEARCH_API_KEY",
    "FLOWMESH_API_KEY",
    "OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "FLOWMESH_SENTINEL_SECRET",
)


def test_sanitized_launch_env_keeps_runtime_drops_secrets() -> None:
    source = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/worker",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "XDG_CACHE_HOME": "/scratch/cache",
        "TMPDIR": "/scratch/tmp",
        **{name: "leaked" for name in _SECRETS},
    }
    cleaned = sanitized_launch_env(source)
    assert cleaned["PATH"] == "/usr/bin:/bin"
    assert cleaned["HOME"] == "/home/worker"
    assert cleaned["LC_ALL"] == "C.UTF-8"
    assert cleaned["XDG_CACHE_HOME"] == "/scratch/cache"
    for name in _SECRETS:
        assert name not in cleaned


def test_clean_launch_environ_hides_secrets_during_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _SECRETS:
        monkeypatch.setenv(name, "leaked")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    with clean_launch_environ():
        # The app-server captures os.environ at this instant, so no secret is present.
        for name in _SECRETS:
            assert name not in os.environ
        assert "PATH" in os.environ

    # The launcher's full environment is restored after the spawn window.
    for name in _SECRETS:
        assert os.environ[name] == "leaked"
