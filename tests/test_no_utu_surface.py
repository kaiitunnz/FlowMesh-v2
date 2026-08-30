"""Guard: no legacy UTU agent-execution surface survives in product code.

Every agent runs the resolved-harness episode path, so the UTU executor, its
``UTU_LLM_*`` configuration, and the ``runtime-agent`` dependency group must not
reappear in shipped source, config, docs, examples, or tests. The needles are
assembled from fragments so this guard does not match itself.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCAN_DIRS = ("src", "cli", "sdk", "hook", "docs", "examples", "tests")
_SCAN_FILES = ("pyproject.toml", "AGENTS.md", "CONTRIBUTING.md")
_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".example", ".txt"}
_SELF = Path(__file__).name

# Contiguous tokens that must not appear, each built from fragments.
_NEEDLES = (
    "Agent" + "Executor",
    "agent" + "_executor",
    "UTU" + "_LLM",
    "runtime" + "-agent",
    "you" + "tu-agent",
)


def _scan_paths() -> list[Path]:
    paths: list[Path] = []
    for name in _SCAN_FILES:
        if (path := _ROOT / name).is_file():
            paths.append(path)
    for directory in _SCAN_DIRS:
        for path in (_ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in _SUFFIXES and path.name != _SELF:
                paths.append(path)
    return paths


@pytest.mark.parametrize("needle", _NEEDLES)
def test_no_utu_surface_remains(needle: str) -> None:
    offenders = [
        path.relative_to(_ROOT).as_posix()
        for path in _scan_paths()
        if needle in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, f"{needle!r} still present in: {offenders}"
