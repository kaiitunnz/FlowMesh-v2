"""What an agent may publish from its workspace when it completes.

An agent declares each path of its workspace it exports, and the name the export
publishes under. A path that names credential material a harness or a tool keeps in a
workspace, or the private-state fabric's own markers, is never exported: compilation
refuses it as declared, and the worker refuses a declared directory holding one.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shared.content import is_member_path
from shared.schemas.artifact import CHECKPOINTS_ARTIFACT
from shared.utils.manifest import RESULTS_NAME

# Paths, relative to the workspace, that are never exported, nor anything under them.
EXCLUDED_EXPORT_PATHS: tuple[str, ...] = (
    ".attachment",
    ".unsealable",
    ".git-credentials",
    ".netrc",
    ".ssh",
    ".aws",
    ".docker/config.json",
    ".npmrc",
    ".pypirc",
    ".config/gh/hosts.yml",
    ".config/gcloud",
    ".kube/config",
)


def excluded_export_path(path: str) -> str | None:
    """The excluded path ``path`` is or lies under, or None."""
    for excluded in EXCLUDED_EXPORT_PATHS:
        if path == excluded or path.startswith(f"{excluded}/"):
            return excluded
    return None


# Published names an export may not take or lie under: a task's checkpoints and the
# name its result downloads as.
RESERVED_EXPORT_NAMES: tuple[str, ...] = (CHECKPOINTS_ARTIFACT, RESULTS_NAME)


def reserved_export_name(name: str) -> str | None:
    """The reserved name ``name`` is or lies under, or None."""
    top = name.split("/", 1)[0]
    return top if top in RESERVED_EXPORT_NAMES else None


def _member_path(value: str) -> str:
    if not is_member_path(value):
        raise ValueError(f"{value!r} is not a relative path of plain segments")
    return value


class AgentExport(BaseModel):
    """One workspace path an agent publishes when it completes, under ``name``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    name: str | None = None

    @field_validator("path", "name")
    @classmethod
    def _plain(cls, value: str | None) -> str | None:
        return value if value is None else _member_path(value)

    @model_validator(mode="after")
    def _named(self) -> Self:
        if self.name is None:
            object.__setattr__(self, "name", self.path)
        return self

    @property
    def published_name(self) -> str:
        return self.name or self.path


class ExportFailure(BaseModel):
    """Why a completing agent's exports could not be published."""

    model_config = ConfigDict(frozen=True)

    kind: str
    message: str


EXPORT_MISSING = "ExportMissing"
EXPORT_UNSAFE = "ExportUnsafe"


__all__ = [
    "AgentExport",
    "EXCLUDED_EXPORT_PATHS",
    "EXPORT_MISSING",
    "EXPORT_UNSAFE",
    "ExportFailure",
    "excluded_export_path",
]
