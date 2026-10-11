"""A freed workflow's history: what its task and output reads derive from runtime
state, kept beside its stored task records once the runtime frees it."""

import json
import struct
from typing import Any

from pydantic import BaseModel, Field

from shared.tasks.result_binding import ResultBinding

from ..orchestration.state import ResultPublication
from .models import TaskInputElement, TaskOccurrence
from .v2.representations.results import ResultDeclaration

HISTORY_SCHEMA = 1

# The history hash fields holding a workflow's certificate and its published outputs'
# declarations. A field starting with "@" holds the certificate, a declaration list, a
# published output's member list or one member; every other field is one of its tasks.
CERTIFICATE_FIELD = "@certificate"
DECLARATIONS_FIELD = "@declarations"

_FLIP_ALL = 0xFFFF_FFFF_FFFF_FFFF
_FLIP_SIGN = 0x8000_0000_0000_0000


class TaskDerived(BaseModel):
    """The task-info fields a task's runtime state derives beside its record."""

    depends_on: list[str] = Field(default_factory=list)
    pending_dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    completed: bool = False
    failed: bool = False
    input_element: TaskInputElement | None = None
    occurrence: TaskOccurrence | None = None

    def info_fields(self) -> dict[str, Any]:
        """The derived fields, by name, as a task's details carry them."""
        return {name: getattr(self, name) for name in TaskDerived.model_fields}


class TaskHistory(TaskDerived):
    """The task-info fields and result a freed task's runtime state derived."""

    result: ResultBinding | None = None


class OutputHistory(BaseModel):
    """One published output member, and the value binding its read resolves to."""

    name: str
    scope_id: str | None = None
    key: str | None = None
    sequence: int | None = None
    publication: ResultPublication | None = None
    value: ResultBinding | None = None
    # Why the member's value cannot be read, when it cannot.
    unreadable: str | None = None

    @property
    def field(self) -> str:
        return output_member_field(self.name, self.scope_id, self.key, self.sequence)


class OutputsHistory(BaseModel):
    """A freed workflow's published outputs: each declaration, and the members of
    those read."""

    declarations: list[tuple[str, ResultDeclaration]] = Field(default_factory=list)
    members: list[OutputHistory] = Field(default_factory=list)


def output_index_field(name: str) -> str:
    """The history hash field listing the member fields of one published output."""
    return f"@outputs:{name}"


def output_member_field(
    name: str, scope_id: str | None, key: str | None, sequence: int | None
) -> str:
    """The history hash field holding one published output member."""
    return "@output:" + json.dumps(
        [name, scope_id, key, sequence], separators=(",", ":")
    )


class HistoryCertificate(BaseModel):
    """That a settled workflow's history is complete and nothing still needs its
    runtime state, and the sequence the workflow was freed at."""

    schema_version: int
    seq: int


def history_member(submitted_ts: float, task_id: str) -> str:
    """A task's member in the history indexes, whose lexical order is its listing
    order by submission time, then id."""
    (bits,) = struct.unpack(">Q", struct.pack(">d", submitted_ts + 0.0))
    bits ^= _FLIP_ALL if bits & _FLIP_SIGN else _FLIP_SIGN
    return f"{bits:016x}:{task_id}"


def member_task_id(member: str) -> str:
    return member.partition(":")[2]


__all__ = [
    "CERTIFICATE_FIELD",
    "DECLARATIONS_FIELD",
    "HISTORY_SCHEMA",
    "HistoryCertificate",
    "OutputHistory",
    "OutputsHistory",
    "TaskDerived",
    "TaskHistory",
    "history_member",
    "member_task_id",
    "output_index_field",
    "output_member_field",
]
