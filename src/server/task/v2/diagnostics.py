from enum import StrEnum

from pydantic import BaseModel, ConfigDict

_SOURCE_KIND_LABELS = {
    "legacy_task": "task",
    "stage": "stage",
    "graph_node": "graph node",
    "region": "region",
    "root": "workflow",
}


class Severity(StrEnum):
    """Whether a diagnostic blocks compilation or is advisory."""

    ERROR = "error"
    WARNING = "warning"


class SourceLocation(BaseModel):
    """A readable pointer from a diagnostic back into the frontend source.

    The frontend parser retains no line/column information, so a location names
    the authoring construct (task, stage, graph node, region) that produced the
    logical obligation, which is what an author recognizes.
    """

    model_config = ConfigDict(frozen=True)

    source_kind: str
    source_id: str
    detail: str | None = None

    def render(self) -> str:
        label = _SOURCE_KIND_LABELS.get(self.source_kind, self.source_kind)
        base = f"{label} {self.source_id!r}"
        return f"{base} ({self.detail})" if self.detail else base


class Diagnostic(BaseModel):
    """One validation finding with a readable source location."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    severity: Severity = Severity.ERROR
    location: SourceLocation | None = None

    def render(self) -> str:
        loc = f" at {self.location.render()}" if self.location is not None else ""
        return f"[{self.code}] {self.message}{loc}"


class CompileError(Exception):
    """Raised when compilation produces one or more error-severity diagnostics."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        rendered = "; ".join(
            diag.render() for diag in diagnostics if diag.severity is Severity.ERROR
        )
        super().__init__(rendered or "workflow compilation failed")
