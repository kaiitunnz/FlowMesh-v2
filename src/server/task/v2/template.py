from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .operators import EffectBoundary, LogicalOperator
from .results import LegacyLogicalTaskProjection, ResultDeclaration
from .versioning import VersionId

type SourceKind = Literal["legacy_task", "stage", "graph_node", "root"]


class TemplateEdge(BaseModel):
    """A symbolic wiring edge between two logical operators."""

    model_config = ConfigDict(frozen=True)

    from_op: str
    to_op: str
    from_port: str | None = None
    to_port: str | None = None


class ToolDeclaration(BaseModel):
    """A declared typed tool/service interface within an authority ceiling."""

    model_config = ConfigDict(frozen=True)

    name: str
    interface: str | None = None
    authority_ref: str | None = None


class ResourceDeclaration(BaseModel):
    """A declared resource interface referenced by the workflow."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str | None = None


class SourceMapEntry(BaseModel):
    """Maps a logical operator to its frontend source location."""

    model_config = ConfigDict(frozen=True)

    logical_ref: str
    source_kind: SourceKind
    source_id: str


class LogicalWorkflowTemplate(BaseModel):
    """The durable, symbolic plan-time object describing legal workflow behavior.

    It carries typed operators, port wiring, declared tools/resources, result
    declarations, legacy logical-task projections, effect boundaries, and source
    maps. It carries no activation tags and no worker/replica/endpoint bindings.
    """

    model_config = ConfigDict(frozen=True)

    version: VersionId
    operators: tuple[LogicalOperator, ...] = ()
    edges: tuple[TemplateEdge, ...] = ()
    tool_declarations: tuple[ToolDeclaration, ...] = ()
    resource_declarations: tuple[ResourceDeclaration, ...] = ()
    result_declarations: tuple[ResultDeclaration, ...] = ()
    legacy_projection: tuple[LegacyLogicalTaskProjection, ...] = ()
    effect_boundaries: tuple[EffectBoundary, ...] = ()
    source_map: tuple[SourceMapEntry, ...] = Field(default=())

    @property
    def operator_ids(self) -> frozenset[str]:
        return frozenset(op.operator_id for op in self.operators)

    @model_validator(mode="after")
    def _validate_ownership_links(self) -> "LogicalWorkflowTemplate":
        ids = self.operator_ids
        if len(ids) != len(self.operators):
            raise ValueError("Duplicate operator_id in logical template.")
        for edge in self.edges:
            for ref in (edge.from_op, edge.to_op):
                if ref not in ids:
                    raise ValueError(f"Edge references unknown operator {ref!r}.")
        for decl in self.result_declarations:
            if decl.source_ref not in ids:
                raise ValueError(
                    f"Result declaration {decl.output_id!r} references unknown "
                    f"operator {decl.source_ref!r}."
                )
        for proj in self.legacy_projection:
            if proj.operator_id not in ids:
                raise ValueError(
                    f"Legacy projection {proj.legacy_task_id!r} references unknown "
                    f"operator {proj.operator_id!r}."
                )
        for entry in self.source_map:
            if entry.logical_ref not in ids:
                raise ValueError(
                    f"Source map references unknown operator {entry.logical_ref!r}."
                )
        return self
