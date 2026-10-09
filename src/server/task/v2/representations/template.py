from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .operators import EffectBoundary, LogicalOperator, Port, PortKind, SelectorStep
from .results import LegacyLogicalTaskProjection, ResultDeclaration, Visibility
from .versioning import VersionId

type SourceKind = Literal["legacy_task", "stage", "graph_node", "region", "root"]


class DependencyUse(StrEnum):
    """What a consumer needs from one of its incoming edges."""

    VALUE_REQUIRED = "value_required"
    """The consumer reads the value; a dead route makes the consumer inactive."""
    ROUTE_REQUIRED = "route_required"
    """The consumer runs only on this route, whatever it reads from it."""
    ORDER_ONLY = "order_only"
    """The consumer only runs after the source; any live such route lets it run."""


class BoundaryKind(StrEnum):
    """Which boundary of a region definition an edge crosses."""

    ENTRY = "entry"
    """From a definition input into a member."""
    FEEDBACK = "feedback"
    """Back to the owning loop, enabling its next time."""
    EGRESS = "egress"
    """Out of the owning loop, once its frontier closes."""
    RETURN = "return"
    """Out of a child to its spawn or call."""


BOUNDARY_NODES: dict[BoundaryKind, str] = {
    BoundaryKind.ENTRY: "$ingress",
    BoundaryKind.FEEDBACK: "$feedback",
    BoundaryKind.EGRESS: "$egress",
    BoundaryKind.RETURN: "$return",
}
"""The node standing for a definition's boundary at the outer end of a boundary
edge."""
RETURN_KINDS = frozenset(
    {BoundaryKind.FEEDBACK, BoundaryKind.EGRESS, BoundaryKind.RETURN}
)


class TemplateEdge(BaseModel):
    """A symbolic wiring edge between two logical operators of one definition, or
    across that definition's boundary.

    A boundary edge names the boundary node of its kind (``$ingress`` and its input
    port, or ``$feedback``/``$egress``/``$return`` and the boundary port) at its outer
    end. ``projection`` selects part of the source value for the consumer under the
    projection rule ``projection_version``; ``use`` says whether the consumer needs the
    value, the route, or only the ordering. A derived edge stands for a value the
    consumer reads by name through an ancestor rather than a dependency it declares.
    """

    model_config = ConfigDict(frozen=True)

    from_op: str
    to_op: str
    from_port: str | None = None
    to_port: str | None = None
    edge_id: str = ""
    use: DependencyUse = DependencyUse.ORDER_ONLY
    projection: tuple[SelectorStep, ...] = ()
    projection_version: int = 1
    boundary: BoundaryKind | None = None
    # The region definition the edge belongs to; None at the root.
    definition: str | None = None
    derived: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_stored(cls, data: Any) -> Any:
        """Name an edge stored without an identity by its endpoints and ports, read a
        stored port binding as a value delivery, and a stored feedback flag as a
        feedback boundary."""
        if not isinstance(data, dict):
            return data
        data = data.copy()
        if data.pop("feedback", False) and data.get("boundary") is None:
            data["boundary"] = BoundaryKind.FEEDBACK
        if "use" not in data and data.get("to_port") is not None:
            data["use"] = DependencyUse.VALUE_REQUIRED
        if not data.get("edge_id"):
            data["edge_id"] = (
                f"{data.get('from_op')}.{data.get('from_port') or ''}"
                f"->{data.get('to_op')}.{data.get('to_port') or ''}"
            )
        return data

    @property
    def is_forward(self) -> bool:
        """Whether the edge joins two operators of one definition."""
        return self.boundary is None


class DefinitionKind(StrEnum):
    """What enters a region definition."""

    LOOP_BODY = "loop_body"
    """A loop runs it at every logical time."""
    CHILD = "child"
    """A spawn or call runs it as one child activation."""


class EntryRole(StrEnum):
    """How a definition input binds at entry."""

    CARRIED = "carried"
    """Seeded at loop ingress and replaced by each feedback."""
    INVARIANT = "invariant"
    """Bound once at loop ingress, readable at every time."""
    PARAM = "param"
    """A child's invocation parameter: its spawned element or call input."""
    CAPTURE = "capture"
    """A parent value a child captures at entry."""


class DefinitionPort(BaseModel):
    """A typed input of a region definition."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: PortKind = PortKind.VALUE
    role: EntryRole


class RegionDefinition(BaseModel):
    """A finite declared subgraph a loop or a spawn/call enters.

    Its members are operators of the template; its edges stay among members, and the
    definition crosses its boundary only through its boundary edges.
    """

    model_config = ConfigDict(frozen=True)

    definition_id: str
    kind: DefinitionKind
    source_id: str
    members: tuple[str, ...] = ()
    inputs: tuple[DefinitionPort, ...] = ()
    returns: tuple[Port, ...] = ()


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
    source_map: tuple[SourceMapEntry, ...] = ()
    definitions: tuple[RegionDefinition, ...] = ()

    @property
    def operator_ids(self) -> frozenset[str]:
        return frozenset(op.operator_id for op in self.operators)

    def definition_of(self) -> dict[str, str]:
        """Each definition member's definition id; a root operator is absent."""
        return {
            member: definition.definition_id
            for definition in self.definitions
            for member in definition.members
        }

    def boundary_edges(
        self, definition_id: str, *kinds: BoundaryKind
    ) -> list[TemplateEdge]:
        """A definition's boundary edges of the given kinds, in catalog order."""
        return [
            edge
            for edge in self.edges
            if edge.definition == definition_id
            and edge.boundary is not None
            and (not kinds or edge.boundary in kinds)
        ]

    def published_outputs(self) -> list[tuple[str, ResultDeclaration]]:
        """Each published declaration with its public name, in declaration order.

        The public name is the node the author wrote the output on, read through the
        source map, so an internal output id never reaches a client.
        """
        names = {entry.logical_ref: entry.source_id for entry in self.source_map}
        return [
            (names.get(decl.source_ref, decl.source_ref), decl)
            for decl in self.result_declarations
            if decl.visibility is Visibility.PUBLISHED
        ]

    @model_validator(mode="after")
    def _validate_ownership_links(self) -> "LogicalWorkflowTemplate":
        ids = self.operator_ids
        if len(ids) != len(self.operators):
            duplicated = sorted(
                op_id
                for op_id, count in Counter(
                    op.operator_id for op in self.operators
                ).items()
                if count > 1
            )
            raise ValueError(f"Duplicate operator_id in logical template: {duplicated}")
        for edge in self.edges:
            for ref in (edge.from_op, edge.to_op):
                if ref not in ids and not (
                    edge.boundary is not None and ref == _outer_end(edge)
                ):
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
        for definition in self.definitions:
            for ref in definition.members:
                if ref not in ids:
                    raise ValueError(
                        f"Definition {definition.definition_id!r} references unknown "
                        f"operator {ref!r}."
                    )
        return self


def _outer_end(edge: TemplateEdge) -> str | None:
    if edge.boundary is None:
        return None
    return BOUNDARY_NODES[edge.boundary]
