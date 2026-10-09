from enum import StrEnum
from typing import Literal

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


class TemplateEdge(BaseModel):
    """A symbolic wiring edge between two logical operators of one definition.

    ``projection`` selects part of the source value for the consumer; ``use`` says
    whether the consumer needs the value, the route, or only the ordering.
    """

    model_config = ConfigDict(frozen=True)

    from_op: str
    to_op: str
    from_port: str | None = None
    to_port: str | None = None
    # A feedback edge is a structured back-edge into a LoopContext region,
    # excluded from the forward topology that unstructured-cycle detection rejects.
    feedback: bool = False
    edge_id: str = ""
    use: DependencyUse = DependencyUse.ORDER_ONLY
    projection: tuple[SelectorStep, ...] = ()


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


class ReturnKind(StrEnum):
    """Where a definition's return binding leaves to."""

    FEEDBACK = "feedback"
    """Back to the owning loop, enabling its next time."""
    EGRESS = "egress"
    """Out of the owning loop, once its frontier closes."""
    RETURN = "return"
    """Out of a child to its spawn or call."""


class DefinitionPort(BaseModel):
    """A typed input of a region definition."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: PortKind = PortKind.VALUE
    role: EntryRole


class EntryBinding(BaseModel):
    """Delivers a definition input to one member's input."""

    model_config = ConfigDict(frozen=True)

    port: str
    to_op: str
    to_port: str | None = None
    use: DependencyUse = DependencyUse.VALUE_REQUIRED
    projection: tuple[SelectorStep, ...] = ()


class ReturnBinding(BaseModel):
    """Delivers one member output out of a definition through a boundary port."""

    model_config = ConfigDict(frozen=True)

    kind: ReturnKind
    port: str
    from_op: str
    from_port: str | None = None
    projection: tuple[SelectorStep, ...] = ()


class RegionDefinition(BaseModel):
    """A finite declared subgraph a loop or a spawn/call enters.

    Its members are operators of the template; ``edges`` stay among members, and the
    definition crosses its boundary only through its entry and return bindings.
    """

    model_config = ConfigDict(frozen=True)

    definition_id: str
    kind: DefinitionKind
    source_id: str
    members: tuple[str, ...] = ()
    inputs: tuple[DefinitionPort, ...] = ()
    returns: tuple[Port, ...] = ()
    entries: tuple[EntryBinding, ...] = ()
    return_bindings: tuple[ReturnBinding, ...] = ()


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
        for definition in self.definitions:
            for ref in definition.members:
                if ref not in ids:
                    raise ValueError(
                        f"Definition {definition.definition_id!r} references unknown "
                        f"operator {ref!r}."
                    )
        return self
