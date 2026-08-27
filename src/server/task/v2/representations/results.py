from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CardinalityKind(StrEnum):
    """The shape of a declared logical output."""

    SINGLETON = "singleton"
    KEYED_COLLECTION = "keyed_collection"
    APPEND_STREAM = "append_stream"
    AGGREGATE = "aggregate"


class ReleaseConditionKind(StrEnum):
    """When a declared output becomes fetchable."""

    SOURCE_SETTLED = "source_settled"
    SCOPE_CLOSED = "scope_closed"
    JOIN_WINNER = "join_winner"


class Visibility(StrEnum):
    """Whether a declared output is user-published or internal."""

    PUBLISHED = "published"
    INTERNAL = "internal"


class ResultDeclaration(BaseModel):
    """A declared logical output of the workflow.

    Carries the logical output identity, its cardinality kind, release
    condition, visibility/retention, keying, and the source port or region it
    resolves from. Exact serialization is left open for later compiler work; a
    declaration is not an instruction to materialize a result for every operator
    or attempt.
    """

    model_config = ConfigDict(frozen=True)

    output_id: str
    source_ref: str = Field(
        description="Logical operator/region/port this resolves from."
    )
    cardinality: CardinalityKind = CardinalityKind.SINGLETON
    release: ReleaseConditionKind = ReleaseConditionKind.SOURCE_SETTLED
    visibility: Visibility = Visibility.PUBLISHED
    retention: str | None = None
    keying: str | None = None
    value_type: str | None = Field(
        default=None,
        description="Declared value-type identity (e.g. result task_type).",
    )


class LegacyLogicalTaskProjection(BaseModel):
    """Maps one legacy task result value to an induced logical-output slot.

    Only the legacy task's structured result value is projected. Its logs and
    arbitrary artifacts stay source-mapped diagnostics under the legacy task
    identity; they are not promoted to logical output slots.
    """

    model_config = ConfigDict(frozen=True)

    legacy_task_id: str
    operator_id: str
    induced_output_id: str
    value_type: str = Field(description="Result task_type keying the induced value.")
    source_ref: str
