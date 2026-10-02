import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...app_state import get_logger, get_runtime
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...schemas.outputs import (
    OutputOutcome,
    WorkflowOutputEntry,
    WorkflowOutputMember,
    WorkflowOutputPage,
    WorkflowOutputValue,
)
from ...task.outputs import OutputMember, decode_member_cursor, paginate_members
from ...task.results import ResultUnavailable, ResultUnreadable
from ...task.runtime import TaskRuntime
from ...task.v2.representations.results import CardinalityKind
from ._errors import api_error
from ._listing import (
    PAGE_LIMIT_DEFAULT,
    PageAfter,
    PageBefore,
    PageLimit,
    page_bounds,
)

router = APIRouter(prefix="/workflows", tags=["Outputs"])


async def _authorize(
    workflow_id: str, principal: PrincipalContext, logger: logging.Logger
) -> None:
    """Both checks run before anything about the outputs is looked up, so a caller who
    may not read them learns nothing about which exist."""
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )


def _no_outputs(workflow_id: str) -> HTTPException:
    return api_error(
        status.HTTP_404_NOT_FOUND,
        "output_not_found",
        f"workflow {workflow_id} has no published outputs",
    )


def _present[M: WorkflowOutputMember](
    model: type[M], member: OutputMember, **extra: Any
) -> M:
    publication = member.publication
    return model(
        name=member.name,
        cardinality=member.declaration.cardinality.value,
        value_type=member.declaration.value_type,
        scope=member.scope_id,
        key=member.key,
        sequence=member.sequence,
        outcome=(
            OutputOutcome(publication.outcome.value)
            if publication is not None
            else OutputOutcome.PENDING
        ),
        **extra,
    )


@router.get(
    "/{workflow_id}/outputs",
    summary="List published outputs",
    description=(
        "List the members of a workflow's published outputs, ordered by output name, "
        "scope, and key."
    ),
    response_description="A page of published output members",
)
async def list_outputs(
    workflow_id: str,
    limit: PageLimit = PAGE_LIMIT_DEFAULT,
    before: PageBefore = None,
    after: PageAfter = None,
    output: str | None = Query(None, description="Only this output's members."),
    scope: str | None = Query(None, description="Only members in this scope."),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> WorkflowOutputPage:
    after_bound, before_bound = page_bounds(after, before, decode_member_cursor)
    await _authorize(workflow_id, principal, logger)
    outputs = runtime.published_outputs(workflow_id, output)
    if outputs is None:
        raise _no_outputs(workflow_id)
    members = [m for m in outputs.members if scope is None or m.scope_id == scope]
    selected = paginate_members(members, limit, after=after_bound, before=before_bound)
    entries = [
        _present(WorkflowOutputEntry, member, cursor=member.cursor)
        for member in selected
    ]
    return WorkflowOutputPage(
        entries=entries,
        next_cursor=entries[-1].cursor if entries else None,
        prev_cursor=entries[0].cursor if entries else None,
        open=outputs.open,
    )


@router.get(
    "/{workflow_id}/outputs/{output_name}",
    summary="Get a published output",
    description=(
        "Get the value of one published output member. A singleton is selected by "
        "its name alone, a collection member by its scope and key. A collection "
        "whose spawn failed holds one failed member selected by its name alone."
    ),
    response_description="The published output member and its value",
)
async def get_output(
    workflow_id: str,
    output_name: str,
    scope: str | None = Query(None, description="Scope of a collection member."),
    key: str | None = Query(None, description="Key of a collection member."),
    sequence: int | None = Query(None, description="Sequence of the member."),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> WorkflowOutputValue:
    await _authorize(workflow_id, principal, logger)
    found = runtime.published_output(workflow_id, output_name, scope, key, sequence)
    if found is None:
        raise _no_outputs(workflow_id)
    if (declaration := found.declaration) is None:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "output_not_found",
            f"workflow {workflow_id} publishes no output named {output_name!r}",
        )
    keyed = declaration.cardinality is CardinalityKind.KEYED_COLLECTION
    member = found.member
    # A collection whose spawn failed holds one member with no scope and no key.
    if keyed and (scope is None or key is None) and member is None:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            f"output {output_name!r} is a collection; select a member by scope and key",
        )
    if member is None and (not keyed or not found.open):
        # A settled workflow publishes nothing more, so a missing member never comes.
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "output_not_found",
            f"output {output_name!r} has no member at the given selectors",
        )
    if member is None or member.publication is None:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "output_pending",
            f"output {output_name!r} has not settled at the given selectors",
        )
    if member.publication.outcome.value != OutputOutcome.SUCCESS:
        return _present(WorkflowOutputValue, member)
    try:
        envelope = await asyncio.to_thread(runtime.read_output, member)
    except ResultUnavailable as exc:
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "content_unavailable", str(exc)
        ) from exc
    except ResultUnreadable as exc:
        raise api_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "output_unreadable", str(exc)
        ) from exc
    return _present(WorkflowOutputValue, member, value=envelope.result)
