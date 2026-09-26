import asyncio
import logging

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
from ...task.outputs import InvalidCursor, OutputMember, PublishedOutputs, page
from ...task.results import ResultUnavailable, ResultUnreadable
from ...task.runtime import TaskRuntime

router = APIRouter(prefix="/workflows", tags=["Outputs"])


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message})


async def _published(
    workflow_id: str,
    principal: PrincipalContext,
    runtime: TaskRuntime,
    logger: logging.Logger,
) -> PublishedOutputs:
    """A workflow's published outputs, once the caller may read the workflow's results.

    Both checks run before anything about the outputs is looked up, so a caller who may
    not read them learns nothing about which exist.
    """
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.READ, logger
    )
    outputs = runtime.published_outputs(workflow_id)
    if outputs is None or (
        outputs.org_id != principal.org_id and "*" not in principal.scopes
    ):
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "output_not_found",
            f"workflow {workflow_id} has no published outputs",
        )
    return outputs


def _member(member: OutputMember) -> WorkflowOutputMember:
    publication = member.publication
    return WorkflowOutputMember(
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
    limit: int = Query(100, ge=1, le=1000, description="Maximum members to return."),
    before: str | None = Query(
        None, description="Return members strictly before this cursor."
    ),
    after: str | None = Query(
        None, description="Return members strictly after this cursor."
    ),
    output: str | None = Query(None, description="Only this output's members."),
    scope: str | None = Query(None, description="Only members in this scope."),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> WorkflowOutputPage:
    if before and after:
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "only one of before/after may be set",
        )
    outputs = await _published(workflow_id, principal, runtime, logger)
    members = [
        member
        for member in outputs.members
        if (output is None or member.name == output)
        and (scope is None or member.scope_id == scope)
    ]
    try:
        selected = page(members, limit, after=after, before=before)
    except InvalidCursor as exc:
        raise _error(status.HTTP_400_BAD_REQUEST, "invalid_cursor", str(exc)) from exc
    entries = [
        WorkflowOutputEntry(cursor=member.cursor, **dict(_member(member)))
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
        "Get the value of one published output member. A collection member is "
        "selected by its scope and key."
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
    name = output_name
    outputs = await _published(workflow_id, principal, runtime, logger)
    named = [member for member in outputs.members if member.name == name]
    if not named:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "output_not_found",
            f"workflow {workflow_id} publishes no output named {name!r}",
        )
    if named[0].keyed and (scope is None or key is None):
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            f"output {name!r} is a collection; select a member by scope and key",
        )
    member = next(
        (m for m in named if (m.scope_id, m.key, m.sequence) == (scope, key, sequence)),
        None,
    )
    if member is None and not named[0].keyed:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "output_not_found",
            f"output {name!r} has no member at the given selectors",
        )
    if member is None or member.publication is None:
        raise _error(
            status.HTTP_409_CONFLICT,
            "output_pending",
            f"output {name!r} has not settled at the given selectors",
        )
    presented = _member(member)
    if member.publication.outcome.value != OutputOutcome.SUCCESS:
        return WorkflowOutputValue(**dict(presented))
    try:
        envelope = await asyncio.to_thread(runtime.read_output, member)
    except ResultUnavailable as exc:
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "content_unavailable", str(exc)
        ) from exc
    except ResultUnreadable as exc:
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "output_unreadable", str(exc)
        ) from exc
    return WorkflowOutputValue(value=envelope.result, **dict(presented))
