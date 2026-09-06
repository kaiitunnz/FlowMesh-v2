"""Capture a held model turn's facade calls into a control-recordable turn group.

The facade injects an agent's pinned fabric tools into a model turn; when the model
co-emits calls to them, this captures them into a ``FacadeTurnGroup`` whose search
members carry a request digest (their raw request stays worker-private, kept in custody
for the worker egress) and whose spawn members carry their args for control to admit a
child. The digest is what routes a search member to the worker egress rather than the
in-server broker. The group id and each member's correlation derive from ``(task, turn
base)``, so a re-drive of the same turn recovers the same identities and makes no
duplicate work.
"""

import json
from dataclasses import dataclass
from typing import Any

from shared.harness import BoundaryEventKind
from shared.tools.facade import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeDescriptor,
    FacadeTurnGroup,
)
from shared.tools.model.schema import ModelToolCall
from shared.tools.search.schema import (
    ToolRequest,
    parse_search_request,
    tool_request_digest,
)


@dataclass(frozen=True)
class FacadeCapture:
    """A captured turn group plus the worker-private search requests to keep."""

    group: FacadeTurnGroup
    stashes: tuple[tuple[str, ToolRequest], ...]


def partition_facade_calls(
    tool_calls: tuple[ModelToolCall, ...], descriptors: list[FacadeDescriptor]
) -> tuple[list[ModelToolCall], list[ModelToolCall]]:
    """Split a turn's tool calls into the fabric-facade calls and the rest."""
    names = {d.name for d in descriptors}
    facade = [call for call in tool_calls if call.name in names]
    other = [call for call in tool_calls if call.name not in names]
    return facade, other


def build_facade_capture(
    task_id: str,
    facade_calls: list[ModelToolCall],
    descriptors: list[FacadeDescriptor],
    turn_base: int,
) -> FacadeCapture:
    """Build the turn group for a turn's facade calls, ordered by emission.

    A search member carries the digest of its worker-private request; a spawn member
    carries its args and target region. The search requests are returned for the caller
    to keep in custody, so the worker egress reads them back under the recorded digest.
    """
    by_name = {d.name: d for d in descriptors}
    group_id = f"{task_id}:{turn_base}"
    members: list[FacadeCallMember] = []
    stashes: list[tuple[str, ToolRequest]] = []
    for ordinal, call in enumerate(facade_calls):
        descriptor = by_name[call.name]
        correlation = f"{group_id}:{ordinal}"
        if descriptor.kind is BoundaryEventKind.SPAWN:
            members.append(
                FacadeCallMember(
                    ordinal=ordinal,
                    kind=BoundaryEventKind.SPAWN,
                    completion_mode=FacadeCompletionMode.ADMIT_AND_CLOSE,
                    call_correlation=correlation,
                    harness_call_id=call.call_id,
                    tool_name=call.name,
                    interface_or_region=_spawn_region(call.arguments),
                    request_payload=call.arguments,
                )
            )
        else:
            request = parse_search_request(call.arguments)
            members.append(
                FacadeCallMember(
                    ordinal=ordinal,
                    kind=BoundaryEventKind.INVOCATION,
                    completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
                    call_correlation=correlation,
                    harness_call_id=call.call_id,
                    tool_name=call.name,
                    interface_or_region=descriptor.interface,
                    request_digest=tool_request_digest(
                        request.interface, request.query, request.max_results
                    ),
                )
            )
            stashes.append((correlation, request))
    group = FacadeTurnGroup(
        group_id=group_id,
        activation_id=task_id,
        turn_id=str(turn_base),
        members=tuple(members),
    )
    return FacadeCapture(group=group, stashes=tuple(stashes))


def turn_base(responses_input: Any) -> int:
    """The settled-outcome count in a Responses input history, for a re-drive-stable id.

    A harness posts its full turn history each turn, so a re-drive before a facade's
    outcome injects derives the same base, while every injected outcome adds a
    function-call output that advances the base for the next turn's facades.
    """
    if not isinstance(responses_input, list):
        return 0
    return sum(
        1
        for item in responses_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )


def _spawn_region(arguments: str) -> str | None:
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    return args.get("region") if isinstance(args, dict) else None
