"""Pin each agent's fabric-owned facade tools at compile time.

An agent's facades are derived from its declared authority and child regions: a
fabric-served tool interface it may invoke (``search/v1``) becomes an injected
function-tool facade, and a spawnable agent gets the ``spawn_agent`` facade. An agent
can never call a fabric tool it did not declare.

What is pinned here is the declared upper bound, not the set one dispatch offers: an
activation's effective grant may be narrower than its operator's ceiling, and the
dispatch projects the locally resolved facades onto it.
"""

import json
from typing import Any

from shared.harness.boundary import BoundaryEventKind
from shared.sandbox import SANDBOX_EXECUTE_INTERFACE
from shared.tools.facade import FacadeResolution

from ....orchestration.tool_dispatch import FABRIC_TOOL_INTERFACES
from ..representations.operators import AgentOperator, FacadeDescriptor
from .project import LoweringAccumulator
from .sandbox import egress_requested

_SPAWN_AGENT_NAME = "spawn_agent"
_DEFAULT_SEARCH_NAME = "web_search"
_RUN_COMMAND_NAME = "run_command"


def _spawn_agent_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": _SPAWN_AGENT_NAME,
        "description": (
            "Delegate a subtask to a declared child agent region and await its "
            "mediated result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "the declared child region role name",
                },
                "args": {
                    "type": "object",
                    "description": "structured input for the child agent",
                },
            },
            "required": ["region"],
        },
    }


def run_command_schema(egress: bool) -> str:
    """The injected ``run_command`` tool schema for the fence a dispatch grants.

    The compiler renders it for the pinned binding, which is the upper bound; a dispatch
    re-renders it when the activation's effective grant is narrower than what the
    binding asked for, so the model is told the fence it actually has.
    """
    return json.dumps(_run_command_body(egress))


def _run_command_body(egress: bool) -> dict[str, Any]:
    # Truthful in both modes: a model told it has no network will not try, and one told
    # it has network will use the egress its workflow paid to authorize.
    network = (
        "the command can reach the network, and anything it does out there is not "
        "retried or deduplicated for you"
        if egress
        else "the command has no network access"
    )
    return {
        "type": "function",
        "name": _RUN_COMMAND_NAME,
        "description": (
            "Run a command in your own workspace and return its exit code, stdout, "
            f"and stderr. The workspace is the only writable path and {network}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "the program and its arguments",
                },
                "timeout_sec": {
                    "type": "number",
                    "description": "how long to allow the command to run",
                },
            },
            "required": ["command"],
        },
    }


def _search_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": (
            "Search the web for current information and return ranked results with "
            "titles, URLs, and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the search query"},
                "max_results": {
                    "type": "integer",
                    "description": "how many results to return",
                },
            },
            "required": ["query"],
        },
    }


def pin_agent_facades(acc: LoweringAccumulator) -> None:
    """Derive and pin each agent operator's facade ceiling, on the accumulator."""
    tool_name_for = {
        tool.interface: tool.name
        for tool in acc.tool_declarations
        if tool.interface is not None
    }
    for index, op in enumerate(acc.operators):
        if not isinstance(op, AgentOperator):
            continue
        facades: list[FacadeDescriptor] = []
        if op.child_region_refs or op.child_template_ref is not None:
            facades.append(
                FacadeDescriptor(
                    name=_SPAWN_AGENT_NAME,
                    kind=BoundaryEventKind.SPAWN,
                    tool_schema=json.dumps(_spawn_agent_schema()),
                )
            )
        for interface in op.authority.invoke:
            if interface not in FABRIC_TOOL_INTERFACES:
                continue
            name = tool_name_for.get(interface, _DEFAULT_SEARCH_NAME)
            facades.append(
                FacadeDescriptor(
                    name=name,
                    kind=BoundaryEventKind.INVOCATION,
                    interface=interface,
                    tool_schema=json.dumps(_search_schema(name)),
                )
            )
        if op.sandbox_binding is not None:
            # The worker resolves this one in the held turn: it is the agent's own
            # fenced state transition, not a call the fabric settles.
            facades.append(
                FacadeDescriptor(
                    name=_RUN_COMMAND_NAME,
                    kind=BoundaryEventKind.STATE_ACCESS,
                    interface=SANDBOX_EXECUTE_INTERFACE,
                    tool_schema=run_command_schema(egress_requested(op)),
                    resolution=FacadeResolution.LOCAL_INLINE,
                )
            )
        if facades:
            acc.operators[index] = op.model_copy(update={"facades": tuple(facades)})
