"""Pin each agent's fabric-owned facade tools at compile time.

An agent's facades are derived from its declared authority and child regions: a
fabric-served tool interface it may invoke (``search/v1``) becomes an injected
function-tool facade, and a spawnable agent gets the ``spawn_agent`` facade. The model
gateway injects only the facades pinned here, so an agent can never call a fabric tool
it did not declare.
"""

import json
from typing import Any

from shared.harness.boundary import BoundaryEventKind

from ....orchestration.tool_dispatch import FABRIC_TOOL_INTERFACES
from ..representations.operators import AgentOperator, FacadeDescriptor
from .project import LoweringAccumulator

_SPAWN_AGENT_NAME = "spawn_agent"
_DEFAULT_SEARCH_NAME = "web_search"


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
    """Derive and pin each agent operator's facade set on the accumulator in place."""
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
        if facades:
            acc.operators[index] = op.model_copy(update={"facades": tuple(facades)})
