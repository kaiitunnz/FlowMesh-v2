"""Which upstream values a task's spec reads, and what each dependency is used for.

A task reads an upstream value through a ``${name.path}`` placeholder rendered at
dispatch, a ``data.expr``/``data.node`` projection or ``graph_template`` column
resolved on its worker, its guard's ``condition.node``, or an SSH ``inputs[].stage``.
``${name.task_id}`` reads only the upstream's identity. Extraction follows those
grammars without evaluating any of them.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from shared.tasks.placeholders import PLACEHOLDER_PATTERN

from ...parser import INGRESS, ParsedDependency, ParsedTask
from ..representations.template import DependencyUse

_EXPR_ROOT = re.compile(r"[.\[]")


@dataclass(frozen=True)
class SpecReads:
    """The upstream names a spec reads, split by what it reads of them."""

    values: frozenset[str] = frozenset()
    identities: frozenset[str] = frozenset()
    # Placeholders naming no ``stage.path``, which render to nothing.
    malformed: tuple[str, ...] = ()


def spec_reads(task: ParsedTask) -> SpecReads:
    """Collect the upstream names a task's spec reads."""
    values: set[str] = set()
    identities: set[str] = set()
    malformed: list[str] = []
    for text in _strings(task.task.model_dump(mode="python")):
        for match in PLACEHOLDER_PATTERN.finditer(text):
            expr = match.group(1).strip()
            name, dot, path = expr.partition(".")
            if not dot or not name.strip():
                malformed.append(match.group(0))
            elif path.strip() == "task_id":
                identities.add(name.strip())
            else:
                values.add(name.strip())
    spec = task.task.spec.model_dump(mode="python")
    values.update(_data_sources(spec.get("data")))
    if isinstance(condition := spec.get("condition"), dict):
        if isinstance(node := condition.get("node"), str) and node.strip():
            values.add(node.strip())
    for entry in spec.get("inputs") or ():
        if isinstance(entry, dict) and isinstance(stage := entry.get("stage"), str):
            values.add(stage.strip())
    return SpecReads(frozenset(values), frozenset(identities), tuple(malformed))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _data_sources(data: Any) -> Iterable[str]:
    """The upstream node each ``expr``/``node`` projection under ``data`` names."""
    if isinstance(data, Mapping):
        for key, item in data.items():
            if key == "expr" and isinstance(item, str) and item.strip():
                yield _EXPR_ROOT.split(item.strip(), maxsplit=1)[0]
            elif key == "node" and isinstance(item, str) and item.strip():
                yield item.strip()
            else:
                yield from _data_sources(item)
    elif isinstance(data, list):
        for item in data:
            yield from _data_sources(item)


@dataclass(frozen=True)
class ReadClassification:
    """A task's dependency uses, the operators it reads, and the names it cannot."""

    uses: tuple[DependencyUse, ...]
    # Operators whose values the spec reads, directly or through an ancestor name.
    value_reads: tuple[str, ...]
    # Entry ports the spec reads, for a task inside a region definition.
    entry_reads: frozenset[str]
    unresolved: tuple[str, ...]


def classify_reads(
    task: ParsedTask,
    names: Mapping[str, str],
    value_op: Mapping[str, str],
    ancestors: frozenset[str],
    routed: frozenset[str],
) -> ReadClassification:
    """Classify each of a task's dependencies by what its spec needs from it.

    ``names`` maps the names visible in the task's scope to operators; each binding's
    ``input`` name is visible too, as is each ``$ingress`` binding's. ``ancestors`` are
    the operators an upstream name may resolve through, and ``routed`` the operators
    whose outputs are branch arms. A dependency the spec reads, or one with a named
    input, is a required value; an unread branch arm is a required route; any other
    is ordering only.
    """
    reads = spec_reads(task)
    aliases = {dep.input: dep for dep in task.dependencies if dep.input}
    resolved_values: set[str] = set()
    entry_reads: set[str] = set()
    unresolved: list[str] = list(reads.malformed) if task.dependencies else []
    identity_ops: set[str] = set()

    def _resolve(name: str) -> str | None:
        if (dep := aliases.get(name)) is not None:
            if dep.source == INGRESS:
                entry_reads.add(dep.port or name)
                return None
            return value_op.get(dep.source, dep.source)
        if (op := names.get(name)) is not None and op in ancestors:
            return op
        if name == INGRESS:
            return None
        if task.dependencies:
            unresolved.append(name)
        return None

    for name in sorted(reads.values):
        if (op := _resolve(name)) is not None:
            resolved_values.add(op)
    for name in sorted(reads.identities):
        if (op := _resolve(name)) is not None:
            identity_ops.add(op)

    uses: list[DependencyUse] = []
    for dep in task.dependencies:
        source = value_op.get(dep.source, dep.source)
        uses.append(_dependency_use(dep, source, resolved_values, identity_ops, routed))
    return ReadClassification(
        uses=tuple(uses),
        value_reads=tuple(sorted(resolved_values)),
        entry_reads=frozenset(entry_reads),
        unresolved=tuple(dict.fromkeys(unresolved)),
    )


def _dependency_use(
    dep: ParsedDependency,
    source: str,
    values: set[str],
    identities: set[str],
    routed: frozenset[str],
) -> DependencyUse:
    if dep.input is not None or source in values:
        return DependencyUse.VALUE_REQUIRED
    if source in identities or (dep.port is not None and source in routed):
        return DependencyUse.ROUTE_REQUIRED
    return DependencyUse.ORDER_ONLY
