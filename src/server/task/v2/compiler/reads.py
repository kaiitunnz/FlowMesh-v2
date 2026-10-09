"""Which upstream values a task's spec reads, and what each dependency is used for.

A task reads an upstream value through a ``${name.path}`` placeholder rendered at
dispatch, a ``data.expr``/``data.node`` projection or ``graph_template`` column
resolved on its worker, its guard's ``condition.node``, or an SSH ``inputs[].stage``.
Inside a region definition every input is a value, so ``${name}`` reads one whole.
``${name.task_id}`` reads only the upstream's identity, which a task's result has and
a value routed through a definition input, a projection, or a region does not.
Extraction follows those grammars without evaluating any of them.
"""

import re
from collections.abc import Container, Iterable, Mapping
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
    # SSH ``inputs[].stage`` names, which mount a task's result; also in ``values``.
    stages: frozenset[str] = frozenset()


def spec_reads(task: ParsedTask) -> SpecReads:
    """Collect the upstream names a task's spec reads."""
    values: set[str] = set()
    identities: set[str] = set()
    stages: set[str] = set()
    malformed: list[str] = []
    scoped = task.definition is not None
    for text in _strings(task.task.model_dump(mode="python")):
        for match in PLACEHOLDER_PATTERN.finditer(text):
            expr = match.group(1).strip()
            name, dot, path = expr.partition(".")
            if not name.strip() or not (dot or scoped):
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
            stages.add(stage.strip())
    values.update(stages)
    return SpecReads(
        frozenset(values), frozenset(identities), tuple(malformed), frozenset(stages)
    )


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
    """What a task needs from each of its dependencies and from the upstreams it
    reads by name without depending on them directly."""

    uses: tuple[DependencyUse, ...]
    # Each upstream operator read through an ancestor name, with the use it needs.
    derived: tuple[tuple[str, DependencyUse], ...]
    unresolved: tuple[str, ...]
    # Input names that would hide a different node of the task's scope.
    shadowing: tuple[str, ...]
    # Names read for a task identity that carry a value with none.
    identityless: tuple[str, ...] = ()
    # Each upstream node read by its node name rather than an input name, with the
    # operator producing its value.
    node_reads: tuple[tuple[str, str], ...] = ()
    # Each SSH input stage read from an operator, with that operator.
    stage_reads: tuple[tuple[str, str], ...] = ()


def binding_name(dep: ParsedDependency) -> str | None:
    """The name a dependency makes visible to its consumer's spec: its input name,
    or for a ``$ingress`` entry its definition input."""
    if dep.source == INGRESS:
        return dep.input or dep.port
    return dep.input


def unnamed_projection(
    dependencies: Iterable[ParsedDependency], names: Mapping[str, str]
) -> str | None:
    """The authored name of the first dependency that projects a value it gives no
    name to read by, if any; ``names`` maps the scope's names to operators."""
    source = next(
        (dep.source for dep in dependencies if dep.project and not binding_name(dep)),
        None,
    )
    if source is None:
        return None
    return next((name for name, op in names.items() if op == source), source)


def classify_reads(
    task: ParsedTask,
    dependencies: list[ParsedDependency],
    names: Mapping[str, str],
    value_op: Mapping[str, str],
    ancestors: Container[str],
    routed: frozenset[str],
    regions: frozenset[str],
) -> ReadClassification:
    """Classify each of a task's dependencies by what its spec needs from it.

    ``names`` maps the names visible in the task's scope to operators; each
    dependency's binding name is visible too. ``ancestors`` are the operators an
    upstream name may resolve through, ``routed`` the operators whose outputs are
    branch arms, and ``regions`` the operators producing a region's value. A
    dependency the spec reads, or one with a named input, is a required value; an
    identity read or an unread branch arm is a required route; any other is ordering
    only. An ancestor read by name but not depended on directly is a derived
    requirement of the same kind. An identity read of a definition input, a projected
    input, or a region's value is identityless.
    """
    reads = spec_reads(task)
    aliases = {
        name: index
        for index, dep in enumerate(dependencies)
        if (name := binding_name(dep)) is not None
    }
    sources = [value_op.get(dep.source, dep.source) for dep in dependencies]
    # A root task without dependencies may carry ``${...}`` text that is not a read.
    checked = bool(dependencies) or task.definition is not None
    unresolved: list[str] = list(reads.malformed) if checked else []

    def _resolve(name: str) -> set[int] | str | None:
        """The dependencies a name reads, or the ancestor it reads past them."""
        if (index := aliases.get(name)) is not None:
            return {index}
        if (op := names.get(name)) is not None and op in ancestors:
            direct = {
                i
                for i, (dep, source) in enumerate(zip(dependencies, sources))
                if source == op and dep.source != INGRESS
            }
            return direct or op
        if checked:
            unresolved.append(name)
        return None

    value_deps: set[int] = set()
    identity_deps: set[int] = set()
    derived: dict[str, DependencyUse] = {}
    for name in sorted(reads.values):
        match _resolve(name):
            case set() as indexes:
                value_deps |= indexes
            case str() as op:
                derived[op] = DependencyUse.VALUE_REQUIRED
    identityless: list[str] = []
    for name in sorted(reads.identities):
        if _identityless(name, aliases, dependencies, sources, names, regions):
            identityless.append(name)
        match _resolve(name):
            case set() as indexes:
                identity_deps |= indexes
            case str() as op:
                derived.setdefault(op, DependencyUse.ROUTE_REQUIRED)

    uses: list[DependencyUse] = []
    for index, (dep, source) in enumerate(zip(dependencies, sources)):
        if dep.input is not None or index in value_deps:
            uses.append(DependencyUse.VALUE_REQUIRED)
        elif index in identity_deps or (dep.port is not None and source in routed):
            uses.append(DependencyUse.ROUTE_REQUIRED)
        else:
            uses.append(DependencyUse.ORDER_ONLY)
    shadowing = [
        name
        for name, index in aliases.items()
        if (node := names.get(name)) is not None and node != sources[index]
    ]
    node_reads = [
        (name, node_op)
        for name in sorted(reads.values)
        if name not in aliases
        and (node_op := names.get(name)) is not None
        and node_op in ancestors
    ]
    stage_reads = [
        (name, stage_op)
        for name in sorted(reads.stages)
        if (stage_op := _stage_source(name, aliases, dependencies, sources, names))
        is not None
    ]
    return ReadClassification(
        uses=tuple(uses),
        derived=tuple(sorted(derived.items())),
        unresolved=tuple(dict.fromkeys(unresolved)),
        shadowing=tuple(sorted(shadowing)),
        identityless=tuple(identityless),
        node_reads=tuple(node_reads),
        stage_reads=tuple(stage_reads),
    )


def _stage_source(
    name: str,
    aliases: Mapping[str, int],
    dependencies: list[ParsedDependency],
    sources: list[str],
    names: Mapping[str, str],
) -> str | None:
    """The operator an SSH input stage reads; None for a definition input or a name
    that resolves to nothing."""
    if (index := aliases.get(name)) is not None:
        return None if dependencies[index].source == INGRESS else sources[index]
    return names.get(name)


def _identityless(
    name: str,
    aliases: Mapping[str, int],
    dependencies: list[ParsedDependency],
    sources: list[str],
    names: Mapping[str, str],
    regions: frozenset[str],
) -> bool:
    """Whether a name reads a value no task's result is: a definition input, a
    projected input, or a region's value."""
    if (index := aliases.get(name)) is not None:
        dep = dependencies[index]
        return dep.source == INGRESS or bool(dep.project) or sources[index] in regions
    return names.get(name) in regions
