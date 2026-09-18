"""Drift guard: the Collector's allow-list vs. semconv.py's own attribute keys.

The Collector config is static YAML consumed by a Go binary, so it cannot import
``shared.telemetry.semconv`` the way Python code does. This test is the substitute: it
enumerates every public ``flowmesh.``-valued string constant in semconv.py *by value*,
classifies it as an attribute key or a span name, and asserts every attribute key is
covered by the YAML's ``keep_matching_keys`` pattern -- so a new key added under any
naming convention fails loudly here instead of being silently dropped by the processor.

Enumerating by value rather than by a naming convention (e.g. only names ending in
``_ATTRIBUTE_PREFIX``) is deliberate: an earlier allow-list covered every
``*_ATTRIBUTE_PREFIX`` constant and still silently dropped ``flowmesh.node_id`` /
``flowmesh.worker_id`` / ``flowmesh.role`` (``RESOURCE_*`` constants, no ``.`` suffix,
attached as plain datapoint attributes by the GPU sampler) -- a name-based enumeration
that only checks the keys it already expects can never catch that class of gap.
"""

import re

import yaml
from flowmesh_cli.core.assets import asset_path

import shared.telemetry.semconv as semconv

_COLLECTOR_CONFIG_PATH = asset_path(
    "flowmesh_cli_stack.assets", "otel-collector-config.yaml"
)


def _span_name_constant(name: str) -> bool:
    """True for a span *name* constant/prefix, never an attribute key."""
    return name.startswith("SPAN_") or name.endswith("_SPAN_PREFIX")


def _public_flowmesh_attribute_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for name, value in vars(semconv).items():
        if name.startswith("_") or not isinstance(value, str):
            continue
        if not value.startswith("flowmesh."):
            continue
        if _span_name_constant(name):
            continue
        keys.append((name, value))
    return keys


def _allowlist_patterns() -> list[str]:
    doc = yaml.safe_load(_COLLECTOR_CONFIG_PATH.read_text())
    processor = doc["processors"]["transform/allowlist"]
    patterns = []
    for statement_group in (
        processor["trace_statements"],
        processor["metric_statements"],
    ):
        for entry in statement_group:
            for statement in entry["statements"]:
                match = re.search(
                    r'keep_matching_keys\([^,]+,\s*"((?:[^"\\]|\\.)*)"\)', statement
                )
                # The allowlist also carries statements that clear a field outright
                # rather than filter its keys; only key filters have a pattern to drift.
                if match is None:
                    continue
                patterns.append(match.group(1).replace("\\\\", "\\"))
    assert patterns, "the allowlist should declare at least one key filter"
    return patterns


def test_every_semconv_attribute_key_is_covered_by_every_allowlist_pattern() -> None:
    attribute_keys = _public_flowmesh_attribute_keys()
    assert (
        attribute_keys
    ), "sanity: semconv.py should declare at least one attribute key"

    for pattern in _allowlist_patterns():
        compiled = re.compile(pattern)
        missing = [
            (name, value)
            for name, value in attribute_keys
            if not compiled.search(value)
        ]
        assert not missing, (
            f"collector allow-list pattern {pattern!r} drops semconv "
            f"attribute keys: {missing}"
        )


def test_span_name_constants_are_excluded_deliberately_not_incidentally() -> None:
    """Guards the classifier itself: at least one real span-name constant must exist and
    must be excluded, so the exclusion path is exercised rather than vacuously true."""
    span_name_constants = [
        name
        for name, value in vars(semconv).items()
        if not name.startswith("_")
        and isinstance(value, str)
        and value.startswith("flowmesh.")
        and _span_name_constant(name)
    ]
    assert span_name_constants, "expected at least one SPAN_*/*_SPAN_PREFIX constant"
    attribute_key_names = {name for name, _ in _public_flowmesh_attribute_keys()}
    assert attribute_key_names.isdisjoint(span_name_constants)
