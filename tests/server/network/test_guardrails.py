"""Guardrails: the route substrate carries no resident capacity concern.

The echo seam proves the route ladder and relay mechanics only. Its modules must not
import resident-capacity code or reference a claim/credit/route-authorization identifier
(prose in docstrings is fine; actual code coupling is not).
"""

import ast
from pathlib import Path

import server.network.deputy as deputy
import server.network.listeners as listeners
import server.network.resolver as resolver
import server.network.reverse_relay as reverse_relay
import server.network.service as service

_SUBSTRATE = (deputy, listeners, resolver, reverse_relay, service)

_FORBIDDEN_IDENTIFIERS = {
    "ServiceClaim",
    "RouteAuthorization",
    "AdmissionHandoff",
    "AdmissionController",
}


def _tree(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text())


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _identifiers(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_substrate_modules_do_not_import_resident() -> None:
    for module in _SUBSTRATE:
        assert not any(
            "resident" in name for name in _imported_modules(_tree(module))
        ), f"{module.__name__} imports resident code"


def test_substrate_modules_touch_no_claim_identifier() -> None:
    for module in _SUBSTRATE:
        used = _identifiers(_tree(module)) & _FORBIDDEN_IDENTIFIERS
        assert not used, f"{module.__name__} references {sorted(used)}"
