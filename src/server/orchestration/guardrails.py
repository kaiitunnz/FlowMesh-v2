"""Scope budget guardrails for structured dynamic regions.

Dynamic regions can nest scopes (call/spawn), iterate a loop, and fan out children
without a static bound. These conservative per-instance caps turn an unbounded region
into a durable ``scope_budget_exhausted`` failure rather than an unbounded materializing
ledger. They are env-configurable and injectable so tests can drive small budgets.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeBudget:
    max_scope_depth: int = 64  # nested call/spawn/recursion scopes
    max_loop_iterations: int = 1000  # loop_time per LoopContext activation
    max_activations: int = 10_000  # total dynamic activations per instance

    @classmethod
    def from_env(cls) -> "ScopeBudget":
        return cls(
            max_scope_depth=_int_env(
                "FLOWMESH_V2_MAX_SCOPE_DEPTH", cls.max_scope_depth
            ),
            max_loop_iterations=_int_env(
                "FLOWMESH_V2_MAX_LOOP_ITERATIONS", cls.max_loop_iterations
            ),
            max_activations=_int_env(
                "FLOWMESH_V2_MAX_ACTIVATIONS", cls.max_activations
            ),
        )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default
