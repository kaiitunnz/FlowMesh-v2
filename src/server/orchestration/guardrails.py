"""Scope budget guardrails for structured dynamic regions.

Dynamic regions can nest scopes (call/spawn), iterate a loop, and fan out children
without a static bound. These conservative per-instance caps turn an unbounded region
into a durable ``scope_budget_exhausted`` failure rather than an unbounded materializing
engine. The engine takes a budget by injection, so configuration supplies production
limits and tests drive small caps.
"""

from dataclasses import dataclass, replace
from typing import Self

from ..config import OrchestrationConfig


@dataclass(frozen=True)
class ScopeBudget:
    max_scope_depth: int = 64  # nested call/spawn/recursion scopes
    max_loop_iterations: int = 1000  # logical times one loop instance may run
    max_activations: int = 10_000  # total dynamic activations per instance
    max_spawns_per_turn: int = 32  # spawn children admitted in one facade turn group
    max_spawns_per_region: int = 256  # spawn children admitted per child region

    def pinned(self, max_loop_iterations: int | None) -> Self:
        """This budget with the loop bound a running instance was built under."""
        if max_loop_iterations is None:
            return self
        return replace(self, max_loop_iterations=max_loop_iterations)

    @classmethod
    def from_config(cls, config: OrchestrationConfig) -> Self:
        overrides = {
            "max_scope_depth": config.max_scope_depth,
            "max_activations": config.max_activations,
            "max_spawns_per_turn": config.max_spawns_per_turn,
            "max_spawns_per_region": config.max_spawns_per_region,
        }
        return cls(**{k: v for k, v in overrides.items() if v is not None})
