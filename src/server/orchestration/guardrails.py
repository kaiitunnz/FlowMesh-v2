"""Scope budget guardrails for structured dynamic regions.

Dynamic regions can nest scopes (call/spawn), iterate a loop, and fan out children
without a static bound. These conservative per-instance caps turn an unbounded region
into a durable ``scope_budget_exhausted`` failure rather than an unbounded materializing
engine. The engine takes a budget by injection, so configuration supplies production
limits and tests drive small caps.
"""

from dataclasses import dataclass
from typing import Self

from ..config import OrchestrationConfig


@dataclass(frozen=True)
class ScopeBudget:
    max_scope_depth: int = 64  # nested call/spawn/recursion scopes
    max_loop_iterations: int = 1000  # loop_time per LoopContext activation
    max_activations: int = 10_000  # total dynamic activations per instance

    @classmethod
    def from_config(cls, config: OrchestrationConfig) -> Self:
        overrides = {
            "max_scope_depth": config.max_scope_depth,
            "max_loop_iterations": config.max_loop_iterations,
            "max_activations": config.max_activations,
        }
        return cls(**{k: v for k, v in overrides.items() if v is not None})
