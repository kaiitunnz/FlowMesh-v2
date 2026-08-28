"""Scope budget guardrails for structured dynamic regions.

Dynamic regions can nest scopes (call/spawn), iterate a loop, and fan out children
without a static bound. These conservative per-instance caps turn an unbounded region
into a durable ``scope_budget_exhausted`` failure rather than an unbounded materializing
engine. The engine takes a budget by injection, so configuration supplies production
limits and tests drive small caps.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeBudget:
    max_scope_depth: int = 64  # nested call/spawn/recursion scopes
    max_loop_iterations: int = 1000  # loop_time per LoopContext activation
    max_activations: int = 10_000  # total dynamic activations per instance
