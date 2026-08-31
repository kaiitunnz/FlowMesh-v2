"""The renderer for an agent's first-turn dataflow inputs.

A harness adapter renders the resolved input bindings into a delimited envelope beside
the agent's static instruction. The renderer is the sole place text is composed; it
orders members in their canonical ordinal order and may add only presentation labels. It
never chooses membership or ordering (those come from the durable manifest) and never
truncates (an oversized input is a compile-time bounds decision, not a render-time one).
It is a pure deterministic function of the manifest, so a restart re-renders the same.
"""

from collections.abc import Sequence

from .adapter import InputBinding

__all__ = ["render_input_envelope"]


def render_input_envelope(instruction: str, bindings: Sequence[InputBinding]) -> str:
    """The static instruction followed by a delimited envelope of the input bindings."""
    if not bindings:
        return instruction
    lines: list[str] = [instruction, "", "=== INPUTS ==="]
    for binding in sorted(bindings, key=lambda b: (b.ordinal, b.port)):
        lines.append(f"--- input: {binding.port} ({binding.provenance}) ---")
        for member in sorted(binding.members, key=lambda m: m.ordinal):
            label = _member_label(member.source_operator_id, member.child_index)
            lines.append(f"[{label}] (outcome={member.outcome})")
            lines.append(member.value if member.value is not None else "(no value)")
    lines.append("=== END INPUTS ===")
    return "\n".join(lines)


def _member_label(source_operator_id: str, child_index: int | None) -> str:
    if child_index is None:
        return source_operator_id
    return f"{source_operator_id}#{child_index}"
