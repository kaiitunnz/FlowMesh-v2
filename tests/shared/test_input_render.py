"""The first-turn input renderer emits members in canonical order, without truncating.

The renderer is the sole place an agent's dataflow inputs become text. It orders by the
manifest ordinal (never arrival), preserves every member, and renders deterministically
so a restart re-renders byte-identically. Empty bindings return the bare instruction.
"""

from shared.harness import InputBinding, InputBindingMember, render_input_envelope


def _member(op: str, value: str, ordinal: int, child: int | None = None):
    return InputBindingMember(
        source_operator_id=op,
        source_activation_id=f"act-{op}",
        child_index=child,
        outcome="success",
        value=value,
        content_digest=f"d-{op}",
        ordinal=ordinal,
    )


def test_no_bindings_returns_the_bare_instruction() -> None:
    assert render_input_envelope("do the thing", ()) == "do the thing"


def test_members_render_in_ordinal_order_not_argument_order() -> None:
    binding = InputBinding(
        port="reviews",
        provenance="join_aggregate",
        ordinal=0,
        members=(
            _member("r", "third", 2, child=2),
            _member("r", "first", 0, child=0),
            _member("r", "second", 1, child=1),
        ),
    )
    out = render_input_envelope("Merge the findings.", [binding])
    assert out.index("first") < out.index("second") < out.index("third")
    assert "r#0" in out and "r#1" in out and "r#2" in out


def test_render_is_deterministic_and_lossless() -> None:
    bindings = [
        InputBinding(
            port="facet",
            provenance="spawn_element",
            ordinal=0,
            members=(_member("planner", "retrieval methods", 0, child=0),),
        ),
        InputBinding(
            port="context",
            provenance="producer",
            ordinal=1,
            members=(_member("brief", "a" * 5000, 0),),
        ),
    ]
    first = render_input_envelope("Research the facet.", bindings)
    second = render_input_envelope("Research the facet.", list(reversed(bindings)))
    assert first == second  # ordering comes from the ordinal, not the list order
    assert "a" * 5000 in first  # a large value is delivered whole, never truncated
    assert first.count("--- input:") == 2
