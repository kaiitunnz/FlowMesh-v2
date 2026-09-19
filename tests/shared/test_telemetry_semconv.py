"""The control-plane stage/window taxonomy: exact membership, no ``nested``."""

from shared.schemas.network import Transport
from shared.telemetry import semconv
from shared.telemetry.semconv import (
    ControlPlaneStage,
    ControlPlaneWindow,
    control_span_name,
    transport_span_name,
)

_EXPECTED_STAGES = {
    "compile_lower",
    "compile_assemble",
    "compile_episodes",
    "compile_finalize",
    "compile_validate",
    "engine_build",
    "ds_initial_advance",
    "ds_drive",
    "dispatch",
    "admission",
    "permit",
    "relay",
    "ledger_snapshot",
}

_EXPECTED_WINDOWS = {"submit", "queue", "post_start"}


def test_control_plane_stage_membership_is_exact() -> None:
    assert {stage.value for stage in ControlPlaneStage} == _EXPECTED_STAGES


def test_control_plane_window_membership_is_exact() -> None:
    assert {window.value for window in ControlPlaneWindow} == _EXPECTED_WINDOWS


def test_nested_is_not_a_window_or_a_stage() -> None:
    assert "nested" not in {s.value for s in ControlPlaneStage}
    assert "nested" not in {w.value for w in ControlPlaneWindow}


def test_control_span_name_uses_the_dotted_convention() -> None:
    assert control_span_name(ControlPlaneStage.DS_DRIVE) == "flowmesh.control.ds_drive"


def test_transport_span_name_uses_the_dotted_convention() -> None:
    assert (
        transport_span_name(Transport.WORKER_DIRECT)
        == "flowmesh.transport.worker_direct"
    )


def test_logical_and_physical_namespaces_are_disjoint_prefixes() -> None:
    logical_keys = [
        v
        for k, v in vars(semconv).items()
        if k.startswith("LOGICAL_") and isinstance(v, str)
    ]
    physical_keys = [
        v
        for k, v in vars(semconv).items()
        if k.startswith("PHYSICAL_") and isinstance(v, str)
    ]

    assert logical_keys, "expected at least one flowmesh.logical.* constant"
    assert physical_keys, "expected at least one flowmesh.physical.* constant"
    assert all(key.startswith(semconv.LOGICAL_ATTRIBUTE_PREFIX) for key in logical_keys)
    assert all(
        key.startswith(semconv.PHYSICAL_ATTRIBUTE_PREFIX) for key in physical_keys
    )
