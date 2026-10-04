"""An ``Omni`` instance is reused only by a task with the same pipeline layout."""

from worker.executors.omni_executor_base import OmniExecutorBase


def test_a_different_stage_layout_gets_its_own_omni() -> None:
    base = {"stage_overrides": {"0": {"devices": "0"}}}
    other = {"stage_overrides": {"0": {"devices": "1"}}}
    deployed = {"deploy_config": {"stages": [{"stage_id": 0}]}}

    specs = {
        OmniExecutorBase._build_omni_spec("org/omni", cfg)
        for cfg in ({}, base, other, deployed, {"deploy_config": "/opt/deploy.yaml"})
    }

    assert len(specs) == 5


def test_the_same_layout_shares_one_omni() -> None:
    assert OmniExecutorBase._build_omni_spec(
        "org/omni", {"stage_overrides": {"0": {"devices": "0"}, "1": {"x": 1}}}
    ) == OmniExecutorBase._build_omni_spec(
        "org/omni", {"stage_overrides": {"1": {"x": 1}, "0": {"devices": "0"}}}
    )
