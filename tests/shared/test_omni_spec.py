"""An omni spec carrying the removed inline stage_configs is not dispatchable."""

import pytest

from server.task.parser import parse_workflow
from shared.tasks.envelope import TaskEnvelopeTemplate
from shared.tasks.specs.omni import (
    OmniText2GeneralSpecStrict,
    OmniText2GeneralSpecTemplate,
)
from shared.tasks.task_type import TaskType

_SPEC = {
    "taskType": TaskType.OMNI_TEXT2GENERAL,
    "model": {"source": {"identifier": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}},
    "data": {"type": "list", "items": ["hi"]},
    "omni": {"stage_configs": {"stage_args": []}},
}
_WORKFLOW = """
apiVersion: flowmesh/v1
kind: Omni
metadata: {name: omni}
spec:
  taskType: omni_text2general
  model: {source: {identifier: Qwen/Qwen3-Omni-30B-A3B-Instruct}}
  data: {type: list, items: [hi]}
  omni: {stage_configs: {stage_args: []}}
"""


@pytest.mark.parametrize(
    "spec_type", [OmniText2GeneralSpecStrict, OmniText2GeneralSpecTemplate]
)
def test_stage_configs_is_not_dispatchable(
    spec_type: type[OmniText2GeneralSpecStrict] | type[OmniText2GeneralSpecTemplate],
) -> None:
    spec = spec_type.model_validate(_SPEC)

    with pytest.raises(ValueError, match="omni.stage_configs"):
        spec.validate_dispatchable()


def test_a_submission_carrying_stage_configs_is_rejected() -> None:
    with pytest.raises(ValueError, match="omni.stage_configs is not supported"):
        parse_workflow(_WORKFLOW, "native")


def test_a_stored_task_carrying_stage_configs_still_loads() -> None:
    envelope = TaskEnvelopeTemplate.model_validate(
        {"apiVersion": "flowmesh/v1", "kind": "Omni", "metadata": {}, "spec": _SPEC}
    )

    assert envelope.spec.taskType == TaskType.OMNI_TEXT2GENERAL


def test_stage_overrides_stay_dispatchable() -> None:
    spec = OmniText2GeneralSpecStrict.model_validate(
        _SPEC | {"omni": {"stage_overrides": {"0": {"devices": "0"}}}}
    )

    spec.validate_dispatchable()
