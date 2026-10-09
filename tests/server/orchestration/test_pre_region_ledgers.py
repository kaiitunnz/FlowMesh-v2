"""A ledger stored before control states existed resumes its fired root controls."""

import json
from pathlib import Path
from typing import Any

import pytest

from server.orchestration import OrchestrationEngine
from server.orchestration.engine.advance import Advance
from server.orchestration.state import (
    ControlStatus,
    LedgerSnapshot,
    PublicationOutcome,
    WorkItemStatus,
)
from server.task.v2 import PersistedV2Workflow

from .control_flow import ECHO, compile_text, workflow

# Bundles and ledgers a server stored before control states existed: a fired merge, a
# spawn with one child outstanding, and a released join, each beside an unsettled
# ``slow`` task its consumer also waits on.
_LEDGERS = json.loads(
    (Path(__file__).parent / "fixtures" / "pre_region_ledgers.json").read_text()
)


def _restore(name: str) -> tuple[OrchestrationEngine, dict[str, Any]]:
    stored = _LEDGERS[name]
    engine = OrchestrationEngine(
        LedgerSnapshot.model_validate(stored["snapshot"]),
        PersistedV2Workflow.model_validate(stored["bundle"]),
    )
    return engine, stored


def _settle(engine: OrchestrationEngine, task_id: str) -> Advance:
    engine.on_dispatched(task_id, "w1")
    return engine.on_succeeded(task_id)


def test_a_merge_that_fired_before_the_upgrade_releases_its_consumer() -> None:
    engine, stored = _restore("merge_fired")
    ops = stored["ops"]
    merge = engine.control_state(ops["m"])
    assert merge is not None and merge.status is ControlStatus.LIVE
    members = merge.outputs["out"].members
    assert [m.outcome for m in members] == [PublicationOutcome.SUCCESS] * 2

    assert ops["after"] in _settle(engine, ops["slow"]).ready


def test_a_spawn_open_before_the_upgrade_releases_its_join_when_children_settle() -> (
    None
):
    engine, stored = _restore("spawn_open")
    ops = stored["ops"]
    spawn = engine.control_state(ops["fan"])
    assert spawn is not None and spawn.status is ControlStatus.LIVE
    assert engine.control_state(ops["collect"]) is None

    outstanding = [
        child
        for child in stored["children"]
        if engine.work_item(child).status is WorkItemStatus.READY  # type: ignore[union-attr]
    ]
    assert len(outstanding) == 1
    _settle(engine, outstanding[0])
    collect = engine.control_state(ops["collect"])
    assert collect is not None and collect.status is ControlStatus.LIVE
    assert len(collect.outputs["out"].members) == 2

    assert ops["after"] in _settle(engine, ops["slow"]).ready


def test_a_join_released_before_the_upgrade_carries_its_aggregate() -> None:
    engine, stored = _restore("join_released")
    ops = stored["ops"]
    collect = engine.control_state(ops["collect"])
    assert collect is not None and collect.status is ControlStatus.LIVE
    members = collect.outputs["out"].members
    assert [m.outcome for m in members] == [PublicationOutcome.SUCCESS] * 2

    assert ops["after"] in _settle(engine, ops["slow"]).ready


@pytest.mark.parametrize("name", sorted(_LEDGERS))
def test_a_restored_pre_upgrade_ledger_round_trips(name: str) -> None:
    engine, stored = _restore(name)
    again = OrchestrationEngine(
        engine.to_snapshot(), PersistedV2Workflow.model_validate(stored["bundle"])
    )
    assert again.to_snapshot().control_states == engine.to_snapshot().control_states


_BRANCH = f"""
      - name: decide
        spec: {ECHO}
      - name: route
        dependsOn: [{{node: decide, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: hit}}, {{name: miss}}]
          selection: {{input: input}}
      - name: after
        dependsOn: [{{node: route, port: hit}}]
        spec: {ECHO}
"""

_LOOP = f"""
      - name: seed
        spec: {ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
      - name: after
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {ECHO}
"""

_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {ECHO}
        edges:
          - from: {{node: step}}
            to: {{node: $egress, port: state}}
"""


def _stored_without(
    text: str, region: str, field: str
) -> tuple[OrchestrationEngine, dict[str, str]]:
    """An engine over a bundle whose ``region`` was stored without ``field``."""
    bundle = compile_text(text)
    names = {e.source_id: e.logical_ref for e in bundle.template.source_map}
    operators = [
        op.model_copy(update={field: None}) if op.operator_id == names[region] else op
        for op in bundle.template.operators
    ]
    stored = bundle.model_copy(
        update={"template": bundle.template.model_copy(update={"operators": operators})}
    )
    return OrchestrationEngine.build("wfl-cf", "owner", "org", stored), names


@pytest.mark.parametrize(
    ("text", "region", "field", "first"),
    [
        (workflow(_BRANCH), "route", "rule", "decide"),
        (workflow(_LOOP, _BODY), "refine", "body_ref", "seed"),
    ],
)
def test_a_stored_branch_or_loop_without_its_contract_fails_closed(
    text: str, region: str, field: str, first: str
) -> None:
    engine, names = _stored_without(text, region, field)
    assert engine.initial_advance().ready == [names[first]]
    advance = _settle(engine, names[first])
    state = engine.control_state(names[region])
    assert state is not None and state.status is ControlStatus.FAILED
    assert state.reason is not None
    assert state.reason.startswith("LegacyControlRegionUnsupported")
    assert names["after"] in advance.failed
    assert engine.pending_branch_reads() == []
