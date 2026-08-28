"""Episode-cut lowering and its contract-equivalence to the transparent plan.

The episode-cut lowering annotates run-to-yield boundaries and fuses pure local leaf
chains. It must stay contract-equivalent to the transparent lowering: the same declared
outputs, effect visibility, and progress closure when the engine runs over either plan.
"""

from server.orchestration import OrchestrationEngine, PublicationOutcome
from server.task.parser import parse_workflow
from server.task.v2 import (
    FrontendWorkflowSource,
    PersistedV2Workflow,
    compile_workflow,
)
from server.task.v2.mode import LoweringStrategy
from server.task.v2.representations.plan import EpisodeBoundaryKind

_CHAIN = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: c
        dependsOn: [b]
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""

_MIXED = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: mixed}
spec:
  graph:
    nodes:
      - name: gen
        spec: {taskType: inference, inference: {model: m, prompt: p}}
      - name: eff
        dependsOn: [gen]
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
"""


def _plans(text: str) -> tuple[object, object, object]:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, transparent = compile_workflow("wfl-t", parsed, source)
    _, episode = compile_workflow(
        "wfl-t", parsed, source, strategy=LoweringStrategy.EPISODE_CUT
    )
    return template, transparent, episode


def _ref(template: object, node_name: str) -> str:
    """Resolve a graph node name to its operator id via the source map."""
    for entry in template.source_map:  # type: ignore[attr-defined]
        if entry.source_id == node_name:
            return entry.logical_ref
    raise KeyError(node_name)


def _covered_refs(plan: object) -> set[str]:
    refs: set[str] = set()
    for node in plan.nodes:  # type: ignore[attr-defined]
        if node.logical_ref:
            refs.add(node.logical_ref)
        if node.episode:
            refs.update(node.episode.fused_refs)
    return refs


def test_pure_leaf_chain_fuses_into_one_episode() -> None:
    template, transparent, episode = _plans(_CHAIN)
    assert len(transparent.nodes) == 3  # type: ignore[attr-defined]
    assert all(n.episode is None for n in transparent.nodes)  # type: ignore[attr-defined]
    # The three deterministic pure echoes fuse into a single run-to-yield episode.
    assert len(episode.nodes) == 1  # type: ignore[attr-defined]
    (node,) = episode.nodes  # type: ignore[attr-defined]
    assert node.episode is not None
    assert node.episode.boundary is EpisodeBoundaryKind.TASK
    a, b, c = (_ref(template, name) for name in ("a", "b", "c"))
    assert node.logical_ref == a and set(node.episode.fused_refs) == {b, c}


def test_boundaries_reflect_service_and_effect_cuts() -> None:
    template, _, episode = _plans(_MIXED)
    by_ref = {
        n.logical_ref: n.episode.boundary  # type: ignore[attr-defined]
        for n in episode.nodes  # type: ignore[attr-defined]
    }
    # A sampled model call cuts at a service-issue boundary; an external call at effect.
    assert by_ref[_ref(template, "gen")] is EpisodeBoundaryKind.SERVICE_ISSUE
    assert by_ref[_ref(template, "eff")] is EpisodeBoundaryKind.EFFECT
    # Neither fuses: a service issue and an effect are episode edges, not local ops.
    assert all(not n.episode.fused_refs for n in episode.nodes)  # type: ignore[attr-defined]


def test_episode_cut_preserves_source_map_coverage() -> None:
    template, transparent, episode = _plans(_CHAIN)
    ids = template.operator_ids  # type: ignore[attr-defined]
    assert _covered_refs(transparent) == ids
    # Fusion folds operators into an episode; none is dropped from the source map.
    assert _covered_refs(episode) == ids


def _run(bundle: PersistedV2Workflow) -> dict[str, PublicationOutcome]:
    eng = OrchestrationEngine.build("wfl-x", "o", "g", bundle)
    adv = eng.initial_advance()
    pending = list(adv.ready)
    while pending:
        task = pending.pop(0)
        adv = eng.on_succeeded(task)
        pending.extend(adv.ready)
    outcomes: dict[str, PublicationOutcome] = {}
    for decl in bundle.template.result_declarations:
        pub = eng.resolve_output(decl.output_id)
        if pub is not None:
            outcomes[decl.output_id] = pub.outcome
    return outcomes


def test_contract_equivalence_across_cuts() -> None:
    parsed = parse_workflow(_CHAIN, "native")
    source = FrontendWorkflowSource.capture(_CHAIN, "native", name="wf")
    t_template, t_plan = compile_workflow("wfl-t", parsed, source)
    e_template, e_plan = compile_workflow(
        "wfl-t", parsed, source, strategy=LoweringStrategy.EPISODE_CUT
    )
    transparent = PersistedV2Workflow(source=source, template=t_template, plan=t_plan)
    episode = PersistedV2Workflow(source=source, template=e_template, plan=e_plan)
    # Running the engine over either lowering resolves the same declared outputs.
    episode_outcomes = _run(episode)
    assert _run(transparent) == episode_outcomes
    assert len(episode_outcomes) == 3
    assert set(episode_outcomes.values()) == {PublicationOutcome.SUCCESS}
