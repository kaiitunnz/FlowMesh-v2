"""A sandbox task compiles to a session that owns its own private-state lineage.

The leaf normalizes its host profile into a sandbox service dependency, so admission
resolves it like any other resident consumer, and the engine answers that the session —
not only an agent — owns activation-private state.
"""

from server.orchestration.engine import OrchestrationEngine
from server.registries.workflow import PersistedV2Workflow
from server.task.parser import parse_workflow
from server.task.v2 import FrontendWorkflowSource, compile_workflow
from server.task.v2.representations.operators import LeafOperator, ServiceInterface
from shared.private_state import BundleProfile

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: sandbox-wf}
spec:
  stages:
    - name: build
      spec:
        taskType: sandbox
        sandbox: {profile: posix-default, isolation: tenant-a}
        commands:
          - argv: ["sh", "-c", "echo one > note.txt"]
          - argv: ["cat", "note.txt"]
"""


def _bundle() -> PersistedV2Workflow:
    parsed = parse_workflow(_WF, "native")
    source = FrontendWorkflowSource.capture(_WF, "native", name="wf")
    template, plan = compile_workflow("wfl-sandbox", parsed, source)
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def _session(template) -> LeafOperator:
    return next(
        op
        for op in template.operators
        if isinstance(op, LeafOperator) and op.service_dependency is not None
    )


def test_a_sandbox_leaf_depends_on_its_sandbox_host_family() -> None:
    dependency = _session(_bundle().template).service_dependency
    assert dependency is not None
    assert dependency.interface is ServiceInterface.SANDBOX
    assert dependency.service_ref == "posix-default"
    # The isolation domain keys the reuse domain, so two tenants never share a host.
    assert "iso=tenant-a" in dependency.service_family
    assert dependency.adapter is None


def test_a_sandbox_session_owns_a_sandbox_private_state_lineage() -> None:
    bundle = _bundle()
    engine = OrchestrationEngine.build("wfl-sandbox", "o", "g", bundle)
    leaf = _session(bundle.template)
    assert engine.sandbox_session_operator(leaf.operator_id) is not None
    assert (
        engine.private_state_profile(leaf.operator_id) is BundleProfile.SANDBOX_SESSION
    )


def test_a_sandbox_session_mints_its_invocation_before_any_attempt() -> None:
    bundle = _bundle()
    engine = OrchestrationEngine.build("wfl-sandbox", "o", "g", bundle)
    leaf = _session(bundle.template)
    invocation_id = engine.ensure_invocation(leaf.operator_id)
    assert invocation_id is not None
    # The identity is stable: admission links to the same one the first attempt reuses.
    assert engine.ensure_invocation(leaf.operator_id) == invocation_id


def test_a_session_grant_binds_the_sandbox_profile() -> None:
    bundle = _bundle()
    engine = OrchestrationEngine.build("wfl-sandbox", "o", "g", bundle)
    leaf = _session(bundle.template)
    granted = engine.grant_private_state(leaf.operator_id, "wrk-1", 1)
    assert granted is not None
    binding, attachment = granted
    assert binding.reference.profile is BundleProfile.SANDBOX_SESSION
    assert binding.generation == 0
    assert attachment.write_epoch == 1
