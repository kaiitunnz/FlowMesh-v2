"""Inline task-spec credentials are vaulted at submission and restored at dispatch."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from server.task.v2 import PersistedV2Workflow
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.representations.operators import LeafOperator
from shared.tasks.components.output import OutputDestinationHTTP
from shared.tasks.specs import ApiSpecStrict, ApiSpecTemplate
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.redact import REDACTED
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry

_AUTH = "Bearer sk-user-inline-SECRET"
_QUERY_KEY = "q-user-inline-SECRET"
_OUTPUT_KEY = "out-user-inline-SECRET"
_HF = "hf-user-inline-SECRET"
_SECRETS = (_AUTH, _QUERY_KEY, _OUTPUT_KEY, _HF)


def _api_workflow(api_version: str, auth: str = _AUTH) -> str:
    return f"""
apiVersion: {api_version}
kind: Workflow
metadata: {{name: creds}}
spec:
  graph:
    nodes:
      - name: call
        spec:
          taskType: api
          api:
            url: "https://api.example/v1/chat?api_key={_QUERY_KEY}"
            headers: {{Authorization: "{auth}", Accept: application/json}}
            json: {{model: m}}
          output:
            destination:
              type: http
              url: "https://sink.example/"
              headers: {{X-Api-Key: "{_OUTPUT_KEY}"}}
      - name: shell
        spec:
          taskType: ssh
          command: ["true"]
          env: {{HF_TOKEN: "{_HF}", MODE: plain}}
"""


def _runtime(
    registry: FakeRegistry | None = None, vault: InMemoryCredentialVault | None = None
) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry or FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("task-credentials"),
        secret_vault=vault or InMemoryCredentialVault(),
    )


def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


def _dispatch(runtime: TaskRuntime, task_id: str) -> tuple[mock.Mock, Any]:
    worker = Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    disp = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("task-credentials-dispatch"),
    )
    disp.dispatch_once(task_id)
    return registry, disp


def _api(task: Any) -> dict[str, Any]:
    spec = task.spec
    assert isinstance(spec, (ApiSpecStrict, ApiSpecTemplate)) and spec.api is not None
    return spec.api


def _merge_key(runtime: TaskRuntime, task_id: str) -> str | None:
    record = runtime.get_record(task_id)
    assert record is not None
    return record.merge_key


def _no_secret(text: str) -> bool:
    return not any(secret in text for secret in _SECRETS)


@pytest.mark.parametrize("api_version", ["flowmesh/v1", "flowmesh/v2"])
def test_no_persisted_or_served_surface_holds_an_inline_credential(api_version):
    registry = FakeRegistry()
    runtime = _runtime(registry)
    _, ids = _register(runtime, _api_workflow(api_version))

    blobs = [
        *registry.task_blobs.values(),
        *registry.v2_blobs.values(),
        *registry.ledger_blobs.values(),
    ]
    assert blobs and all(_no_secret(blob) for blob in blobs)
    for task_id in ids.values():
        info = runtime.describe_task(task_id)
        assert info is not None
        assert _no_secret(info.model_dump_json())
        assert "credential_refs" not in info.model_dump()

    call = runtime.get_record(ids["call"])
    assert call is not None
    assert _api(call.task)["headers"] == {
        "Authorization": REDACTED,
        "Accept": "application/json",
    }
    assert _api(call.task)["url"] == REDACTED


@pytest.mark.parametrize("api_version", ["flowmesh/v1", "flowmesh/v2"])
def test_a_dispatch_carries_the_task_its_own_credentials(api_version):
    runtime = _runtime()
    _, ids = _register(runtime, _api_workflow(api_version))

    registry, disp = _dispatch(runtime, ids["call"])

    assert disp.failed == []
    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    api = _api(message.task)
    assert api["headers"]["Authorization"] == _AUTH
    assert api["url"] == f"https://api.example/v1/chat?api_key={_QUERY_KEY}"
    output = message.task.spec.output
    assert output is not None
    assert isinstance(output.destination, OutputDestinationHTTP)
    assert output.destination.headers == {"X-Api-Key": _OUTPUT_KEY}
    # The record the dispatch rendered from keeps only the refs.
    record = runtime.get_record(ids["call"])
    assert record is not None and _no_secret(record.model_dump_json())


def test_a_credential_no_longer_retained_fails_its_task_and_substitutes_nothing():
    vault = InMemoryCredentialVault()
    runtime = _runtime(vault=vault)
    workflow_id, ids = _register(runtime, _api_workflow("flowmesh/v1"))
    vault.purge(workflow_id)

    registry, disp = _dispatch(runtime, ids["call"])

    registry.publish_task.assert_not_called()
    assert len(disp.failed) == 1
    task_id, message, kwargs = disp.failed[0]
    assert task_id == ids["call"]
    assert message == "credential_not_retained"
    assert kwargs["payload"]["reason"] == "credential_not_retained"


def test_a_rendering_error_quotes_no_restored_credential():
    runtime = _runtime()
    _, ids = _register(runtime, _api_workflow("flowmesh/v1"))
    boom = RuntimeError(f"cannot render headers {{'Authorization': '{_AUTH}'}}")
    with mock.patch.object(
        CapturingDispatcher, "_resolve_stage_references", side_effect=boom
    ):
        _, disp = _dispatch(runtime, ids["call"])

    assert len(disp.failed) == 1
    _, message, kwargs = disp.failed[0]
    assert _no_secret(message) and _no_secret(kwargs["payload"]["error"])
    assert REDACTED in message


def _inference_workflow(tokens: dict[str, str], data_keys: dict[str, str]) -> str:
    nodes = "".join(
        f"""
      - name: {name}
        spec:
          taskType: inference
          model: {{source: {{identifier: m}}, config: {{token: "{token}"}}}}
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [{name}], api_key: "{data_keys[name]}"}}"""
        for name, token in tokens.items()
    )
    return f"""
apiVersion: flowmesh/v1
kind: Workflow
metadata: {{name: merge-creds}}
spec:
  graph:
    nodes:{nodes}
"""


_T1, _T2 = "hf-token-ONE-secret", "hf-token-TWO-secret"
_K1, _K2 = "data-key-ONE-secret", "data-key-TWO-secret"


def test_merge_keys_tell_credentials_apart_and_carry_none():
    runtime = _runtime()
    payload = _inference_workflow(
        {"a": _T1, "b": _T1, "c": _T2}, {"a": _K1, "b": _K2, "c": _K1}
    )
    _, ids = _register(runtime, payload)
    _, again = _register(runtime, payload)

    keys = {name: _merge_key(runtime, task_id) for name, task_id in ids.items()}
    assert keys["a"] == keys["b"] != keys["c"]
    for key in keys.values():
        assert key is not None
        assert all(secret not in key for secret in (_T1, _T2, _K1, _K2))
    # A credential names a merge key only within its own workflow.
    assert _merge_key(runtime, again["a"]) != keys["a"]


def test_a_merged_dispatch_carries_each_task_its_own_credentials():
    runtime = _runtime()
    payload = _inference_workflow({"a": _T1, "b": _T1}, {"a": _K1, "b": _K2})
    _, ids = _register(runtime, payload)

    registry, disp = _dispatch(runtime, ids["a"])

    assert disp.failed == []
    message = registry.publish_task.call_args[0][1]
    assert message.task.spec.model.config == {"token": _T1}
    assert message.task.spec.data["api_key"] == _K1
    [child] = message.merged_children
    assert child.task_id == ids["b"]
    assert child.spec.model.config == {"token": _T1}
    assert child.spec.data["api_key"] == _K2


def test_a_merged_child_whose_credential_is_gone_leaves_the_merge():
    vault = InMemoryCredentialVault()
    runtime = _runtime(vault=vault)
    payload = _inference_workflow({"a": _T1, "b": _T1}, {"a": _K1, "b": _K2})
    workflow_id, ids = _register(runtime, payload)
    child = runtime.get_record(ids["b"])
    assert child is not None and child.credential_refs is not None
    gone = child.credential_refs["/data/api_key"]
    del vault.redis.hashes[f"workflow:{workflow_id}:model_secret"][gone]

    registry, disp = _dispatch(runtime, ids["a"])

    assert disp.failed == []
    message = registry.publish_task.call_args[0][1]
    assert message.task_id == ids["a"] and not message.merged_children
    released = runtime.get_record(ids["b"])
    assert released is not None and released.merge_key is None


def test_a_task_whose_credential_renders_from_a_stage_never_merges():
    runtime = _runtime()
    payload = _inference_workflow({"a": _T1, "b": "${a.token}"}, {"a": _K1, "b": _K2})
    _, ids = _register(runtime, payload)

    assert _merge_key(runtime, ids["a"]) is not None
    assert _merge_key(runtime, ids["b"]) is None


def test_a_restart_dispatches_a_task_with_its_credentials():
    registry = FakeRegistry()
    vault = InMemoryCredentialVault()
    _, ids = _register(_runtime(registry, vault), _api_workflow("flowmesh/v1"))

    restarted = _runtime(registry, vault)
    asyncio.run(restarted.rehydrate())
    publisher, disp = _dispatch(restarted, ids["call"])

    assert disp.failed == []
    message = publisher.publish_task.call_args[0][1]
    assert message.task.spec.api["headers"]["Authorization"] == _AUTH


_PRE_VAULT = Path(__file__).parent / "fixtures" / "pre_vault_records.json"
_LEGACY = (
    "legacy-inline-credential",
    "legacy-model-credential",
    "legacy-settled-credential",
)


def _pre_vault_registry() -> tuple[FakeRegistry, dict[str, Any]]:
    """Records a build that kept inline credentials in plaintext stored."""
    stored = json.loads(_PRE_VAULT.read_text())
    registry = FakeRegistry()
    registry.task_blobs.update(stored["task_blobs"])
    registry.sched.update(stored["sched"])
    registry.workflow_task_ids.update(stored["workflow_task_ids"])
    return registry, stored


def _task_named(runtime: TaskRuntime, workflow_id: str, name: str) -> str:
    return next(
        record.task_id
        for record in runtime.list_tasks()
        if record.workflow_id == workflow_id and record.graph_node_name == name
    )


def test_a_restart_vaults_credentials_stored_before_they_were_vaulted():
    registry, stored = _pre_vault_registry()
    assert any(secret in "".join(registry.task_blobs.values()) for secret in _LEGACY)
    vault = InMemoryCredentialVault()
    runtime = _runtime(registry, vault)

    asyncio.run(runtime.rehydrate())

    blobs = "".join(registry.task_blobs.values())
    assert not any(secret in blobs for secret in _LEGACY)
    for info in runtime.list_tasks():
        assert not any(secret in info.model_dump_json() for secret in _LEGACY)
    infer = runtime.get_record(_task_named(runtime, stored["live"], "infer"))
    assert infer is not None and infer.merge_key is not None
    # The settled workflow's credentials are masked and nothing of it is vaulted.
    assert set(vault.redis.hashes) == {f"workflow:{stored['live']}:model_secret"}

    call = _task_named(runtime, stored["live"], "call")
    publisher, disp = _dispatch(runtime, call)
    assert disp.failed == []
    message = publisher.publish_task.call_args[0][1]
    assert message.task.spec.api["headers"]["Authorization"] == (
        "Bearer legacy-inline-credential"
    )


def test_a_second_restart_keeps_the_credentials_the_first_vaulted():
    registry, stored = _pre_vault_registry()
    vault = InMemoryCredentialVault()
    asyncio.run(_runtime(registry, vault).rehydrate())
    vaulted = json.dumps(vault.redis.hashes, sort_keys=True)

    restarted = _runtime(registry, vault)
    asyncio.run(restarted.rehydrate())

    assert json.dumps(vault.redis.hashes, sort_keys=True) == vaulted
    call = _task_named(restarted, stored["live"], "call")
    publisher, _ = _dispatch(restarted, call)
    message = publisher.publish_task.call_args[0][1]
    assert message.task.spec.api["headers"]["Authorization"] == (
        "Bearer legacy-inline-credential"
    )


def test_a_record_that_cannot_be_vaulted_loads_and_runs_as_stored(monkeypatch):
    registry, stored = _pre_vault_registry()
    runtime = _runtime(registry)

    def refuse(*args: Any) -> Any:
        raise RuntimeError("unreadable spec")

    monkeypatch.setattr("server.task.runtime.take_spec_credentials", refuse)
    asyncio.run(runtime.rehydrate())

    call = _task_named(runtime, stored["live"], "call")
    record = runtime.get_record(call)
    assert record is not None and record.credential_refs is None
    publisher, disp = _dispatch(runtime, call)
    assert disp.failed == []
    message = publisher.publish_task.call_args[0][1]
    assert message.task.spec.api["headers"]["Authorization"] == (
        "Bearer legacy-inline-credential"
    )


_PRESIGNED = (
    "https://bucket.s3.amazonaws.com/lora.tar?X-Amz-Credential=AKIA-PRESIGNED-SECRET"
    "&X-Amz-Signature=sig"
)


def _adapter_leaf(adapter_url: str, service: str = "") -> str:
    service_line = f"\n          service: {service}" if service else ""
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: adapter}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {{identifier: Qwen/Qwen3-4B}}
            adapters: [{{type: lora, name: my-lora, url: "{adapter_url}"}}]
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [hi]}}{service_line}
"""


def _plan_nodes(registry: FakeRegistry, workflow_id: str) -> list[Any]:
    bundle = PersistedV2Workflow.model_validate_json(registry.v2_blobs[workflow_id])
    return list(bundle.plan.nodes)


def test_a_leaf_whose_adapter_url_carries_a_credential_runs_self_contained():
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, _ = _register(runtime, _adapter_leaf(_PRESIGNED))

    [node] = _plan_nodes(registry, workflow_id)
    assert node.embodiment_menu is None
    assert node.service_family_requirement is None
    assert node.residency_intent is None
    persisted = "".join([*registry.v2_blobs.values(), *registry.ledger_blobs.values()])
    assert "PRESIGNED-SECRET" not in persisted
    report = runtime.inspect_v2(_adapter_leaf(_PRESIGNED))
    assert report is not None
    assert "PRESIGNED-SECRET" not in report.model_dump_json()


@pytest.mark.parametrize("service", ["{mode: resident}", "{isolation: team}"])
def test_resident_serving_of_a_credentialed_adapter_is_refused(service):
    payload = _adapter_leaf(_PRESIGNED, service)
    runtime = _runtime()

    with pytest.raises(CompileError, match="adapter"):
        _register(runtime, payload)
    with pytest.raises(CompileError, match="adapter"):
        runtime.inspect_v2(payload)


def test_resident_serving_of_a_plain_adapter_url_keeps_its_source():
    registry = FakeRegistry()
    url = "https://example.com/lora.tar"
    workflow_id, _ = _register(
        _runtime(registry), _adapter_leaf(url, "{mode: resident}")
    )

    [node] = _plan_nodes(registry, workflow_id)
    assert node.service_family_requirement is not None
    bundle = PersistedV2Workflow.model_validate_json(registry.v2_blobs[workflow_id])
    [leaf] = bundle.template.operators
    assert isinstance(leaf, LeafOperator) and leaf.service_dependency is not None
    assert leaf.service_dependency.adapter_source == url
