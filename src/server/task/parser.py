import copy
import json
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from shared.tasks import TaskEnvelopeTemplate, TaskType
from shared.tasks.components import TaskAnnotations
from shared.utils import new_task_id, parse_bool_env
from shared.utils.json import safe_get

from .n8n_parser import translate_n8n_workflow

_ALLOWED_TASK_TYPES = ", ".join(member.value for member in TaskType)
_ENABLE_SERVER_SSH_PROXY = parse_bool_env("ENABLE_SERVER_SSH_PROXY", True)
_ENABLE_SERVER_SERVE_PROXY = parse_bool_env("ENABLE_SERVER_SERVE_PROXY", True)
_ENABLE_SERVER_PORT_FORWARD = parse_bool_env("ENABLE_SERVER_PORT_FORWARD", True)


class _WorkflowMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    owner: str | None = None
    annotations: TaskAnnotations | None = None


class _WorkflowRootInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: Any | None = None
    kind: Any | None = None
    metadata: _WorkflowMetadataInput | None = None
    spec: dict[str, Any]


class _WorkflowNodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Any | None = None
    id: Any | None = None
    spec: dict[str, Any] | None = None
    dependsOn: list[Any] | None = None
    region: dict[str, Any] | None = None
    feedback: dict[str, Any] | None = None


@dataclass
class ParsedTaskSpec:
    task: TaskEnvelopeTemplate
    depends_on: list[str]
    local_name: str | None
    graph_node_name: str | None
    load: int
    position_in_epoch: int | None = None
    selected_worker: list[str] | None = None
    v2: dict[str, Any] | None = None
    feedback: dict[str, Any] | None = None


@dataclass
class ParsedRegion:
    name: str
    region: dict[str, Any]
    depends_on: list[str]
    feedback: dict[str, Any] | None = None


@dataclass
class ParsedWorkflow:
    tasks: list["ParsedTask"]
    schedule_in_epoch_order: bool | None
    epoch_groups: list[list[str]] | None
    regions: list[ParsedRegion] = field(default_factory=list)
    api_version: Any | None = None


@dataclass
class ParsedTask:
    task_id: str
    task: TaskEnvelopeTemplate
    depends_on: list[str]
    local_name: str | None
    graph_node_name: str | None
    load: int
    position_in_epoch: int | None = None
    selected_worker: list[str] | None = None
    v2: dict[str, Any] | None = None
    feedback: dict[str, Any] | None = None


def parse_workflow(payload: str, format: str) -> ParsedWorkflow:
    match format:
        case "native":
            try:
                data = yaml.safe_load(payload)
            except Exception as exc:
                raise ValueError(f"Failed to parse YAML: {exc}")
        case "n8n":
            try:
                payload_dict = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON payload: {exc}",
                )
            if not isinstance(payload_dict, dict):
                raise ValueError(
                    "n8n payload must be a JSON object",
                )
            try:
                data = translate_n8n_workflow(payload_dict)
            except Exception as exc:
                raise ValueError(f"Failed to parse n8n workflow: {exc}")
        case _:
            raise ValueError(f"Unsupported format: {format}")
    return _expand_specs(data)


def _build_workflow(
    specs: list[ParsedTaskSpec],
    schedule_in_epoch_order: bool | None,
    epoch_groups: list[list[str]] | None,
    region_specs: list[ParsedRegion] | None = None,
    api_version: Any | None = None,
) -> ParsedWorkflow:
    region_specs = region_specs or []
    results: list[ParsedTask] = []
    local_ids: dict[str, str] = {}
    # Region operator ids are their source names; pre-register them so tasks and
    # regions may reference each other regardless of resolution order.
    for region_spec in region_specs:
        local_ids[region_spec.name] = region_spec.name

    def _resolve_feedback(
        feedback: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not feedback:
            return None
        resolved = dict(feedback)
        target = feedback.get("to")
        if isinstance(target, str) and (to := target.strip()):
            resolved["to"] = local_ids.get(to, to)
        return resolved

    for spec in specs:
        _validate_condition_depends_on(spec)
        task_id = new_task_id()
        depends_on = [
            local_ids.get(s, s) for dep in spec.depends_on if dep and (s := dep.strip())
        ]
        results.append(
            ParsedTask(
                task_id=task_id,
                task=spec.task,
                depends_on=depends_on,
                local_name=spec.local_name,
                graph_node_name=spec.graph_node_name,
                load=spec.load,
                position_in_epoch=spec.position_in_epoch,
                selected_worker=spec.selected_worker,
                v2=spec.v2,
                feedback=spec.feedback,
            )
        )
        if spec.local_name:
            local_ids[spec.local_name] = task_id
    for task in results:
        task.feedback = _resolve_feedback(task.feedback)
    regions = [
        ParsedRegion(
            name=region_spec.name,
            region=region_spec.region,
            depends_on=[
                local_ids.get(s, s)
                for dep in region_spec.depends_on
                if dep and (s := dep.strip())
            ],
            feedback=_resolve_feedback(region_spec.feedback),
        )
        for region_spec in region_specs
    ]
    return ParsedWorkflow(
        tasks=results,
        schedule_in_epoch_order=schedule_in_epoch_order,
        epoch_groups=epoch_groups,
        regions=regions,
        api_version=api_version,
    )


def _is_v2_mode(api_version: Any) -> bool:
    # Inline import breaks a parser <-> v2 package import cycle.
    from .v2.mode import ExecutionMode

    return ExecutionMode.is_v2(api_version)


def _build_task_template(
    payload: _WorkflowRootInput,
    local_name: str | None = None,
    graph_node_name: str | None = None,
) -> tuple[TaskEnvelopeTemplate, dict[str, Any] | None]:
    context = _task_context_label(local_name, graph_node_name)
    task_spec = payload.spec
    v2_block = task_spec.pop("v2", None) if isinstance(task_spec, dict) else None
    if v2_block is not None and not _is_v2_mode(payload.apiVersion):
        raise ValueError(
            f"Invalid task payload{context}: spec.v2 requires apiVersion "
            "'flowmesh/v2'"
        )
    task_type = task_spec.get("taskType")
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError(f"Invalid task payload{context}: spec.taskType is required")
    if task_type == TaskType.AGENT and not _is_v2_mode(payload.apiVersion):
        raise ValueError(
            f"Invalid task payload{context}: an 'agent' task requires apiVersion "
            "'flowmesh/v2' with a resolved harness binding (declare spec.harness or "
            "configure a deployment default harness backend). See "
            "examples/templates/agent_episode.yaml."
        )
    try:
        task = TaskEnvelopeTemplate.model_validate(payload)
    except ValidationError as exc:
        if any(
            err.get("type") == "union_tag_invalid"
            and tuple(err.get("loc", ())) == ("spec",)
            for err in exc.errors()
        ):
            raise ValueError(
                f"Invalid task payload{context}: unsupported spec.taskType "
                f"{task_type!r}. Allowed values: {_ALLOWED_TASK_TYPES}"
            ) from exc
        raise ValueError(f"Invalid task payload{context}: {exc}") from exc
    _validate_ssh_access_mode(task, context)
    _validate_serve_access_mode(task, context)
    try:
        task.spec.validate_dispatchable()
    except ValueError as exc:
        raise ValueError(f"Invalid task payload{context}: {exc}") from exc
    return task, v2_block if isinstance(v2_block, dict) else None


def _task_context_label(local_name: str | None, graph_node_name: str | None) -> str:
    if graph_node_name:
        return f" for graph node {graph_node_name!r}"
    if local_name:
        return f" for stage {local_name!r}"
    return ""


def _validate_workflow_root(data: Any) -> _WorkflowRootInput:
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict")
    try:
        return _WorkflowRootInput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid workflow payload: {exc}") from exc


def _validate_workflow_node(node: Any, path: str) -> _WorkflowNodeInput:
    if not isinstance(node, dict):
        raise ValueError(f"{path} must be a mapping/dict")
    try:
        return _WorkflowNodeInput.model_validate(node)
    except ValidationError as exc:
        raise ValueError(f"Invalid workflow payload at {path}: {exc}") from exc


def _expand_specs(data: Any) -> ParsedWorkflow:
    root = _validate_workflow_root(data)

    base_clean = root.model_copy(deep=True)
    base_clean.spec.pop("stages", None)
    base_clean.spec.pop("graph", None)

    stages = root.spec.get("stages")
    graph_nodes = safe_get(root.spec, "graph.nodes")
    if stages and graph_nodes:
        raise ValueError("spec.stages cannot be combined with spec.graph")
    if isinstance(stages, list) and stages:
        return _expand_stages(root, base_clean, stages)
    if isinstance(graph_nodes, list) and graph_nodes:
        return _expand_graph_nodes(root, base_clean, graph_nodes)

    depends_on_raw = root.spec.get("dependsOn") or []
    depends_on = _normalize_dep_list(depends_on_raw)
    task, v2_block = _build_task_template(root)
    return _build_workflow(
        specs=[
            ParsedTaskSpec(
                task=task,
                depends_on=depends_on,
                local_name=None,
                graph_node_name=None,
                load=_compute_task_load(task),
                v2=v2_block,
            )
        ],
        schedule_in_epoch_order=None,
        epoch_groups=None,
        api_version=root.apiVersion,
    )


def _expand_graph_nodes(
    base: _WorkflowRootInput, base_clean: _WorkflowRootInput, nodes: list[Any]
) -> ParsedWorkflow:
    indexed: dict[str, _WorkflowNodeInput] = {}
    base_name = (
        str(base.metadata.name) if base.metadata and base.metadata.name else "task"
    )

    for idx, node in enumerate(nodes):
        node = _validate_workflow_node(node, f"graph.nodes[{idx}]")

        if (node.region is not None or node.feedback is not None) and not _is_v2_mode(
            base.apiVersion
        ):
            raise ValueError(
                f"graph.nodes[{idx}]: region/feedback require apiVersion 'flowmesh/v2'"
            )
        if node.region is not None and node.spec is not None:
            raise ValueError(
                f"graph.nodes[{idx}]: a node declares either 'spec' (a task) or "
                "'region' (a structured region), not both"
            )

        name = node.name or node.id or f"node-{idx+1}"
        name = str(name).strip()
        if not name:
            raise ValueError("Graph node requires a non-empty name")
        if name in indexed:
            raise ValueError(f"Duplicate graph node '{name}'")
        deps = node.dependsOn or []
        if any(str(dep).strip() == name for dep in deps if dep):
            raise ValueError(f"Node '{name}' cannot depend on itself")
        indexed[name] = node

    (
        epoch_groups,
        node_position_in_epoch,
        schedule_in_epoch_order,
        selected_workers,
    ) = _parse_schedule_hint(base, indexed)

    unresolved = dict(indexed)
    results: list[ParsedTaskSpec] = []
    region_specs: list[ParsedRegion] = []

    while unresolved:
        progressed = False
        for name, node in list(unresolved.items()):
            pending = []
            unresolved_internal = False
            for dep in node.dependsOn or []:
                dep = str(dep).strip()
                if not dep:
                    continue
                if dep in unresolved:
                    unresolved_internal = True
                    break
                pending.append(dep)
            if unresolved_internal:
                continue

            if node.region is not None:
                region_specs.append(
                    ParsedRegion(
                        name=name,
                        region=node.region,
                        depends_on=pending,
                        feedback=node.feedback,
                    )
                )
                unresolved.pop(name)
                progressed = True
                continue

            eff, v2_block = _merge_workflow_task(
                base_clean, node.spec, f"{base_name}:{name}", graph_node_name=name
            )

            results.append(
                ParsedTaskSpec(
                    task=eff,
                    depends_on=pending,
                    local_name=name,
                    graph_node_name=name,
                    load=_compute_task_load(eff),
                    position_in_epoch=node_position_in_epoch.get(name),
                    selected_worker=selected_workers.get(name),
                    v2=v2_block,
                    feedback=node.feedback,
                )
            )
            unresolved.pop(name)
            progressed = True

        if not progressed:
            raise ValueError(
                "Graph contains cycles or unresolved dependencies: "
                f"{', '.join(unresolved.keys())}"
            )

    return _build_workflow(
        specs=results,
        schedule_in_epoch_order=schedule_in_epoch_order,
        epoch_groups=epoch_groups,
        region_specs=region_specs,
        api_version=base.apiVersion,
    )


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dst)
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _merge_workflow_task(
    base: _WorkflowRootInput,
    spec_overrides: dict[str, Any] | None,
    metadata_name: str,
    local_name: str | None = None,
    graph_node_name: str | None = None,
) -> tuple[TaskEnvelopeTemplate, dict[str, Any] | None]:
    merged = base.model_copy(deep=True)
    merged.spec = _deep_merge(merged.spec, spec_overrides or {})
    if merged.metadata is None:
        merged.metadata = _WorkflowMetadataInput()
    merged.metadata.name = metadata_name
    return _build_task_template(merged, local_name, graph_node_name)


def _normalize_dep_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("dependsOn must be a list")
    return [s for dep in raw if (s := str(dep).strip())]


def _expand_stages(
    base: _WorkflowRootInput,
    base_clean: _WorkflowRootInput,
    stages: list[Any],
) -> ParsedWorkflow:
    indexed: dict[str, _WorkflowNodeInput] = {}
    base_name = (
        str(base.metadata.name) if base.metadata and base.metadata.name else "task"
    )

    for idx, stage in enumerate(stages):
        stage = _validate_workflow_node(stage, f"spec.stages[{idx}]")
        if stage.region is not None or stage.feedback is not None:
            raise ValueError(
                f"spec.stages[{idx}]: structured regions are only supported in "
                "the graph form (spec.graph.nodes)"
            )
        name = stage.name or stage.id or f"stage-{idx+1}"
        name = str(name).strip()
        if not name:
            raise ValueError(f"Stage[{idx}] requires a non-empty name")
        if name in indexed:
            raise ValueError(f"Duplicate stage '{name}'")
        indexed[name] = stage

    (
        epoch_groups,
        node_position_in_epoch,
        schedule_in_epoch_order,
        selected_workers,
    ) = _parse_schedule_hint(base, indexed)

    results: list[ParsedTaskSpec] = []
    ordered_names = list(indexed.keys())
    for idx, name in enumerate(ordered_names):
        stage = indexed[name]
        eff, v2_block = _merge_workflow_task(
            base_clean, stage.spec, f"{base_name}:{name}", local_name=name
        )

        depends_raw = stage.dependsOn
        depends_on = _normalize_dep_list(depends_raw) if depends_raw is not None else []
        if not depends_on and idx > 0:
            depends_on = [ordered_names[idx - 1]]

        results.append(
            ParsedTaskSpec(
                task=eff,
                depends_on=depends_on,
                local_name=name,
                graph_node_name=None,
                load=_compute_task_load(eff),
                position_in_epoch=node_position_in_epoch.get(name),
                selected_worker=selected_workers.get(name),
                v2=v2_block,
            )
        )

    return _build_workflow(
        specs=results,
        schedule_in_epoch_order=schedule_in_epoch_order,
        epoch_groups=epoch_groups,
        api_version=base.apiVersion,
    )


def _compute_task_load(task: TaskEnvelopeTemplate) -> int:
    resources = task.spec.resources
    if resources is None:
        return 0
    load_value = resources.estimatedLoad
    if load_value is None or not isinstance(load_value, int):
        return 0
    try:
        return int(load_value)
    except Exception:
        return 0


def _parse_schedule_hint(
    base: _WorkflowRootInput, indexed: dict[str, _WorkflowNodeInput]
) -> tuple[list[list[str]] | None, dict[str, int], bool | None, dict[str, list[str]]]:
    """
    Return epoch groups, in-epoch order positions, workflow flag, and worker hints.
    """

    def _parse_node_name(value: Any, path: str) -> str:
        node_name = str(value).strip()
        if node_name == "":
            raise ValueError(f"{path} must contain non-empty node names")
        return node_name

    def _normalize_selected_worker_list(value: Any, path: str) -> list[str]:
        if isinstance(value, str):
            selected = value.strip()
            if not selected:
                raise ValueError(f"{path} must be a non-empty string")
            return [selected]
        if isinstance(value, list):
            candidates = [s for item in value if (s := str(item).strip())]
            if not candidates:
                raise ValueError(f"{path} list must contain non-empty strings")
            return list(dict.fromkeys(candidates))
        raise ValueError(f"{path} must be a string or list of strings")

    annotations = base.metadata.annotations if base.metadata else None
    if annotations is None:
        return None, {}, None, {}

    schedule_hint = annotations.schedule_hint
    if schedule_hint is None:
        return None, {}, None, {}

    schedule_in_epoch_order = schedule_hint.node_schedule_in_epoch_order
    node_order = schedule_hint.node_execution_order
    selected_worker = schedule_hint.selected_worker

    node_names = set(indexed.keys())
    if not isinstance(schedule_in_epoch_order, bool):
        raise ValueError("schedule_hint.node_schedule_in_epoch_order must be a boolean")

    epoch_groups: list[list[str]] | None = None
    node_position_in_epoch: dict[str, int] = {}
    node_epoch_index: dict[str, int] = {}

    if node_order is None:
        schedule_in_epoch_order = None
    else:
        if not isinstance(node_order, list):
            raise ValueError("schedule_hint.node_execution_order must be a list")

        flat_order: list[str] = []
        if node_order and all(isinstance(item, list) for item in node_order):
            epoch_groups = []
            for epoch_idx, epoch_nodes in enumerate(node_order):
                if not isinstance(epoch_nodes, list) or not epoch_nodes:
                    raise ValueError(
                        "schedule_hint.node_execution_order epoch entries "
                        "must be non-empty lists"
                    )
                epoch_names: list[str] = []
                for node_idx, node_name in enumerate(epoch_nodes):
                    parsed_name = _parse_node_name(
                        node_name,
                        "schedule_hint.node_execution_order"
                        f"[{epoch_idx}][{node_idx}]",
                    )
                    flat_order.append(parsed_name)
                    epoch_names.append(parsed_name)
                    node_epoch_index[parsed_name] = epoch_idx
                    if schedule_in_epoch_order:
                        node_position_in_epoch[parsed_name] = node_idx
                epoch_groups.append(epoch_names)
        elif any(isinstance(item, list) for item in node_order):
            raise ValueError(
                "schedule_hint.node_execution_order must be list[str] "
                "or list[list[str]]"
            )
        else:
            if not schedule_in_epoch_order:
                raise ValueError(
                    "schedule_hint.node_execution_order flat list requires "
                    "node_schedule_in_epoch_order to be true"
                )
            epoch_group = []
            for idx, node_name in enumerate(node_order):
                parsed_name = _parse_node_name(
                    node_name,
                    f"schedule_hint.node_execution_order[{idx}]",
                )
                flat_order.append(parsed_name)
                epoch_group.append(parsed_name)
                node_epoch_index[parsed_name] = 0
                node_position_in_epoch[parsed_name] = idx
            epoch_groups = [epoch_group]

        order_set = set(flat_order)
        if node_names != order_set or len(flat_order) != len(order_set):
            raise ValueError(
                "schedule_hint.node_execution_order must be a permutation of nodes"
            )

        # Validate dependency consistency with schedule_hint.
        for name, node in indexed.items():
            deps = node.dependsOn or []
            for dep in deps:
                dep_name = str(dep)
                if dep_name not in node_epoch_index:
                    continue
                if node_epoch_index[dep_name] > node_epoch_index[name]:
                    raise ValueError(
                        "node_execution_order conflicts with dependency: "
                        f"node '{name}' depends on '{dep_name}' "
                        "but it appears after the node in node_execution_order"
                    )
                if (
                    schedule_in_epoch_order
                    and node_epoch_index[dep_name] == node_epoch_index[name]
                    and node_position_in_epoch[dep_name] > node_position_in_epoch[name]
                ):
                    raise ValueError(
                        "node_execution_order conflicts with dependency: "
                        f"node '{name}' depends on '{dep_name}' "
                        "but it appears after the node within its epoch"
                    )

    node_selected_workers: dict[str, list[str]] = {}
    if selected_worker:
        if isinstance(selected_worker, (str, list)):
            global_workers = _normalize_selected_worker_list(
                selected_worker, "schedule_hint.selected_worker"
            )
            node_selected_workers = {name: global_workers for name in node_names}
        else:
            global_workers_payload = selected_worker.global_
            if global_workers_payload is not None:
                global_workers = _normalize_selected_worker_list(
                    global_workers_payload,
                    "schedule_hint.selected_worker['global']",
                )
                node_selected_workers = {name: global_workers for name in node_names}
            selected_workers_payload = selected_worker.selected or {}
            for key, value in selected_workers_payload.items():
                if key not in node_names:
                    raise ValueError(
                        "schedule_hint.selected_worker['selected'] references "
                        f"unknown node '{key}'"
                    )
                node_selected_workers[key] = _normalize_selected_worker_list(
                    value,
                    f"schedule_hint.selected_worker['selected']['{key}']",
                )

    return (
        epoch_groups,
        node_position_in_epoch,
        schedule_in_epoch_order,
        node_selected_workers,
    )


# ------------------------------------------------------------------ #
# Additional task spec validation
# ------------------------------------------------------------------ #


def _validate_ssh_access_mode(task: TaskEnvelopeTemplate, context: str) -> None:
    if task.spec.taskType != TaskType.SSH:
        return
    # Non-interactive SSH tasks don't use access modes.
    if task.spec.interactive is False:
        return
    access_mode = task.spec.accessMode or "direct"
    if access_mode == "proxy" and not _ENABLE_SERVER_SSH_PROXY:
        raise ValueError(
            f"Invalid task payload{context}: SSH accessMode 'proxy' "
            "is disabled on this server"
        )
    if access_mode == "forward" and not _ENABLE_SERVER_PORT_FORWARD:
        raise ValueError(
            f"Invalid task payload{context}: SSH accessMode 'forward' "
            "is disabled on this server"
        )


def _validate_serve_access_mode(task: TaskEnvelopeTemplate, context: str) -> None:
    if task.spec.taskType != TaskType.SERVE:
        return
    access_mode = task.spec.accessMode or "direct"
    if access_mode == "proxy" and not _ENABLE_SERVER_SERVE_PROXY:
        raise ValueError(
            f"Invalid task payload{context}: serve accessMode 'proxy' "
            "is disabled on this server"
        )
    if access_mode == "forward" and not _ENABLE_SERVER_PORT_FORWARD:
        raise ValueError(
            f"Invalid task payload{context}: serve accessMode 'forward' "
            "is disabled on this server"
        )


def _validate_condition_depends_on(spec: ParsedTaskSpec) -> None:
    """Ensure ``spec.condition.node`` appears in the task's effective dependsOn."""
    condition = spec.task.spec.condition
    if condition is None:
        return
    node = condition.node.strip()
    deps = {s for dep in spec.depends_on if dep and (s := dep.strip())}
    if node in deps:
        return
    node_label = spec.graph_node_name or spec.local_name or "<task>"
    raise ValueError(
        f"Task '{node_label}': condition.node '{condition.node}' must appear "
        "in dependsOn."
    )
