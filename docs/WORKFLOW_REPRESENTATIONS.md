# Workflow representations (v2)

FlowMesh builds three durable plan-time representations and a gate that selects
them, decoupling author intent, legal workflow semantics, and physical
realization. The models live in `src/server/task/v2/representations/`; the
compiler that produces them lives in `src/server/task/v2/compiler/`.

## Selecting the v2 track

The workflow-root `apiVersion` selects the track:

- `apiVersion: flowmesh/v2` builds and persists the v2 representations.
- Any other value — `flowmesh/v1`, omitted, or unrecognized — stays on the v1
  path and writes no v2 record.

Existing task, `spec.stages`, and `spec.graph.nodes` forms project to the
acyclic subset of a logical template.

## The three plan-time representations

| Representation | Question | Contents |
| --- | --- | --- |
| `FrontendWorkflowSource` | What did the author submit? | Verbatim payload, format, content digest, submission time. Immutable provenance. |
| `LogicalWorkflowTemplate` | What behavior is legal? | Typed operators, port wiring, tool/resource declarations, result declarations, legacy projections, effect boundaries, source maps. |
| `PhysicalExecutionPlan` | How may the fabric realize it? | Physical nodes, source maps to the template, service-family and residency hooks. |

The source is compiler input retained for provenance and diagnostics, never a
durable runtime object and never conflated with the template/plan version or the
workflow's task-ID list.

### Operator vocabulary

A `LogicalWorkflowTemplate` carries a small symbolic operator vocabulary:

- a generic `LeafOperator` parameterized by a `LeafProfile`
  (determinism, effect, recovery, input provenance, output equality) and a
  symbolic `BindingKey`;
- the opaque-body `AgentOperator` with an `AuthorityCeiling` and a finite
  `BoundarySignature`;
- symbolic region forms `BranchRegion` / `MergeRegion`, `SpawnRegion` /
  `JoinRegion`, and `LoopContextRegion`;
- typed `Port`s carrying values, `StateReference`s, or `ModelRef`s.

A `BindingKey` names compatible executor semantics symbolically. It is not a
worker, replica, episode cut, or feasibility commitment. A template carries no
activation tags and no worker/replica/endpoint bindings.

### Result declarations and the legacy projection

A `ResultDeclaration` names a logical output with its cardinality kind, release
condition, visibility/retention, and the operator it resolves from. Each legacy
task result value induces one singleton output slot through a
`LegacyLogicalTaskProjection`. Task logs and arbitrary artifacts stay
source-mapped diagnostics under the legacy task identity; they are not promoted
to logical output slots.

A legacy `serve` task maps to a residency-administration node with a
`ResidencyIntent`, not a result-owning leaf.

## Compilation

`compile_workflow` (in `src/server/task/v2/compiler/pipeline.py`) lowers a parsed
workflow into the logical template and physical plan. It lowers legacy tasks
into leaf/agent/residency operators, normalizes structured regions and
`spec.v2` leaf declarations into the canonical operator/region form, builds one
physical boundary per legacy task/executor boundary, and runs the validation
passes. It runs before any scheduler state is created, so a rejected submission
mutates nothing.

The `taskType → Executor` mapping is a binding adapter in `bindings.py`: it
translates each legacy task type into a generic `Leaf`/`Agent` profile.
`agent` binds to the opaque-body `Agent`; `serve` binds to a residency request
surface; every other type binds to a generic `Leaf`. The legacy task-type set
is not the v2 operator vocabulary.

## Validation

`validate_compilation` runs consistency passes over the compiled template and
plan and reports each finding as a diagnostic with a readable source location.
It checks port wiring, region and authority-ceiling well-formedness,
recovery/effect/provenance consistency, result declarations, source-map
completeness, and unstructured cycles. An error-severity diagnostic — or a
malformed template — raises `CompileError`.

## Inspection

`build_inspection` compiles a submission into an `InspectionReport` carrying the
template, plan, and diagnostics, with a text rendering of the compiled
structure. `POST /api/v1/workflows/validate` returns it for a `flowmesh/v2`
submission without persisting or executing it.

## Versioning

`VersionId` identifies one immutable revision by `lineage`, `revision`, and
`content_digest`. A revision is never mutated in place. A later revision is a
compatible successor: the same lineage with a strictly higher revision. A
`PhysicalExecutionPlan` pins the `template_version` it lowered.

## Persistence

The bundle persists as one immutable record at `workflow:{id}:v2`, written inside
the same atomic transaction that registers the workflow. It is retrieved with
`WorkflowRegistry.get_v2_workflow[_async]`.
