# Resident-capacity control

Resident-capacity control serves a workflow's `resident` model bindings from reusable
physical capacity. An agent whose managed model resolves to a resident binding does not call
an external endpoint: the invocation is admitted to a compatible model-serving replica the
fabric materializes, sizes, and reclaims. Inference (vLLM) is the first resident family; the
GPU-free `dev_model` executor is a serving stand-in for the same admission path.

The subsystem is two actors over durable control-state (`CS`) stores. Actors do; stores
hold. It adds no worker-local service plane: a worker generically hosts a leased replica and
reports its endpoint and health, but owns no family directory, replica selection, scaling,
or routing.

## Actors

- **Admission controller** — the fast loop. It fair-orders claims from the DemandLedger,
  selects a compatible replica from the derived CapacityPools and the Replica directory,
  atomically records the fenced `ServiceClaim.RESERVED` credit, records an engine enqueue
  acknowledgement as `ACCEPTED`, and consumes a fenced terminal outcome from the
  orchestration ledger (`DS`) by `invocation_id` to release the credit. It writes only claim
  facts.
- **Lifecycle & scale manager** — the slower loop. It owns allocation leases and
  replica-directory lifecycle, performs policy-bounded, demand-driven scale-from-zero for
  approved plan-derived families, drains before an idle teardown, and reconciles preemption.
  Materializing and stopping a replica cross the flat worker plane by dispatching and
  cancelling a serve (or `dev_model`) task; an operator reads a resident replica's logs
  through the normal task-log path.

## Stores

Authoritative `CS` stores are durable and rehydrate on restart. Only `ServiceClaim` facts
are authoritative for admission credits; the two derived views recompute from them on every
read and cannot diverge.

| Store | Kind | Holds |
| --- | --- | --- |
| Service-family registry | Authoritative | Policy-approved, plan-derived family definitions. Registration materializes no capacity. |
| Replica directory | Authoritative | Live replica incarnations, endpoints, health, and the incarnation fence. |
| DemandLedger | Read state | Unadmitted-claim references the controller fair-orders. Never promotes a claim itself. |
| Allocation leases | Authoritative | Lease records and lifecycle ownership per replica. |
| Invocation requests | Authoritative | The durable request record keyed by `invocation_id`, carrying the admission profile. |
| `ServiceClaim` facts | Authoritative | The per-admission credit and causal FSM. The sole authority for credits. |
| CapacityPools | Derived | Feasible replicas for a claim, net of every outstanding credit-bearing claim. |
| Admission-credit ledger | Derived | Per-replica held credit, recomputed from credit-bearing claims. |
| Capacity reports | Telemetry | Conservative normalized safe-capacity evidence, fenced by incarnation and report epoch. Never a credit authority. |

`DS` and `CS` link only through the stable `invocation_id` and fenced terminal outcomes;
neither side infers or overwrites the other's facts. The durable `DS` invocation holds the
identity and terminal state; the admission profile and the mid-claim credit state live only
in `CS`, so both persist to survive a restart.

## Claim admission FSM

A claim carries a fresh `claim_id` and admission epoch under its invocation's stable id.
`PENDING` and `TERMINAL` hold no credit; every other state is a credit-bearing nonterminal
fact.

```
PENDING → RESERVED → ACCEPTED → STREAMING → TERMINAL
PENDING / RESERVED → TERMINAL               (cancel, known enqueue failure, expiry)
RESERVED / ACCEPTED / STREAMING → UNCERTAIN → TERMINAL
TERMINAL --(permitted reissue)--> successor PENDING (same invocation_id, fresh epoch)
```

- **Fencing.** `RESERVED` stamps the selected replica's incarnation onto the claim, so a
  superseded incarnation cannot be mistaken for this credit. Concurrent reservations cannot
  overcommit a replica's reported safe slots.
- **Credit release.** A `TERMINAL` transition releases the derived credit. For an accepted
  or streaming claim the only normal release is a fenced terminal outcome recorded in `DS`
  and consumed by `invocation_id`; a stream close or a telemetry report alone never releases
  it. A pre-acceptance cancellation, known enqueue failure, or expiry records a terminal
  transition directly.
- **Loss and reissue.** A route or incarnation loss moves a credit-bearing claim to
  `UNCERTAIN`, holding its credit until the fenced terminal outcome for its invocation
  settles it. A re-driven boundary resumes the in-flight claim on its replica under the held
  credit; once a claim is terminal, a permitted reissue raises a successor with a fresh
  admission epoch, never reopening a terminal claim or reusing its credit.

## Handoff and execution

A `RESERVED` claim authorizes one opaque, short-lived, claim-bound admission handoff — a
descriptor an engine adapter consumes to reach the selected replica and obtain an enqueue
acknowledgement. It is neither a persisted control object nor a network route authorization.
The handoff is locality-neutral: its fields are data an adapter consumes, not a
server-owned client. The default inference adapter consumes it in-server and relays the
request to the replica's OpenAI-compatible endpoint, so where the bytes run is not fixed by
the contract.

## Replica lifecycle and policy

```
ABSENT → MATERIALIZING → WARM ↔ BUSY
WARM / BUSY → DRAINING → STOPPED
WARM / BUSY → PREEMPTED / FAILED → reconcile and recreate
```

The first eligible `PENDING` claim for an approved family with no capacity triggers a
bounded zero-to-one materialization. Before creating an allocation, policy checks the allowed
model catalog, the per-family replica quota, and the concurrent cold-start limit; a refusal
is a typed durable outcome that creates no allocation, handoff, or credit and never
substitutes an external provider. A drain rejects new claims while admitted work reaches a
safe outcome. Idle teardown is off by default; when a retain window is configured, a
background sweep drains a servable replica that has held no credit past the window and stops
it once drained, and a later eligible claim materializes the family from zero again.

## Selection strategy

Replica selection is swappable and bound per approved family, not per workflow or request,
selected per deployment through `RESIDENT_SELECTION_STRATEGY`. The default
`batch-aware-best-fit` fills the feasible replica with the least remaining safe headroom
before spilling; `least-load` and `round-robin` are also available. The choice changes only
which feasible replica a claim reserves; the engine still owns continuous batching, token
scheduling, and KV allocation.

## Configuration

Resident-capacity control is off by default and enabled per deployment. See the `RESIDENT_*`
rows in [`ENV.md`](ENV.md) for enablement, the serving substrate (`serve` or `dev_model`),
the policy caps, the conservative admission-slot count, the cold-start budget, the per-family
selection strategy, and the idle-teardown retain window (`RESIDENT_IDLE_RETAIN_SEC`, `0`
disables).
