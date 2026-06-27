# Phase 18: Tree Primitives + DecisionTree Core - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Promote the Phase-17 spike's kernel *probes* into **production primitives** with full prim
contracts (`quantiles`, `tree_hist`, `best_split`, `node_partition`, standalone-validated in
`mlrs-backend`) and deliver an **oracle-gated `DecisionTree` core** (level-wise host loop) plus a
**standalone Rust `DecisionTreeClassifier`/`DecisionTreeRegressor` estimator** — primitive-first,
before Phase 19's RandomForest assembles N of them.

**In scope:** the four standalone prims; the host build loop producing the finalized
`SparseTreeNode` array + shared leaf-value buffer; a user-facing Rust estimator (Phase-16
builder/typestate, `fit`/`predict`) with **host-side tree traversal** for predict; the **full**
sklearn split-criterion menu and the **full** sklearn stopping-rule hyperparameter surface
(incl. `max_leaf_nodes` best-first growth + `ccp_alpha` pruning); a tight active-frontier
PoolStats memory gate; Tier-1 oracle gates vs `sklearn.tree.DecisionTree*`.

**Out of scope (downstream phases):** the **Python/PyO3 wheel surface** for DecisionTree (deferred,
no requirement asks for it; RF Phase 19 is the intended Python entry point); **device-batched node
traversal** (FIL, Phase 20 — Phase 18 ships only a host walk); **RNG-driven bootstrap/feature
sampling** + the ensemble/Tier-2 band gate (Phase 19 RF); multiclass leaf-buffer population beyond
what the core's `predict` needs to consume; TreeSHAP (Phase 21).

> **⚠ Scope note (load-bearing for planner + a REQUIREMENTS/ROADMAP sync):** these decisions
> **broaden TREE-02** well past its literal text ("a DecisionTree* *core* (level-wise host loop)
> … matches sklearn on **injected fixed indices**"). The phase now also ships a **public Rust
> estimator**, the **full criterion menu**, the **full hyperparameter surface**, **best-first
> growth**, and **cost-complexity pruning**. This is in-character for this project (the owner
> consistently expands surface beyond the minimal keystone) and is NOT scope creep into a new
> *capability* — it is a deliberate widening of the DecisionTree capability already owned by
> Phase 18. **TREE-02's success criteria and the ROADMAP Phase-18 SC list should be updated to
> reflect this widened surface** (see `confirm_creation` follow-up).

</domain>

<decisions>
## Implementation Decisions

### Public surface scope (Area 1)
- **D-01:** Phase 18 ships **the internal core PLUS a standalone user-facing Rust
  `DecisionTreeClassifier`/`DecisionTreeRegressor`** in `mlrs-algos`, fronted by the Phase-16
  builder/typestate convention (`fit`/`predict`). **No PyO3/Python wheel surface this phase** —
  Python is deferred (RF Phase 19 is the intended Python entry point; no requirement asks for a
  standalone DecisionTree estimator in Python).
- **D-02:** **Predict uses a host-side traversal** over the `SparseTreeNode` array now: follow
  `colid`/`threshold`, stop at the `colid == -1` leaf sentinel, dereference `value` into the shared
  leaf-value buffer. **FIL's device-batched `node_id` traversal stays Phase 20** and supersedes
  this for forests. Prefer promoting the **already-gated Phase-17 Tier-1 witness host walk** into
  the estimator's `predict` rather than authoring a fresh traversal.

### Split-criterion coverage (Area 2)
- **D-03:** Implement and oracle-gate the **full criterion menu**:
  classifier = `gini` / `entropy` / `log_loss`; regressor = `squared_error` / `absolute_error`.
- **D-04:** In sklearn, `log_loss` **≡** `entropy` for `DecisionTreeClassifier` (alias, both Shannon
  entropy) — **one impurity function covers both**; both read directly off `tree_hist`'s per-class
  counts.
- **D-05:** **`absolute_error` is the long-pole.** It is **median-based MAE**, which does NOT fit the
  GATHER **sum / sum-of-squares** histogram path used for `gini`/`entropy`/`squared_error`. It needs
  its **own host-median / MAE candidate-evaluation path** (binned cumulative counts → per-candidate
  median → sum of absolute deviations) and its **own oracle gate**. Flagged for the researcher as
  the highest-risk item; keep it cpu-MLIR-safe (no new banned ops). `squared_error` variance still
  follows the VERDICT caveat #4: a **second `tree_hist` launch on `y²`** for per-cell
  sum-of-squares, not a forked kernel.

### Stopping-rule hyperparameter surface (Area 3)
- **D-06:** Honor and oracle-gate the **full sklearn stopping-rule surface**: `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `max_features`, `min_impurity_decrease`,
  `max_leaf_nodes`, `min_weight_fraction_leaf`, `ccp_alpha`. (The RF-complete subset —
  `max_depth` / `min_samples_split` / `min_samples_leaf` / `max_features` — is contained in this and
  is what Phase 19 inherits.)
- **D-07:** **Dual growth modes in the host loop.** Default = **level-wise** (honors TREE-02's
  "level-wise host loop"). When `max_leaf_nodes` is set, switch to sklearn's **best-first growth**
  (expand the highest-impurity-reduction frontier node first) so the build order matches sklearn and
  the ≤1e-5 / structural gate holds. The GATHER / split-find / partition kernels are **unchanged**;
  only the host **frontier-scheduling** differs between modes.
- **D-08:** **`ccp_alpha` is a post-build pass** — minimal cost-complexity pruning over the finalized
  `SparseTreeNode` tree (compute per-subtree effective α, prune weakest-link), gated vs sklearn's
  pruned tree. It does not change the kernels; it rewrites the node array after growth.

### Prim shape + memory gate (Area 4)
- **D-09:** **`quantiles` is a HOST prim** in `mlrs-backend` (per-feature quantile/percentile bin
  edges + digitize raw `F` → `u32` bins), standalone-gated against a numpy/sklearn reference
  (`np.percentile` / `KBinsDiscretizer` edge semantics). This honors D-10 of Phase 17 (host pre-pass,
  **no device sort**) and keeps abort-signal **A2 = PASS** (no on-device sort/scan kernel exists
  anywhere). "Standalone-validated" for `quantiles` therefore means a **host oracle test**, not a
  kernel launch test.
- **D-10:** `tree_hist`, `best_split`, `node_partition` are the **CubeCL device kernels** (the locked
  Phase-17 op-set: single-owner GATHER histogram, seed-from-candidate-0 argmax with `u32` flags,
  per-sample relabel partition). Each gets its own standalone launch-and-VALUE-assert gate carrying
  the 002-A (loud all-zeros) and 002-B (silent cross-loop) positive guards from the spike.
- **D-11:** **Memory gate = tight active-frontier assert (build-failing, SC-4).** `tree_hist` scratch
  is sized by the **active frontier** (live nodes at the current level), NOT the cumulative node
  count the Phase-17 spike used (~309K-cell peak — the VERDICT's unrealized win). A PoolStats test
  asserts peak histogram allocation **≤ `frontier_nodes × n_features × n_bins × buffers` (+ small
  slack)** and the build **FAILS** if any cumulative-node regression creeps back in. Aligns with the
  project's first-class memory-efficiency value.

### Binning defaults (carried from Phase 17)
- **D-12:** Default **128 bins** (D-10/Phase-17); the 64-bin and frontier-only levers are
  data-backed headroom, not required to clear the gate. `best_split` gates the **decision boundary**
  (routing equivalence), not the raw `threshold` float, wherever host binning makes the binned
  threshold differ from sklearn's node-local midpoint (VERDICT caveat #1).

### Claude's Discretion
- Exact host frontier-scheduler data structure (priority queue vs sorted frontier for best-first),
  prim function signatures, scratch-buffer reuse plumbing, fixture seeding, and the rocm-f32
  skip-with-log wiring follow established `mlrs-backend` prim + Phase-17 conventions — planner/
  researcher's call within the locked op-set and the decisions above.
- Where the dual-growth scheduler and `ccp_alpha` pruning pass live (host module in `mlrs-algos` tree
  module vs a `mlrs-backend` helper) — planner's call.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase-17 spike outputs (PRIMARY — the production prims are re-authored from these)
- `.planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/VERDICT.md` — GO
  verdict; **finalized `SparseTreeNode` contract** (`colid == -1` leaf, `right = left_child + 1`,
  `value` = index into shared leaf buffer); A1–A5 dispositions; the **5 "Caveats for Phase 18"**
  (decision-equivalence threshold gating, regressor function-equivalence tie-break, frontier-memory
  observation, `y²` second-histogram for variance, locked cpu-MLIR op-set)
- `.planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/17-CONTEXT.md` —
  D-01..D-10 (spike disposition, format contract, binning, two-tier gate, criterion-witness scope)
- `crates/mlrs-backend/tests/tree_spike_probes.rs`, `tree_witness.rs`, `tree_bench.rs` — the **live
  runnable gates** (histogram lowering, split-find argmax+tie, Tier-1 sklearn witness clf+reg+
  adversarial, per-tree benchmark); the prim oracle tests should preserve their VALUE-assert + 002-A/
  002-B guard discipline
- `.planning/spikes/003-gather-histogram-lower/`, `004-seed-from-first-split-find/`,
  `005-relabel-partition/`, `006-tier1-decisiontree-witness/` — durable spike evidence copies

### cpu-MLIR op-set + landmines (read before authoring any kernel)
- `.claude/skills/spike-findings-mlrs/SKILL.md` — invoke via `Skill("spike-findings-mlrs")`
- `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md` — proven op-set +
  **002-A** (bare-`ABSOLUTE_POS` 1D launch → loud MLIR fail) and **002-B** (cross-sibling-loop mutable
  accumulator → **silent miscompile**); banned list (`SharedMemory`, `Atomic`, `F::INFINITY`,
  mutable-bool scans, descending-shift loops)

### Requirements, roadmap, gate regimes
- `.planning/REQUIREMENTS.md` §**TREE-02** — the phase requirement + gate (exact/structural on
  fixed-index tree; ≤1e-5 leaf values f64). **Candidate for a scope-sync update** (see domain note).
- `.planning/ROADMAP.md` §"Phase 18" — Goal + 4 success criteria + gate; §"Gate regimes" (two-tier
  convention). **Candidate for a scope-sync update.**
- `.planning/research/PITFALLS.md` §"abort signals" — A1–A5 (A2 = no device sort; keep it PASS)

### Existing code to mirror
- `crates/mlrs-backend/src/prims/knn_graph.rs` — Phase-13 keystone prim shape (the standalone-prim
  + standalone-gate-before-consumer precedent)
- `crates/mlrs-backend/src/prims/{distance,topk,reduce}.rs` — proven cpu-MLIR-safe kernel idioms
- `crates/mlrs-algos/src/cluster/kmeans.rs` — representative builder/typestate estimator shape
  (`fit` / `predict` / wide-builder `Option`-of-data setters); `crates/mlrs-algos/src/typestate.rs`
  — the `Predict` / `PredictLabels` traits the tree estimators implement
- `crates/mlrs-core/src/oracle.rs`, `crates/mlrs-core/tests/compare_test.rs`,
  `crates/mlrs-core/examples/gen_fixture.rs` — committed-blob sklearn-oracle fixture pattern
  (regen needs a numpy venv) for the Tier-1 DecisionTree witness + the `quantiles` host-oracle gate
- `crates/mlrs-backend/src/{capability,runtime}.rs` — `FloatKind` / `ActiveRuntime` for the
  f64-on-rocm skip-with-log gate

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- The three Phase-17 spike kernels (`tree_gather_histogram`, `tree_split_find`, relabel) already
  lower cleanly and VALUE-pass on cpu f64+f32 — Phase 18 **re-authors them as production prims**
  (D-01 of Phase 17: spike code is evidence, not promoted code), preserving the op-set verbatim.
- The Tier-1 witness host build loop + traversal (`tree_witness.rs`) is the seed for both the
  estimator's build loop and its host `predict` (D-02).
- No `tree/` or `ensemble/` module exists yet under `crates/mlrs-algos/src/` — Phase 18 creates the
  tree module; Phase 19 adds ensemble next to it.

### Established Patterns
- **cpu-MLIR-safe kernel authoring** — `CUBE_POS_X`/`UNIT_POS_X==0` (or the proven 2D guarded
  `ABSOLUTE_POS_X/Y`) launch shapes, same-iteration accumulators, statement-form `if` argmax,
  seed-from-candidate-0; no cross-sibling-loop accumulators.
- **Primitive-first + standalone-gate-before-consumer** (Phase 13 / Phase 17 precedent): the four
  prims gate standalone before the DecisionTree core composes them.
- **Builder/typestate estimator convention** (Phase 16 retrofit) — the public DecisionTree estimators
  follow it; `predict` returns labels (clf → `PredictLabels`/i32) or continuous (reg → `Predict`).
- **Tests separated from source** (project AGENTS.md rule) — kernels/build loop in `src/`, harness +
  oracle assertions in `tests/`.

### Integration Points
- The `SparseTreeNode { colid:i32, threshold:F, left_child:i32, value:i32 }` array + shared
  leaf-value buffer is the interface Phase 19 (RF, builds N of them), Phase 20 (FIL, device
  traversal), and Phase 21 (TreeSHAP) all bind to. **Phase 20 FIL must bind to the mlrs
  `colid == -1` leaf sentinel, NOT cuML's `left_child == -1`** (VERDICT cuML-divergence note).

</code_context>

<specifics>
## Specific Ideas

- `entropy`/`log_loss` share one impurity function (sklearn alias) — implement once (D-04).
- `absolute_error` is the one criterion that breaks the sum/sum-sq histogram model; budget a separate
  median/MAE host path + oracle gate for it (D-05).
- `max_leaf_nodes` is the one hyperparameter that changes growth ORDER (best-first); everything else
  is a level-wise host-loop guard (D-07).
- `ccp_alpha` is a post-growth pruning rewrite of the node array, not a build-time guard (D-08).
- Memory gate should be a true regression guard, not a loose ceiling — it encodes the VERDICT's
  frontier-only optimization as an enforced invariant (D-11).
- Benchmark/fixture anchor carried from Phase 17: ≈1000 samples × 20 features × 128 bins × depth 8.

</specifics>

<deferred>
## Deferred Ideas

- **PyO3 / Python wheel surface for DecisionTree** — deferred (no requirement; RF Phase 19 is the
  Python entry point). Revisit if a standalone-DecisionTree Python need surfaces.
- **Device-batched node traversal (FIL)** — Phase 20; Phase 18 ships only the host walk (D-02).
- **RNG-driven bootstrap + feature sampling and the Tier-2 ensemble/predictive band gate** — Phase 19
  (RF). Phase 18's gate is Tier-1 only (deterministic, structural/≤1e-5).
- **Multiclass leaf-value-buffer population beyond the core's predict need** — exercised further in
  Phase 19+ (format is multiclass-uniform from day one via `value`-as-index).
- **On-device bin-edge computation** — host pre-pass chosen (D-09); revisit only if A2 mitigation
  proves insufficient.

None of these are scope creep — all are explicitly downstream of Phase 18.

</deferred>

---

*Phase: 18-tree-primitives-decisiontree-core*
*Context gathered: 2026-06-27*
