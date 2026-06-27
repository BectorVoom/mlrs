# Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove (or refute) that GPU tree construction — **single-owner GATHER histogram → relabel-partition → seed-from-first split-find** — lowers and is tractable under cpu-MLIR, delivering an explicit **GO / ADJUST / ABORT** verdict that gates the entire tree chain (Phases 18→19→20→21: tree prims/DecisionTree → RandomForest → FIL → TreeSHAP). Models the v3.0 Phase 13 KNN-graph keystone spike.

**In scope:** standalone-launching GATHER-histogram + relabel-partition + seed-from-first split-find kernels on cpu(f64)+rocm(f32); a VALUE-asserting single-tree correctness witness vs `sklearn.tree.DecisionTree*` on injected fixed bootstrap/feature indices; a finalized `SparseTreeNode` format contract; a per-tree cost benchmark; A1–A5 abort-signal evaluation; the documented two-tier stochastic-gate convention; the GO/ADJUST/ABORT verdict.

**Out of scope (gated behind a GO verdict):** the production `quantiles`/`tree_hist`/`best_split`/`node_partition` prims (Phase 18), any estimator surface (`DecisionTree*`/`RandomForest*`), FIL, TreeSHAP, RNG-driven bootstrap sampling (spike injects fixed indices), multiclass leaf-buffer implementation, entropy/log_loss/absolute_error criteria.

</domain>

<decisions>
## Implementation Decisions

### Spike disposition
- **D-01:** Mirror Phase 13 exactly. Risky lowering probes live as **throwaway raw experiments** in `.planning/spikes/NNN-*` with a `spike_test.rs`-style live-launch harness; proven findings are wrapped into a **spike-findings skill** (à la `spike-findings-mlrs`). Phase 18 re-authors the production prims from the findings doc — the spike code is evidence, not promoted code. Deliver a `VERDICT.md` (GO/ADJUST/ABORT) as the gating artifact.

### SparseTreeNode format contract (FINALIZED — load-bearing for FIL Phase 20 + TreeSHAP Phase 21)
- **D-02:** Fields: `SparseTreeNode { colid, threshold, left_child, value }`, with **right child = `left_child + 1`** (locked by TREE-01).
- **D-03:** A **leaf** is marked by the sentinel **`colid = -1`** (treelite/cuML-FIL convention). FIL's iterative traversal stops on `colid < 0`. `colid` is therefore a **signed integer (i32)**; `threshold` is `F`; `left_child` is signed (i32).
- **D-04:** `value` is **NOT a scalar prediction** — it is an **offset/index into a shared leaf-value buffer**. This makes the format **multiclass-uniform from day one**: binary, multiclass, and regression leaves all index a side buffer. Downstream agents MUST treat `value` as an index, not a prediction. The spike *defines and validates the contract* with the binary + regression witness; the multiclass leaf-buffer population path is exercised by Phase 19 onward.

### Verdict criteria (A3 cost-tractability)
- **D-05:** A3 is judged by **tractability + scaling shape**, not an absolute cpu-MLIR wall-clock ceiling (cpu is the *correctness* gate; rocm/cuda are the real runtime target). Record the per-tree benchmark on the representative fixture (≈1000 samples × 20 features × 128 bins × depth 8); confirm cost scales **sub-quadratically** in samples/bins and completes in reasonable time. A3 fires only on clearly-impractical cost (e.g. multi-second/tree) or pathological bin-scaling.

### ADJUST ladder (order of re-scope levers if a signal fires)
- **D-06:** **fewer bins (128→64) → frontier-only histogramming → shallower default `max_depth` → defer RF/FIL/TreeSHAP out of v4.0** (last resort). Cheapest change first; the bins lever is data-backed by D-10's 64-vs-128 benchmark.

### Two-tier stochastic-gate convention (milestone-wide standard — DOCUMENTED here)
- **D-07:** **Tier 1 (deterministic core):** on **injected fixed bootstrap/feature indices**, a single tree matches `sklearn.tree.DecisionTree*` — exact split structure + ≤1e-5 leaf values (f64). This is the real correctness witness (abort signal A5).
- **D-08:** **Tier 2 (ensemble/predictive band):** because SplitMix64 ≠ MT19937, forests are gated on **predictive quality only, never element-wise**. Standard = an **absolute band**: classifier test-accuracy within ~0.02–0.05 of `sklearn.ensemble.RandomForest*`, regressor R² within ~0.02–0.05, on a fixed synthetic dataset + fixed `n_estimators` + fixed seed. (Tier 2 is *documented* this phase, not *run* — no forest exists yet; it governs Phase 19.)

### Correctness witness scope (Tier-1 this spike)
- **D-09:** Validate **both** `DecisionTreeClassifier(criterion='gini')` **and** `DecisionTreeRegressor(criterion='squared_error')` on injected fixed indices. This exercises both leaf-value paths the format must carry (class-probability leaves + regression-mean leaf) through the shared leaf-value buffer. `entropy`/`log_loss`/`absolute_error` deferred to Phase 18/19.

### Binning
- **D-10:** Compute **quantile bin edges on the host once per fit** (A2 mitigation — "almost certainly acceptable" per PITFALLS); device kernels consume precomputed edges only — **no on-device sort/scan**. Default **128 bins**; the spike benchmarks **64 vs 128** so the D-06 "fewer bins" lever has data.

### Claude's Discretion
- Exact relabel-partition kernel mechanics, benchmark-harness plumbing, fixture seeding details, and the rocm f32 skip-with-log wiring follow established Phase 13 / `cpu-mlir-kernel-authoring` conventions — planner/researcher's call within the locked op-set.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spike recipe & cpu-MLIR op-set (PRIMARY — read first)
- `.claude/skills/spike-findings-mlrs/SKILL.md` — invoke via `Skill("spike-findings-mlrs")`; the proven GATHER op-set and landmine index
- `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md` — proven op-set + the two landmines: **002-A** (bare-`ABSOLUTE_POS` 1D launch → loud MLIR pass failure) and **002-B** (cross-sibling-loop mutable accumulator → **SILENT miscompile**); banned-entirely list (`SharedMemory`, `Atomic`, `F::INFINITY`, mutable-bool scans, descending-shift loops)
- `.claude/skills/spike-findings-mlrs/references/knn-graph-primitive.md` — kernel-shape/composition reference + duplicate-row VALUE-assert discipline

### Abort signals & milestone gate regimes
- `.planning/research/PITFALLS.md` §"abort signals" — **A1** (lowering fails), **A2** (binning needs unlowering sort), **A3** (correct-but-superlinear cost), **A4** (split-find can't argmax safely), **A5** (correctness fail vs sklearn) — each MUST be evaluated in the verdict
- `.planning/research/SUMMARY.md` — P1 critical risk framing (atomics/SharedMemory) + GATHER inversion rationale
- `.planning/ROADMAP.md` §"Phase 17" — Goal, 5 success criteria, gate; §"Gate regimes" (two-tier convention)
- `.planning/REQUIREMENTS.md` §**TREE-01** — the spike requirement + gate

### Spike precedent to mirror
- `.planning/milestones/v3.0-phases/13-knn-graph-primitive-feasibility-keystone/` — Phase 13 CONTEXT/PLAN/VERIFICATION/DISCUSSION-LOG (structure to replicate)
- `.planning/spikes/MANIFEST.md`, `.planning/spikes/CONVENTIONS.md`, `.planning/spikes/WRAP-UP-SUMMARY.md` — raw-spike layout + wrap-up convention

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/mlrs-backend/tests/spike_test.rs` — the live-launch harness pattern (owns a concrete runtime feature; `runtime::active_client()`, ceiling-div launch dims, host-oracle VALUE assertions). Clone its shape for the tree-kernel probes.
- `crates/mlrs-backend/src/prims/` (e.g. `distance.rs`, `topk.rs`, `reduce.rs`) — proven cpu-MLIR-safe kernel idioms; the feature-loop accumulator + `F::powf` shape from Spike 001 is directly reusable for the histogram inner loop.
- `crates/mlrs-backend/src/capability.rs` + `runtime.rs` — `FloatKind` / `ActiveRuntime` for the f64-on-rocm skip-with-log gate.
- `crates/mlrs-core/src/oracle.rs` + `crates/mlrs-core/tests/compare_test.rs` + `examples/gen_fixture.rs` — the sklearn-oracle fixture pattern (committed blobs; regen needs a numpy venv) for the Tier-1 DecisionTree witness.

### Established Patterns
- **cpu-MLIR-safe kernel authoring** (D-02..D-10 all depend on it): `CUBE_POS_X`/`UNIT_POS_X==0` per-row launch shape; runtime `while` loops with same-iteration accumulators; statement-form `if` for running argmax (seed-from-first, never `F::INFINITY`); no cross-sibling-loop accumulators.
- **Primitive-first + standalone-gate-before-consumer** discipline (Phase 13 keystone precedent).
- **Tests separated from source** (project AGENTS.md rule) — kernels in `src/`, harness/assertions in `tests/`.

### Integration Points
- Spike kernels target `crates/mlrs-backend`; the `SparseTreeNode` format contract (D-02..D-04) is the interface Phase 18 prims, Phase 20 FIL, and Phase 21 TreeSHAP all bind to.

</code_context>

<specifics>
## Specific Ideas

- The single-owner **GATHER inversion**: one unit per `(node, feature, bin)` loops over the node's sample range with local accumulation — the SharedMemory/atomicAdd-free histogram (vs cuML's `__shared__` + `atomicAdd`).
- **Relabel** (not scan/compaction) for node partition.
- **Seed-from-first** statement-form `if` for the gain argmax (avoids A4 / `F::INFINITY` / 002-B).
- Benchmark fixture anchor: ≈1000 samples × 20 features × 128 bins × depth 8.
- VALUE-assert discipline (never non-panic) — a silent miscompile (002-B) compiles, launches, and returns plausible wrong data.

</specifics>

<deferred>
## Deferred Ideas

- **Multiclass leaf-value-buffer population path** — format contract (offset semantics, D-04) is *defined* this spike; populating/consuming it for >2 classes is Phase 19+.
- **`entropy`/`log_loss`/`absolute_error` criteria** — witness covers gini + squared_error only (D-09); other criteria → Phase 18/19.
- **RNG-driven bootstrap/feature sampling** — spike injects fixed indices; SplitMix64 sampling is Phase 19 (RF).
- **Production prims** (`quantiles`/`tree_hist`/`best_split`/`node_partition`) — Phase 18 promotes from findings.
- **On-device bin-edge computation** — host pre-pass chosen (D-10); revisit only if A2 mitigation proves insufficient.

None of these are scope creep — all are explicitly downstream of the gating verdict.

</deferred>

---

*Phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating*
*Context gathered: 2026-06-27*
