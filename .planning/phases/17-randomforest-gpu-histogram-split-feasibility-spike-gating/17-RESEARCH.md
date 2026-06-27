# Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING) - Research

**Researched:** 2026-06-27
**Domain:** cpu-MLIR-safe CubeCL kernel authoring for GPU decision-tree construction (GATHER histogram → relabel-partition → seed-from-first split-find), sklearn-oracle correctness witnessing, and feasibility-verdict delivery — modeled on the Phase 13 KNN-graph keystone spike.
**Confidence:** HIGH — the op-set is already proven in `spike-findings-mlrs` (spikes 001/002, both VALIDATED); the three tree kernels map directly onto idioms already shipped in `distance.rs` / `topk.rs` / `self_drop_gather`; the oracle/fixture pipeline and skip-with-log gate are shipped and reusable. The single MEDIUM-confidence unknown is A3 (GATHER cost tractability), which is exactly what the spike benchmark must measure rather than assume.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Mirror Phase 13 exactly. Risky lowering probes live as **throwaway raw experiments** in `.planning/spikes/NNN-*` with a `spike_test.rs`-style live-launch harness; proven findings are wrapped into a **spike-findings skill** (à la `spike-findings-mlrs`). Phase 18 re-authors the production prims from the findings doc — the spike code is evidence, not promoted code. Deliver a `VERDICT.md` (GO/ADJUST/ABORT) as the gating artifact.
- **D-02:** Fields: `SparseTreeNode { colid, threshold, left_child, value }`, with **right child = `left_child + 1`** (locked by TREE-01).
- **D-03:** A **leaf** is marked by the sentinel **`colid = -1`** (treelite/cuML-FIL convention). FIL's iterative traversal stops on `colid < 0`. `colid` is therefore a **signed integer (i32)**; `threshold` is `F`; `left_child` is signed (i32).
- **D-04:** `value` is **NOT a scalar prediction** — it is an **offset/index into a shared leaf-value buffer**. Multiclass-uniform from day one. Downstream agents MUST treat `value` as an index, not a prediction. The spike *defines and validates the contract* with the binary + regression witness; the multiclass leaf-buffer population path is exercised by Phase 19 onward.
- **D-05:** A3 is judged by **tractability + scaling shape**, not an absolute cpu-MLIR wall-clock ceiling. Record the per-tree benchmark on the representative fixture (≈1000 samples × 20 features × 128 bins × depth 8); confirm cost scales **sub-quadratically** in samples/bins. A3 fires only on clearly-impractical cost (e.g. multi-second/tree) or pathological bin-scaling.
- **D-06:** ADJUST ladder: **fewer bins (128→64) → frontier-only histogramming → shallower default `max_depth` → defer RF/FIL/TreeSHAP out of v4.0** (last resort). Cheapest change first.
- **D-07:** **Tier 1 (deterministic core):** on injected fixed bootstrap/feature indices, a single tree matches `sklearn.tree.DecisionTree*` — exact split structure + ≤1e-5 leaf values (f64). This is the real correctness witness (abort signal A5).
- **D-08:** **Tier 2 (ensemble/predictive band):** forests gated on predictive quality only (accuracy within ~0.02–0.05; R² within ~0.02–0.05). Tier 2 is *documented* this phase, not *run* (no forest exists yet; it governs Phase 19).
- **D-09:** Validate **both** `DecisionTreeClassifier(criterion='gini')` **and** `DecisionTreeRegressor(criterion='squared_error')` on injected fixed indices. `entropy`/`log_loss`/`absolute_error` deferred to Phase 18/19.
- **D-10:** Compute **quantile bin edges on the host once per fit** (A2 mitigation); device kernels consume precomputed edges only — **no on-device sort/scan**. Default **128 bins**; the spike benchmarks **64 vs 128** so the D-06 "fewer bins" lever has data.

### Claude's Discretion
- Exact relabel-partition kernel mechanics, benchmark-harness plumbing, fixture seeding details, and the rocm f32 skip-with-log wiring follow established Phase 13 / `cpu-mlir-kernel-authoring` conventions — planner/researcher's call within the locked op-set.

### Deferred Ideas (OUT OF SCOPE)
- **Multiclass leaf-value-buffer population path** — format contract (offset semantics, D-04) is *defined* this spike; populating/consuming for >2 classes is Phase 19+.
- **`entropy`/`log_loss`/`absolute_error` criteria** — witness covers gini + squared_error only (D-09).
- **RNG-driven bootstrap/feature sampling** — spike injects fixed indices; SplitMix64 sampling is Phase 19 (RF).
- **Production prims** (`quantiles`/`tree_hist`/`best_split`/`node_partition`) — Phase 18 promotes from findings.
- **On-device bin-edge computation** — host pre-pass chosen (D-10); revisit only if A2 mitigation proves insufficient.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TREE-01 | A GPU tree-construction **feasibility spike** proves (or refutes) that a single-owner GATHER histogram/split lowers and is tractable under cpu-MLIR — no SharedMemory, no atomics, no `F::INFINITY` init. Delivers GATHER-histogram + relabel-partition + seed-from-first split-find kernels standalone-launching on cpu(f64)+rocm(f32), a VALUE-asserting correctness test vs `sklearn.tree.DecisionTree*` on injected fixed bootstrap/feature indices, a per-tree cost benchmark, a finalized `SparseTreeNode { colid, threshold, left_child, value }` (right = left+1) format contract, an established two-tier stochastic-gate convention, and an explicit **GO / ADJUST / ABORT** verdict with abort signals A1–A5 evaluated. *(gate: spike verdict + VALUE on fixed-index tree)* | The three-kernel mapping (§Architecture Patterns), the Tier-1 witness mechanics (§Tier-1 Correctness Witness), the SparseTreeNode contract (§SparseTreeNode Contract), the A1–A5 evaluation method (§A1–A5 Abort-Signal Evaluation), and the benchmark/skip-with-log plumbing (§Benchmark Harness) collectively give the planner everything needed to author the spike. |
</phase_requirements>

## Summary

This is a **feasibility spike**, not a build phase. Its job is to answer one make-or-break question — *does GPU decision-tree construction lower and run tractably under `cubecl-cpu` (the f64 correctness gate) using a SharedMemory-free / atomic-free / `F::INFINITY`-free design?* — and to deliver an explicit **GO / ADJUST / ABORT** verdict that gates Phases 18–21 (tree prims → RF → FIL → TreeSHAP). It mirrors Phase 13 (KNN-graph keystone) exactly: throwaway probes run as self-contained live-launch tests in `crates/mlrs-backend/tests/`, the proven kernel+harness source is copied verbatim into `.planning/spikes/NNN-*/`, and a `VERDICT.md` is the gating artifact. [CITED: 17-CONTEXT.md D-01]

The good news, grounded in `spike-findings-mlrs`: **every op the three tree kernels need is already launch-proven under cpu-MLIR.** The GATHER histogram is the same single-owner feature-loop accumulator shipped in `distance.rs` (manhattan/chebyshev) and `self_drop_gather`. The seed-from-first split-find argmax is *literally already implemented* — `topk::select_k` does running-best argmax with NO `F::INFINITY` sentinel and NO mutable-bool flag, exactly the idiom the gain argmax needs. The relabel-partition is a per-sample one-unit GATHER (`CUBE_POS_X` per sample, overwrite own `node_id`), structurally identical to the per-row `self_drop_gather` launch shape. The spike's residual risk is therefore **not** "will it compile" (A1, A4 are low-risk) but **A3 — is the `O(samples × bins)` GATHER cost tractable**, which can only be measured, not reasoned away. [VERIFIED: codebase grep of `crates/mlrs-kernels/src/{distance,topk}.rs`]

**Primary recommendation:** Author all three kernels strictly inside the proven op-set (§Don't Hand-Roll lists the banned constructs), model the split-find argmax directly on `select_k`'s seed-from-candidate-0 pattern, model the histogram on the `manhattan_dist` feature-loop accumulator, and the relabel on the `self_drop_gather` per-row shape. Make the Tier-1 witness assert VALUES (not non-panic) against committed sklearn `DecisionTreeClassifier(gini)` + `DecisionTreeRegressor(squared_error)` fixtures with injected fixed indices, including at least one adversarial/degenerate fixture (the 002-B silent-miscompile discipline). Benchmark 64 vs 128 bins on ≈1000×20×depth-8 to give the D-06 ladder data. Deliver `VERDICT.md` with all five A1–A5 signals explicitly evaluated.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GATHER histogram (per-(node,feature,bin) counts/sums) | `mlrs-kernels` device kernel | — | Pure compute; one `#[cube(launch)]` kernel; must lower cpu-MLIR-safe. The whole feasibility question lives here. |
| Per-level orchestration (frontier loop, launch sequencing, split read-back) | Host (test harness this phase; `mlrs-backend` prim Phase 18) | — | Data-dependent tree depth/frontier is host control flow; kernels cannot recurse (CubeCL). Pitfall 1 step 4. [CITED: PITFALLS.md] |
| Quantile bin-edge computation | Host pre-pass (numpy/sklearn fixture + host Rust at fit time) | — | D-10 — A2 mitigation; NO on-device sort/scan. |
| relabel-partition (sample → child `node_id`) | `mlrs-kernels` device kernel | — | Per-sample GATHER; overwrites own label; no scan/compaction. Pitfall 1 step 2. |
| seed-from-first split-find argmax (best gain over bins) | `mlrs-kernels` device kernel | — | Running-best statement-form `if`; no `F::INFINITY`. Pitfall 1 step 3; idiom proven in `topk::select_k`. |
| Tier-1 correctness witness (VALUE-assert vs sklearn) | Host test (`crates/mlrs-backend/tests/` + committed `.npz` fixtures) | `mlrs-core::oracle` loader | Oracle comparison; sklearn fixtures committed; no Python at test time. |
| `SparseTreeNode` format contract | Plain Rust struct (defined this phase; consumed Phase 18/20/21) | — | The single load-bearing interface binding RF (writer) / FIL (reader) / TreeSHAP (reader). |
| f64-on-rocm skip-with-log | `mlrs-backend::capability::skip_f64_with_log` (shipped) | — | rocm has no f64 (project memory); skip-not-fail with a logged reason. |

## Standard Stack

**Zero new compute dependencies.** The spike consumes only what is already pinned in the workspace. The whole cpu-MLIR safety story is characterized at `cubecl = 0.10.0`; bumping it invalidates the feasibility verdict. [CITED: research/SUMMARY.md "Hard pins"]

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` (+ `cubecl-cpu` MLIR) | 0.10.0 (HARD PIN) | `#[cube(launch)]` kernels; the f64 cpu-MLIR correctness gate | Entire op-set safety story is pinned here; the project's only device-compute layer. [VERIFIED: workspace Cargo.toml pin per SUMMARY.md] |
| `bytemuck` | workspace | host↔device byte casts (`cast_slice`, `from_bytes`) for read-back + the host-`f64` view of `F` | Shipped idiom in every spike/launch test. [VERIFIED: `tests/self_drop_gather_test.rs`] |
| `npyz` | workspace | read committed `.npz` sklearn fixtures (`mlrs-core::oracle::load_npz`); also has a writer for fixture generation w/o numpy | Shipped oracle loader; `by_name` resolved in spike A4. [VERIFIED: `crates/mlrs-core/src/oracle.rs`] |
| `scikit-learn` | ≥1.6 (already pinned; **test-only oracle**) | generate `DecisionTreeClassifier(gini)` / `DecisionTreeRegressor(squared_error)` reference fixtures | sklearn is the milestone oracle; ≥1.6 covers `sklearn.tree`. No version bump. [CITED: REQUIREMENTS.md milestone note; SUMMARY.md] |
| `numpy` | via `/tmp` venv (PEP 668) | fixture generation only (committed blobs; not in the test path) | Project memory: oracle fixtures are committed; regen needs a venv. [CITED: MEMORY oracle-fixture-regen-needs-venv] |

### Supporting (existing in-tree assets — clone, do not re-derive)
| Asset | Path | Purpose | When to Use |
|-------|------|---------|-------------|
| live-launch harness | `crates/mlrs-backend/tests/spike_test.rs` | `runtime::active_client()`, `Bytes::from_elems` upload, `client.empty`, `read_one`+`cast_slice`, by-value scalar args, `from_raw_parts(handle,len)` | Clone its shape for every tree-kernel probe. [VERIFIED] |
| GATHER feature-loop kernel | `crates/mlrs-kernels/src/distance.rs` (`manhattan_dist`, `chebyshev_dist`) | proven `while kk<cols { acc += … }` same-iteration accumulator + statement-form running-max | Model the histogram inner loop + any per-cell max on these. [VERIFIED] |
| seed-from-first argmax kernel | `crates/mlrs-kernels/src/topk.rs` (`select_k`) | running-best argmax with NO `F::INFINITY`, NO mutable-bool, seeded from candidate 0 | **This IS the split-find argmax idiom.** Model `best_split` on it directly. [VERIFIED] |
| per-row GATHER launch shape | `crates/mlrs-kernels/src/distance.rs` (`self_drop_gather`) + `tests/self_drop_gather_test.rs` | `CUBE_POS_X`/`UNIT_POS_X==0` per-row, nested same-iteration count (002-B-safe), all-zeros launch guard | Model the relabel-partition per-sample shape + the "did it actually launch" assertion on this. [VERIFIED] |
| f64 skip-with-log | `crates/mlrs-backend/src/capability.rs` (`skip_f64_with_log`, `active_backend_name`, `log_oracle_dtype`) | f64-on-rocm early-return-with-warn; backend/dtype CI log line | Gate every f64 probe; f32 always runs. [VERIFIED] |
| oracle loader | `crates/mlrs-core/src/oracle.rs` (`load_npz`, `expect_f64`/`expect_f32`, `shape`) | load committed sklearn `.npz` fixtures by name at test time, no Python | Tier-1 witness comparison. [VERIFIED] |
| fixture generator pattern | `scripts/gen_oracle.py` (50+ `gen_*` fns; `c()` cast helper; `main()` dispatch; writes to `tests/fixtures/`) | add `gen_decision_tree_clf()` / `gen_decision_tree_reg()` emitting injected-index fixtures | Mirror existing `gen_*` fns; **encode any tie-break in the generator, not by hand-patching the blob** (Phase-13 CR-01 lesson). [VERIFIED] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single-owner GATHER histogram | `SharedMemory` + `atomicAdd` (cuML/XGBoost textbook) | **Banned** — panics at launch on cpu-MLIR; the entire reason the spike exists. Never. [CITED: PITFALLS.md Pitfall 1] |
| relabel `node_id` array | prefix-sum/scan compaction (physically reorder samples) | Scan has no SharedMemory-free, atomic-free expression under cpu-MLIR. Relabel avoids it entirely. [CITED: PITFALLS.md Pitfall 1 step 2] |
| Host quantile bin edges | on-device percentile/sort | A2 abort risk; D-10 chose host pre-pass ("almost certainly acceptable"). Revisit only if A2 fires. |
| seed-from-candidate-0 argmax | `best_gain = -F::INFINITY` init | `F::INFINITY` banned (panics at launch). `select_k` already proves the seed-from-first alternative. [VERIFIED] |

**Installation:** No new crates. Fixture regen (one-time, blobs committed):
```bash
python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scikit-learn
/tmp/oracle-venv/bin/python scripts/gen_oracle.py decision_tree   # add the dispatch key
```
Probe run vehicle:
```bash
cargo test -p mlrs-backend --features cpu --test <spike_test_file> -- --nocapture
```

## Package Legitimacy Audit

> This phase installs **no new external packages**. All dependencies (`cubecl` 0.10.0, `bytemuck`, `npyz`, and the test-only `scikit-learn`≥1.6 / `numpy` oracle) are already present in the workspace / shipped fixture pipeline and were legitimacy-cleared in prior milestones.

| Package | Registry | Disposition |
|---------|----------|-------------|
| cubecl 0.10.0 | crates.io (hard-pinned, pre-existing) | Approved — no change |
| bytemuck / npyz | crates.io (pre-existing workspace deps) | Approved — no change |
| scikit-learn ≥1.6 / numpy | PyPI (test-only oracle, pre-existing) | Approved — no change |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none
*No `package-legitimacy check` run necessary — no additive packages introduced this phase (zero-new-compute-dependency record, per SUMMARY.md).*

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────── HOST (test harness this phase; mlrs-backend prim Phase 18) ───────────────────────┐
                          │                                                                                                            │
  fixed bootstrap idx ───►│  bootstrap-gather X' = X[bootstrap_idx], y' = y[bootstrap_idx]  (injected, NOT RNG — D-07)               │
  fixed feature  idx ───► │  feature subset cols = feature_idx                                                                        │
  X (n×F), y          ───►│  quantile bin edges  (host pre-pass, D-10 / A2 mitigation — NO on-device sort)  ──► binned X'' (u32 bins) │
                          │                                                                                                            │
                          │   node_id[] all = 0 (root)                                                                                 │
                          │      │                                                                                                     │
                          │      ▼   per tree level (host while loop, bounded by max_depth — kernels can't recurse):                  │
                          │   ┌──────────────────────────────────────────────────────────────────────────────────────┐             │
   DEVICE (cpu-MLIR / ───────►│ (1) GATHER histogram kernel                                                            │             │
   rocm f32; f64 skips)   │   │     one unit per (node,feature,bin); loop samples; if node_id==node && bin==b: acc+=1  │ ──► hist    │
                          │   │     [model: distance.rs manhattan feature-loop accumulator]                            │  (counts/   │
                          │   │                                                                                          │   sums)     │
                          │   │ (2) seed-from-first split-find kernel                                                   │             │
                          │   │     running-best gain argmax over bins; seed from candidate 0; NO F::INFINITY          │ ──► best    │
                          │   │     [model: topk.rs select_k running-best, statement-form if]                          │  (col,bin,  │
                          │   │                                                                                          │   gain)     │
                          │   │ (3) relabel-partition kernel                                                           │             │
                          │   │     one unit per sample; read split for own node; overwrite own node_id := child       │ ──► node_id │
                          │   │     [model: self_drop_gather per-row CUBE_POS_X shape; NO scan/compaction]             │  (mutated)  │
                          │   └──────────────────────────────────────────────────────────────────────────────────────┘             │
                          │      │  read back chosen splits → append SparseTreeNode{colid,threshold,left_child,value}                 │
                          │      ▼  (leaf when pure / max_depth / min_samples → colid=-1, value=leaf-buffer offset)                   │
                          │   flat node array  +  shared leaf-value buffer                                                            │
                          └──────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                                          ▼
                  Tier-1 VALUE assert: split structure EXACT vs sklearn; leaf values ≤1e-5 (f64)   ◄── committed sklearn .npz fixture
                                                                          ▼
                                              cost benchmark (64 vs 128 bins) + A1–A5  ──►  VERDICT.md (GO / ADJUST / ABORT)
```
File-to-implementation mapping is in the Component Responsibilities of the Responsibility Map above; the diagram shows data flow only.

### Recommended Project Structure (spike layout — D-01, mirrors Phase 13)
```
crates/mlrs-backend/tests/
├── tree_spike_NNN_*.rs       # throwaway live-launch probes (deleted after wrap-up;
│                             #   copied verbatim into .planning/spikes/NNN-*/)
.planning/spikes/
├── MANIFEST.md               # append the tree spikes: idea, requirements, spike table
├── CONVENTIONS.md            # (already carries the cpu-MLIR op-set; reuse)
├── NNN-gather-histogram-lower/        # durable verbatim source artifact
├── NNN-seed-from-first-split-find/
├── NNN-relabel-partition/
└── NNN-tier1-decisiontree-witness/
.planning/phases/17-.../
└── VERDICT.md                # GO/ADJUST/ABORT — the gating artifact (D-01)
scripts/gen_oracle.py         # + gen_decision_tree_clf/reg (injected-index fixtures)
tests/fixtures/
└── tree_dt_{clf,reg}_{f32,f64}_seed42.npz   # committed sklearn reference blobs
```

### Pattern 1: Single-owner GATHER histogram (one unit per `(node,feature,bin)`)
**What:** Each output cell `(node, feature, bin)` is owned by exactly one unit; that unit loops over the node's sample range and conditionally accumulates. One writer per cell ⇒ no atomics, no contention. The price is `O(samples × bins)` per `(node,feature)` — this is what A3 measures. [CITED: PITFALLS.md Pitfall 1 step 1]
**When to use:** All histogram accumulation. Never the scatter-add-with-atomics form.
**Example (model directly on the shipped `manhattan_dist` accumulator):**
```rust
// Source: crates/mlrs-kernels/src/distance.rs (manhattan_dist) — the proven
// same-iteration feature-loop accumulator. The histogram is the same shape with
// a node_id + bin filter instead of an abs-diff sum. Launch one unit per output
// cell; index it with ABSOLUTE_POS_X/Y or a linearized CUBE_POS_X (the topk shape).
let cell = ABSOLUTE_POS_X;            // = (node*F + feature)*bins + bin   (or 2D)
if cell < n_cells {
    let mut acc = F::from_int(0i64);  // u32 for counts; F for gradient/value sums
    let mut s = 0u32;
    while s < n_samples {             // runtime while loop, same-iteration accumulate
        // node filter AND bin filter, both read in THIS iteration (no sibling loop)
        if node_id[s as usize] == my_node {
            if binned[(s * n_feat + my_feature) as usize] == my_bin {
                acc += F::from_int(1i64);          // count; or += value[s] for sums
            }
        }
        s += 1u32;
    }
    hist[cell as usize] = acc;
}
```

### Pattern 2: Seed-from-first split-find argmax (NO `F::INFINITY`)
**What:** Scan candidate (feature,bin) split points for the maximum information gain. Seed the running best from the FIRST candidate (candidate 0), then update with statement-form `if` — never `best = -INFINITY`, never an `if`-expression in value position. [CITED: PITFALLS.md Pitfall 1 step 3 / A4]
**When to use:** The gain argmax. This idiom is **already shipped** in `select_k`.
**Example (the existing `select_k` running-best is the template):**
```rust
// Source: crates/mlrs-kernels/src/topk.rs (select_k) — running-best argmax with
// NO F::INFINITY sentinel and NO mutable-bool flag: seed best from candidate 0,
// update only when a later candidate strictly precedes it. The split-find gain
// argmax is structurally identical (maximize gain instead of minimize distance).
let mut best_gain = gain[base as usize];   // SEED from candidate 0 (not -INF)
let mut best_col  = 0u32;
let mut best_bin  = 0u32;
let mut c = 1u32;
while c < n_candidates {
    let g = gain[(base + c) as usize];
    if g > best_gain {                     // statement-form if; lowest-index tie via strict >
        best_gain = g;
        best_col  = col_of[c];
        best_bin  = bin_of[c];
    }
    c += 1u32;
}
```

### Pattern 3: Relabel-partition (per-sample GATHER, no scan/compaction)
**What:** Keep a `sample → node_id` label array. Each level, each sample (owned by one unit) reads the split for its *current* node and overwrites its own `node_id` with the child id (`left_child` or `left_child + 1`). Pure GATHER — samples are never physically reordered, so no prefix-sum. [CITED: PITFALLS.md Pitfall 1 step 2 — "the single insight that makes RF feasible under cpu-MLIR"]
**When to use:** All node partitioning. Never a compaction/scan.
**Example (model on the `self_drop_gather` per-row launch shape):**
```rust
// Source: crates/mlrs-kernels/src/distance.rs (self_drop_gather) launch shape —
// CUBE_POS_X per row, UNIT_POS_X==0 guard (NEVER bare-ABSOLUTE_POS 1D → 002-A).
let s = CUBE_POS_X;                         // one cube per SAMPLE
if s < n_samples {
    if UNIT_POS_X == 0u32 {
        let nid = node_id[s as usize];
        // read this node's split (colid, threshold encoded as bin) from frontier arrays
        let col = split_col[nid as usize];
        let thr = split_bin[nid as usize];
        // statement-form branch → write own child label; right child = left_child+1 (D-02)
        let mut child = left_child[nid as usize];          // go-left default
        if binned[(s * n_feat + col) as usize] > thr {
            child = left_child[nid as usize] + 1i32;       // go-right (D-02)
        }
        node_id[s as usize] = child;
    }
}
```

### Anti-Patterns to Avoid
- **`SharedMemory` histogram "just for the spike":** invalidates the entire feasibility verdict; the spike's whole purpose is to prove the SharedMemory-free path. Never. [CITED: PITFALLS.md tech-debt table]
- **Bare-`ABSOLUTE_POS` 1D launch for a per-row/per-sample loop kernel:** loud MLIR pass failure ("operation with block successors must terminate its parent block"); reads back all zeros. Use `CUBE_POS_X`/`UNIT_POS_X==0`. **FINDING 002-A.** [CITED: spike-findings cpu-mlir-kernel-authoring]
- **Cross-sibling-loop mutable accumulator:** a flag/counter written in one `while` and read in a SEPARATE sibling `while` SILENTLY miscompiles. Recompute per-cell positional values with a self-contained nested accumulate in the SAME outer iteration. **FINDING 002-B.** [CITED: spike-findings]
- **`F::INFINITY` argmax init / `if`-expression in value position / mutable-bool scan / descending-shift loop:** banned (panic at launch). [CITED: spike-findings banned list]
- **Hand-patching a fixture blob to match the prim's tie-break:** Phase-13 CR-01/CR-02 — the committed fixture must be reproducible from the committed generator, and the tie-break must be an *independent* rule in the generator, or the gate goes circular. [CITED: 13-VERIFICATION.md]
- **Asserting non-panic instead of VALUES:** a silent miscompile (002-B) compiles, launches, and returns plausible-wrong data. Always assert returned values against the oracle. [CITED: spike-findings R-9]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Histogram accumulation | `SharedMemory` + `atomicAdd` scatter | single-owner GATHER (Pattern 1) | atomics/SharedMemory panic on cpu-MLIR; the spike exists to avoid this |
| Best-gain argmax | `best = -F::INFINITY` init loop | seed-from-candidate-0 (Pattern 2, = `select_k`) | `F::INFINITY` banned; the alternative is already shipped |
| Node partition | prefix-sum/scan + sample reorder | relabel `node_id` GATHER (Pattern 3) | scan has no atomic-free, SharedMemory-free cpu-MLIR form |
| Quantile bin edges | on-device sort/percentile | host pre-pass once per fit (D-10) | on-device sort risks A2 abort |
| `.npz` fixture read | zip + npy-header parser | `mlrs_core::oracle::load_npz` (`npyz` `by_name`) | shipped, resolved in spike A4 |
| Live kernel launch boilerplate | new launch scaffolding | clone `spike_test.rs` / `self_drop_gather_test.rs` | upload/empty/read-back/by-value-scalar idiom is proven |
| f64-on-rocm handling | ad-hoc cfg gating | `capability::skip_f64_with_log()` | shipped skip-with-log; logs the reason in CI |
| "did the kernel actually run?" check | trusting a green non-panic | all-zeros read-back guard (002-A symptom) | a kernel that never launched reads back zeros |

**Key insight:** The three tree kernels are *recompositions of already-proven idioms*, not new lowering risk. The histogram = `manhattan_dist`'s accumulator + filters; the split-find = `select_k`'s running-best; the relabel = `self_drop_gather`'s per-row shape. The genuinely-unknown quantity is **cost** (A3), not compilability.

## Runtime State Inventory

> N/A — this is a greenfield spike (new throwaway kernels + new fixtures + a verdict doc). It renames nothing, migrates no stored data, registers no OS/service state, and writes no production code. Section omitted by design.

## SparseTreeNode Contract (D-02 / D-03 / D-04 — load-bearing for Phases 18/20/21)

The format the spike **defines and validates**:

```rust
// colid: i32  — split feature column; SENTINEL colid = -1 marks a LEAF (D-03).
//               FIL's iterative traversal stops on colid < 0 (treelite/cuML-FIL convention).
// threshold: F — split threshold (the real-valued edge for `colid`'s bin cut).
// left_child: i32 — index of the left child in the flat node array.
//               RIGHT child is implicit: right = left_child + 1 (D-02, locked by TREE-01).
// value: i32/usize — NOT a scalar prediction. An OFFSET/INDEX into a shared
//               leaf-value buffer (D-04). Multiclass-uniform from day one:
//               binary, multiclass, and regression leaves all index a side buffer.
struct SparseTreeNode<F> { colid: i32, threshold: F, left_child: i32, value: i32 }
```

**Validation method this phase (binary clf + regression witness, D-09):**
1. **`right = left_child + 1` (D-02):** assert in the witness that the regenerated tree's right child is always `left_child + 1` — i.e. the node array is laid out so a parent's two children are adjacent. Cross-check by traversing the built tree against the sklearn split structure.
2. **Leaf sentinel `colid = -1` (D-03):** assert every leaf node has `colid == -1` and every internal node has `colid >= 0`. A host reference walk must terminate exactly on `colid < 0`.
3. **`value` is an offset, not a prediction (D-04):** the witness must dereference `value` into the shared leaf-value buffer and compare *that* to sklearn's leaf output. For the **classifier** leaf path: `value` indexes class-probability entries (binary → the buffer slot holds the positive-class probability / both-class probs). For the **regressor** leaf path: `value` indexes the regression-mean entry. Exercising BOTH (D-09) proves the offset semantics carry both leaf-value shapes through ONE uniform field — the multiclass-uniformity claim. Assert `≤1e-5` on the dereferenced f64 leaf values; assert EXACT on the structural fields (`colid`, `left_child`, the chosen split bin).

**cuML reference divergence (note for the planner):** cuML's `flatnode.h` `SparseTreeNode` marks a leaf with `left_child_id == -1` and keeps `colid` as a plain split column; `RightChildId() = left_child_id + 1`. [VERIFIED: cuml-main/cpp/include/cuml/tree/flatnode.h] mlrs's D-03 deliberately diverges to the **`colid = -1`** leaf sentinel (treelite/FIL iterative-traversal convention) so FIL stops on `colid < 0`. The `right = left+1` rule is shared. The spike should record this divergence explicitly in `VERDICT.md` / findings so Phase 20 FIL binds to the mlrs convention, not cuML's.

## Tier-1 Correctness Witness (D-07 / D-09 — the real correctness gate, = abort signal A5)

**Goal:** prove the histogram/gain/partition MATH is correct (not just that kernels launch) by reproducing `sklearn.tree.DecisionTree*` EXACTLY on injected fixed indices.

**Determinism recipe (how to make a single tree match sklearn):**
1. **Inject fixed bootstrap indices + fixed feature subset** — do NOT draw from SplitMix64 (that is Phase 19). Hand `X[bootstrap_idx]`, `y[bootstrap_idx]`, and the feature column subset to BOTH the spike builder and sklearn. This removes RNG from the comparison, so element-wise match is achievable (the only place in the milestone where it is, for trees). [CITED: PITFALLS.md Pitfall 2 / D-07]
2. **Match sklearn's split semantics:** sklearn `DecisionTreeClassifier(criterion='gini')` and `DecisionTreeRegressor(criterion='squared_error')` with explicit `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` set so the injected feature subset is what's actually considered. Use mid-point thresholds consistent with the binning (the host quantile edges, D-10) — document any threshold-representation difference (binned cut vs sklearn's exact mid-point) and gate the *decision boundary*, not the raw float, where binning makes them differ.
3. **Tie-break independence (Phase-13 CR lesson):** if a gain tie can occur (two splits with equal gain), the generator must encode the SAME deterministic tie-break sklearn uses (lowest feature index, then lowest threshold) as an *independent* rule, and the witness should assert the boundary case explicitly. Do NOT conform the fixture to the prim's pick by hand. [CITED: 13-VERIFICATION.md CR-01/CR-02]

**Oracle/fixture mechanics:**
- Add `gen_decision_tree_clf(seed, dtype)` and `gen_decision_tree_reg(seed, dtype)` to `scripts/gen_oracle.py`, mirroring the existing `gen_*` fns (`from sklearn.tree import DecisionTree{Classifier,Regressor}`; the `c()` cast helper; write to `tests/fixtures/`). Emit named arrays: `X`, `y`, `bootstrap_idx`, `feature_idx`, plus the sklearn tree's `children_left`, `children_right`, `feature`, `threshold`, `value` (the sklearn `tree_` attributes) so the witness can compare structure AND leaf values. Generate both f32 and f64. [VERIFIED: gen_oracle.py structure]
- Fixtures are **committed `.npz` blobs**; regen needs a `/tmp` venv (numpy+sklearn; PEP 668). No Python at test time — the witness loads via `mlrs_core::oracle::load_npz`. [CITED: MEMORY oracle-fixture-regen-needs-venv]
- **VALUE-assert, not non-panic (002-B discipline / R-9):** assert (a) exact split structure (`colid`/feature per node, `left_child`, leaf sentinel), and (b) `≤1e-5` (f64) on dereferenced leaf values. Use `mlrs_core::compare::assert_slice_close` with `F64_TOL` (1e-5/1e-5) for leaf values. [VERIFIED: compare_test.rs, oracle.rs]
- **Adversarial/degenerate fixture (silent-miscompile backstop):** include at least one fixture with a degenerate case the happy path would miss — e.g. a node that must become a pure leaf (all one class) and a tie in gain (two equally-good splits). This is the histogram analogue of Phase 13's duplicate-point row. [CITED: PITFALLS.md security table — "VALUE-asserting oracle on an adversarial fixture"]

## A1–A5 Abort-Signal Evaluation (each MUST be explicitly answered in VERDICT.md)

| Signal | What it asks | Concrete evidence the spike must produce | Expected (per research) |
|--------|--------------|------------------------------------------|--------------------------|
| **A1 — Lowering fails** | Does the GATHER histogram kernel (nested sample-loop with node_id + bin filters) trip an MLIR pass failure no statement-form/loop rewrite resolves? | Probe launches the histogram kernel under `--features cpu` (f64) and asserts a **non-zero, correct** read-back (not the 002-A all-zeros symptom). | LOW risk — feature-loop accumulators are spike-001-proven. Gating compile check. |
| **A2 — Binning needs unlowering sort** | Do bin edges require a device sort/percentile that won't lower AND can't move to host? | Confirm the **host quantile bin-edge pre-pass** (D-10) produces edges the device kernels merely *consume* (binned `u32` input); show NO sort/scan kernel exists in the spike. | MITIGATED — host pre-pass "almost certainly acceptable" (D-10/PITFALLS). Abort only if binning must be on-device AND needs scan/sort. |
| **A3 — Correct but superlinear cost** | Is the `O(samples × bins)` GATHER tractable on a realistic load? | **Benchmark per-tree build on ≈1000×20×depth-8 at 64 AND 128 bins** (D-05/D-10); report wall-clock and the scaling shape as samples/bins grow. Judge **sub-quadratic in samples/bins**, not an absolute ceiling. | UNKNOWN — the one real measurement. A3 fires only on clearly-impractical cost (multi-second/tree) or pathological bin-scaling. Feeds the D-06 "fewer bins" lever. |
| **A4 — Split-find can't argmax safely** | Can the gain argmax be expressed without `F::INFINITY` or a cross-sibling-loop accumulator? | VALUE-assert the seed-from-first split-find kernel picks the correct best (col,bin,gain) on a fixture with a known answer (incl. a tie). | LOW risk — seed-from-first is proven in `select_k`. Verify with VALUE assertion, not non-panic. |
| **A5 — Correctness fail vs sklearn** | Does a single tree on injected fixed indices reproduce sklearn's split + leaf values? | The full Tier-1 witness (above): exact split structure + ≤1e-5 f64 leaf values vs `DecisionTreeClassifier(gini)` AND `DecisionTreeRegressor(squared_error)`. | The real correctness witness. If it fails, the histogram/gain math is wrong (not the RNG). |

**Verdict logic (D-05/D-06):** GO if A1/A2/A4/A5 pass and A3 is tractable+sub-quadratic. ADJUST if A3 is borderline — apply the D-06 ladder in order (128→64 bins, backed by the 64-vs-128 benchmark → frontier-only histogramming → shallower default `max_depth` → defer as last resort). ABORT only if A1 or A5 cannot be resolved within the op-set, or A3 is pathological at every ladder rung.

## Benchmark Harness + f64-on-rocm Skip-with-Log

**Harness:** clone the `spike_test.rs` / `self_drop_gather_test.rs` shape — `runtime::active_client()`, `Bytes::from_elems` upload, `client.empty(n*size_of::<F>())`, `read_one` + `bytemuck::cast_slice`, by-value scalar args, `from_raw_parts(handle, len)` (consumes handle; clone before read-back). [VERIFIED: spike_test.rs lines 30-81]

**Cost measurement (A3):** time the host per-level loop driving the three kernels for a full depth-8 tree on ≈1000×20, at 64 and 128 bins. Keep it simple wall-clock (`std::time::Instant`) inside the probe with `--nocapture`; this is a feasibility *measurement*, not a Criterion micro-benchmark. Run targeted (`--test <file>`) — the full mlrs-backend cpu suite is slow and a full `cargo test --features cpu` can exhaust disk (project memory: `backend-test-suite-slow`, `full-cargo-test-exhausts-disk`). Report the 64-vs-128 delta so the D-06 bins lever is data-backed.

**Skip-with-log (rocm f32 gate):** every f64 probe starts with
```rust
// Source: crates/mlrs-backend/src/capability.rs + tests/self_drop_gather_test.rs
if capability::skip_f64_with_log() {
    println!("tree spike f64 backend={}: SKIPPED (no f64 on this adapter)",
             capability::active_backend_name());
    return;
}
```
f32 probes always run. The gate is **cpu(f64) + rocm(f32)**; f64-on-rocm SKIPS-with-log (rocm has no f64 — project memory `rocm-is-runnable-gpu-gate`). Emit `capability::log_oracle_dtype(...)` / a backend log line so CI shows which dtype ran on which backend. [VERIFIED: capability.rs, self_drop_gather_test.rs]

**cpu-MLIR landmine note (`cubecl-cpu-no-shared-memory` memory):** beyond the banned list, the cpu MLIR backend has historically panicked at launch on SharedMemory kernels using mutable bool / `F::INFINITY` / shift-loops. The three tree kernels are SharedMemory-free by design; keep them so.

## Common Pitfalls

### Pitfall 1: Reaching for atomics/SharedMemory because that's how every GPU tree builder works
**What goes wrong:** The textbook histogram scatter-adds into a `__shared__` array with `atomicAdd`; both panic at launch on cpu-MLIR. **Why it happens:** it's the canonical, invisible pattern (cuML's own). **How to avoid:** the GATHER inversion (Pattern 1) — one owner per cell, no contention. **Warning signs:** any urge to "just use a small SharedMemory scratch." [CITED: PITFALLS.md Pitfall 1]

### Pitfall 2: The kernel "passes" but never actually launched (002-A)
**What goes wrong:** a bare-`ABSOLUTE_POS` 1D launch silently never runs; output reads back all zeros and a non-panic check goes green. **Why it happens:** MLIR pass failure on the wrong launch shape. **How to avoid:** `CUBE_POS_X`/`UNIT_POS_X==0` shape + an explicit all-zeros read-back guard asserting a non-trivial value exists. **Warning signs:** zeros where you expect counts. [CITED: spike-findings 002-A]

### Pitfall 3: Silent miscompile returns plausible-wrong histogram (002-B)
**What goes wrong:** a counter written in one `while` and read in a sibling `while` never updates — compiles, launches, returns wrong-but-plausible data. **Why it happens:** the cube macro can't carry the cross-loop value. **How to avoid:** same-iteration nested accumulate; VALUE-assert on an adversarial fixture. **Warning signs:** a histogram correct on a toy node but off on a degenerate one. [CITED: spike-findings 002-B]

### Pitfall 4: Circular oracle — fixture conformed to the prim's own tie-break
**What goes wrong:** a hand-patched fixture matches the prim's pick, so a tie-break miscompile still passes; and re-running the generator reverts the blob. **Why it happens:** encoding the tie-break by hand instead of in the generator. **How to avoid:** encode the independent tie-break rule in `gen_oracle.py`; keep the committed blob reproducible from the committed generator; assert the boundary case explicitly. [CITED: 13-VERIFICATION.md CR-01/CR-02]

### Pitfall 5: A3 declared "fine" without a real benchmark
**What goes wrong:** the kernel is correct on a toy node, so cost is assumed fine; it explodes at 128 bins × depth 8 × 20 features on realistic n. **Why it happens:** `O(samples × bins)` looks cheap on toy fixtures. **How to avoid:** measure 64 vs 128 bins on the representative fixture and report the scaling shape; the verdict must cite the number, not "it compiled." [CITED: PITFALLS.md performance traps; D-05]

### Pitfall 6: Data-dependent memory blow-up (carried-in discipline)
**What goes wrong:** materializing the full `(nodes × features × bins × classes)` histogram or `2^max_depth` node arrays resident at once. **Why it happens:** "allocate the max." **How to avoid (for the spike's measurement honesty):** histogram only the active frontier; size node arrays to actual count. The build-failing PoolStats gate is a Phase 18 deliverable, but the spike should *note* whether the GATHER layout keeps scratch bounded by the frontier (feeds the D-06 "frontier-only" lever). [CITED: PITFALLS.md Pitfall 6]

## Code Examples

All load-bearing examples are inlined in §Architecture Patterns (Patterns 1–3) and §Benchmark Harness, each cited to its shipped source file (`distance.rs` manhattan accumulator, `topk.rs` select_k running-best, `self_drop_gather` per-row shape, `capability.rs` skip-with-log, `spike_test.rs` launch idiom). No additional examples needed — the spike's kernels are recompositions of these.

## State of the Art

| Old Approach | Current Approach (mlrs) | When Changed | Impact |
|--------------|-------------------------|--------------|--------|
| `__shared__` histogram + `atomicAdd` scatter (cuML/XGBoost) | single-owner GATHER, one writer per cell | this spike | only viable form under cpu-MLIR |
| prefix-sum/scan node partition | relabel `node_id` GATHER (no reorder) | this spike | avoids the unlowering scan |
| `best = -INFINITY` argmax init | seed-from-candidate-0 running best | shipped in `select_k` | `F::INFINITY` is banned |
| cuML leaf = `left_child_id == -1`, `colid` plain | mlrs leaf = `colid == -1` sentinel (D-03) | this spike | FIL stops on `colid < 0` (treelite convention) |
| `value` = scalar prediction | `value` = offset into shared leaf-value buffer (D-04) | this spike | multiclass-uniform format from day one |

**Deprecated/outdated:** the cuML CUDA histogram kernel (`builder_kernels_impl.cuh`) is a *reference for what NOT to port* — its atomic+SharedMemory design is the exact thing the GATHER inversion replaces. [CITED: SUMMARY.md sources]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A3 (GATHER cost) will be tractable + sub-quadratic at 128 bins | A1–A5 table | If wrong, the spike fires ADJUST/ABORT — but that IS the spike's job; the benchmark resolves it, so the "risk" is the verdict working as designed. |
| A2 | sklearn `DecisionTreeRegressor(squared_error)` leaf values and binned-threshold decision boundaries reconcile to ≤1e-5 given host quantile binning | Tier-1 Witness | If binning shifts a boundary, the witness must gate the decision (label/leaf) rather than the raw float threshold — note in the witness, not a blocker. |
| A3 | The `colid = -1` leaf sentinel (D-03) is fully compatible with a `left_child + 1` right-child layout (D-02) without ambiguity for leaves | SparseTreeNode Contract | Leaves carry no children, so `left_child` is unused for leaves; confirm the witness treats `colid < 0` as the sole leaf test. Low risk — locked decision. |

**If A1–A5 (the abort signals) are the spike's deliverable, these assumptions are resolved BY running the spike — that is the point of a feasibility phase.**

## Open Questions (RESOLVED — answered by the plans / spike execution)

> These are intentional EMPIRICAL unknowns that the spike itself answers, not ambiguities that block
> planning. Each recommendation is incorporated into the relevant plan; the spike run closes them.

1. **Threshold representation: binned cut vs sklearn's exact midpoint.**
   - What we know: D-10 bins on host quantile edges; sklearn splits on exact feature midpoints.
   - What's unclear: whether the witness should gate the raw `threshold` float or the resulting decision boundary when binning rounds it.
   - Recommendation: gate EXACT on structure (`colid`, `left_child`, leaf sentinel) and the chosen bin; gate the leaf VALUES ≤1e-5; for thresholds, assert the boundary classifies the same samples (decision-equivalence) rather than byte-equality. Document in the witness.
   - **RESOLVED:** incorporated into Plan 17-03 Task 1 — the witness uses decision-equivalence threshold gating (exact structure + ≤1e-5 leaf values). Final disposition recorded in VERDICT.md (Phase-18 caveat).

2. **Linearized cube indexing for the histogram (1D `ABSOLUTE_POS_X` over `n_cells` vs 2D).**
   - What we know: both the 2D `ABSOLUTE_POS_X/Y` (distance kernels) and the per-row `CUBE_POS_X` (topk/self_drop) shapes are proven.
   - What's unclear: which is cleanest for `(node,feature,bin)` cells.
   - Recommendation: a probe can try the 2D guarded shape first (proven in `manhattan_dist`); fall back to the `CUBE_POS_X` per-cell shape if the 2D guard mis-lowers. Either is in the op-set.
   - **RESOLVED:** incorporated into Plan 17-02 Task 1 — a live probe selects the cleaner indexing shape (both are in the proven op-set; the probe picks, the 002-A all-zeros guard catches a mis-lower). Result recorded in 17-02-SUMMARY.md.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust + cargo + `--features cpu` (cubecl-cpu MLIR) | all kernel probes (f64 gate) | ✓ (shipped, prior phases green) | cubecl 0.10.0 | — |
| `--features rocm` on gfx1100 | f32 GPU gate (opportunistic) | ✓ (project memory: runnable; f64 unsupported) | ROCm 7.1.1 | f64 SKIPS-with-log |
| numpy + scikit-learn (≥1.6) via `/tmp` venv | one-time fixture regen (PEP 668) | ✗ at system level; ✓ via venv | sklearn ≥1.6 | fixtures are committed blobs — only needed to regenerate |
| `npyz` (oracle `.npz` loader) | Tier-1 witness at test time (no Python) | ✓ (workspace dep) | workspace | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** numpy/sklearn — needed only to (re)generate the committed `.npz` fixtures via a `/tmp` venv; the test path itself needs no Python. [CITED: MEMORY oracle-fixture-regen-needs-venv]

## Validation Architecture

> `workflow.nyquist_validation: true` in config — section required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (live cubecl launch under `--features cpu`; oracle-fixture comparison vs sklearn) |
| Config file | none — Wave 0 adds the spike probe test file(s) + `tree_dt_{clf,reg}_{f32,f64}_seed42.npz` fixtures |
| Quick run command | `cargo test -p mlrs-backend --features cpu --test <tree_spike_file> -- --nocapture` |
| Full suite command | `cargo test -p mlrs-backend --features cpu` (slow — prefer targeted; full run can exhaust disk per project memory) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TREE-01 (SC-1, A1/A4) | GATHER-histogram + relabel-partition + seed-from-first split-find kernels standalone-LAUNCH on cpu(f64) + rocm(f32); no SharedMemory/atomics/`F::INFINITY`; non-zero correct read-back (002-A guard) | unit (live launch) | `cargo test -p mlrs-backend --features cpu --test <tree_spike_file>` | ❌ Wave 0 |
| TREE-01 (SC-2, A5) | single tree on injected fixed indices VALUE-matches `DecisionTreeClassifier(gini)` + `DecisionTreeRegressor(squared_error)` — exact split structure + ≤1e-5 f64 leaf values; incl. adversarial/degenerate + tie fixture | unit (oracle) | `cargo test -p mlrs-backend --features cpu --test <tree_witness_file>` | ❌ Wave 0 |
| TREE-01 (SC-3) | `SparseTreeNode{colid,threshold,left_child,value}` contract: `colid=-1` leaf sentinel, right=left+1, `value` dereferences into shared leaf buffer | unit (assertion within witness) | (same as SC-2) | ❌ Wave 0 |
| TREE-01 (SC-4, A3) | per-tree cost benchmark recorded at 64 AND 128 bins on ≈1000×20×depth-8; A1–A5 each evaluated | bench (wall-clock probe, `--nocapture`) | `cargo test -p mlrs-backend --features cpu --test <tree_bench_file> -- --nocapture` | ❌ Wave 0 |
| TREE-01 (SC-5) | GO/ADJUST/ABORT `VERDICT.md` delivered; two-tier stochastic-gate convention documented | manual (doc review of VERDICT.md) | manual | ❌ Wave 0 |

### Silent-Miscompile Backstops (the 002-B discipline — central to this phase)
- **VALUE assertions, never non-panic:** every kernel probe compares returned values to an in-test host oracle / committed sklearn fixture. A green "it didn't panic" is insufficient — 002-B compiled, launched, and returned wrong data.
- **All-zeros launch guard (002-A):** assert a non-trivial non-zero value is present in the read-back; a kernel that never launched reads back zeros.
- **Adversarial/degenerate fixture (property-style backstop):** at least one fixture with a gain tie + a forced-pure-leaf node — the histogram analogue of Phase 13's duplicate-point row — so a boundary miscompile cannot ship green. Tie-break encoded independently in the generator (Phase-13 CR lesson).
- **Both leaf-value paths (D-09):** classifier (probability leaf) AND regressor (mean leaf) through the one `value` offset field, proving the format carries both shapes.

### Oracle / Fixture Sampling
- Oracle = `sklearn.tree.DecisionTree{Classifier(gini),Regressor(squared_error)}`; fixtures generated by `scripts/gen_oracle.py` (new `gen_decision_tree_{clf,reg}`), committed as `.npz`, loaded at test time via `mlrs_core::oracle::load_npz` (no Python in the loop).
- Both f32 and f64 fixtures; f64 is the correctness gate (cpu), f64-on-rocm SKIPS-with-log.
- Tolerance: `mlrs_core::tolerance::F64_TOL` (abs 1e-5 / rel 1e-5) via `assert_slice_close` for leaf values; EXACT (`assert_eq`) for structural integer fields.

### Sampling Rate
- **Per task commit:** `cargo test -p mlrs-backend --features cpu --test <relevant tree spike file>` (targeted; <~few s each).
- **Per wave merge:** all tree spike + witness test files green (full suite backgrounded — it is slow / disk-heavy).
- **Phase gate:** all probe + witness files green AND `VERDICT.md` delivered with A1–A5 evaluated before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/tree_spike_*.rs` — live-launch probes for the three kernels (covers TREE-01 SC-1/A1/A4)
- [ ] `crates/mlrs-backend/tests/tree_witness_*.rs` — Tier-1 VALUE-assert vs sklearn (covers SC-2/SC-3/A5)
- [ ] `scripts/gen_oracle.py` — add `gen_decision_tree_clf` / `gen_decision_tree_reg` (injected-index, independent tie-break)
- [ ] `tests/fixtures/tree_dt_{clf,reg}_{f32,f64}_seed42.npz` — committed sklearn reference blobs (incl. adversarial/tie fixture)
- [ ] benchmark probe (64 vs 128 bins, ≈1000×20×depth-8)
- [ ] `VERDICT.md` skeleton (GO/ADJUST/ABORT + A1–A5 + two-tier convention)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a throwaway spike with device launches and no auth/session/network/crypto surface; the applicable controls are device-launch input validation only.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate kernel launch geometry (rows/cols/bins/k, `u32`-overflow) BEFORE any `unsafe` launch — the shipped `distance.rs`/`topk.rs::validate_geometry` precedent. Pass only validated element counts to `ArrayArg::from_raw_parts`. |
| V6 Cryptography | no | — |

### Known Threat Patterns for cubecl device launches
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unvalidated tree dims (n_bins, depth, n_samples) before `unsafe` launch | Tampering / DoS | host-side typed validation returning `PrimError` before launch (the `topk::validate_geometry` pattern); overflow-`u32` dim checks |
| `ArrayArg::from_raw_parts` length mismatch on histogram/node buffers | Tampering | pass validated element counts only; kernels bounds-check every index (`if cell < n_cells`, `if s < n_samples`) |
| Silent kernel miscompile returns plausible-wrong histogram | Information integrity (Repudiation) | VALUE-asserting oracle on an adversarial fixture (not non-panic) — the 002-B lesson applied to histograms |

## Sources

### Primary (HIGH confidence)
- `Skill("spike-findings-mlrs")` — `references/cpu-mlir-kernel-authoring.md` (proven op-set; 002-A loud launch failure; 002-B silent cross-loop miscompile; banned `SharedMemory`/`Atomic`/`F::INFINITY`/mutable-bool/shift-loops), `references/knn-graph-primitive.md` (single-owner GATHER idiom; VALUE-assert on adversarial fixture)
- `crates/mlrs-kernels/src/distance.rs` — `manhattan_dist`/`chebyshev_dist` feature-loop accumulator + `self_drop_gather` per-row GATHER (the histogram + relabel shapes)
- `crates/mlrs-kernels/src/topk.rs` — `select_k` seed-from-candidate-0 running-best argmax (the split-find shape, NO `F::INFINITY`)
- `crates/mlrs-backend/tests/spike_test.rs` + `tests/self_drop_gather_test.rs` — live-launch harness + all-zeros (002-A) guard
- `crates/mlrs-backend/src/capability.rs` — `skip_f64_with_log` / `active_backend_name` / `log_oracle_dtype` (rocm f32 gate)
- `crates/mlrs-core/src/oracle.rs` + `tests/compare_test.rs` + `examples/gen_fixture.rs` + `scripts/gen_oracle.py` — committed-`.npz` sklearn-oracle fixture pipeline
- `cuml-main/cpp/include/cuml/tree/flatnode.h` — cuML `SparseTreeNode` reference (leaf = `left_child_id==-1`, `right=left+1`) — the divergence point for D-03
- `.planning/research/PITFALLS.md` §Pitfall 1/2 + abort signals A1–A5; `.planning/research/SUMMARY.md` §Phase 17 (GATHER rationale)
- `.planning/phases/17-.../17-CONTEXT.md` (D-01..D-10); `.planning/REQUIREMENTS.md` §TREE-01
- `.planning/milestones/v3.0-phases/13-knn-graph-primitive-feasibility-keystone/` (13-VERIFICATION.md CR-01/CR-02 oracle-reproducibility lesson; 13-VALIDATION.md structure); `.planning/spikes/{MANIFEST,CONVENTIONS,WRAP-UP-SUMMARY}.md`

### Secondary (MEDIUM confidence)
- Project `MEMORY.md` — `cubecl-cpu-no-shared-memory`, `rocm-is-runnable-gpu-gate` (f64 unsupported on rocm), `oracle-fixture-regen-needs-venv`, `backend-test-suite-slow`, `full-cargo-test-exhausts-disk`, `knn-oracle-tiebreak-needs-overfetch`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new deps; every asset verified by codebase grep
- Architecture (3-kernel mapping): HIGH — each kernel maps to a shipped, VALIDATED idiom (`distance.rs`/`topk.rs`/`self_drop_gather`)
- Pitfalls: HIGH — grounded in spike-findings 002-A/002-B and Phase-13 CR-01/CR-02 (both documented in-repo)
- A3 cost tractability: MEDIUM — the one genuine unknown, resolvable only by the spike's own benchmark (by design)

**Research date:** 2026-06-27
**Valid until:** 2026-07-27 (stable — pinned cubecl 0.10.0; op-set proven; revisit only if the cubecl pin moves)
</content>
</invoke>
