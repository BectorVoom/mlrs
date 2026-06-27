---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
plan: 03
subsystem: tree-correctness-witness
tags: [tree, randomforest, oracle, sklearn, witness, TREE-01, spike, nyquist-wave-2, A5]
requires:
  - crates/mlrs-backend/tests/tree_spike/mod.rs (build_tree gini + SparseTreeNode + 3 kernel wrappers)
  - tests/fixtures/tree_dt_{clf,reg}{,_adv}_{f32,f64}_seed42.npz (Plan 01 sklearn reference blobs)
  - crates/mlrs-core/src/oracle.rs (load_npz by-name loader, no Python at test time)
  - crates/mlrs-core (assert_slice_close + F64_TOL 1e-5/1e-5)
  - crates/mlrs-backend/src/capability.rs (skip_f64_with_log + active_backend_name)
provides:
  - Tier-1 VALUE witness vs sklearn DecisionTreeClassifier(gini) + DecisionTreeRegressor(squared_error) + adversarial
  - A5 verdict evidence (clf + reg + adversarial all GREEN on cpu f64+f32)
  - decision-equivalence + function-equivalence gating recipe (Open Question 1 / Pitfall 4 resolution)
affects:
  - Plan 05 VERDICT.md (A5 = GREEN; threshold + split-feature-tie caveats recorded here)
tech-stack:
  added: []
  patterns:
    - lockstep tree traversal (structure + decision-equivalence + leaf values) for the no-tie classifier
    - function-equivalence (induced partition + per-row predictions) for the RNG-tie regressor
    - decision-exact host binning (midpoints between sorted-unique values) so every sklearn split is representable
    - second histogram launch on y^2 → per-cell sum-of-squares → host variance-reduction gain (same three kernels)
key-files:
  created:
    - crates/mlrs-backend/tests/tree_witness.rs
  modified: []
decisions:
  - "Classifier (no gain-ties) keeps the strict per-node lockstep: exact split FEATURE + decision-equivalence + leaf values <=1e-5"
  - "Regressor split-feature ties at 2-sample nodes are sklearn-splitter-RNG (BestSplitter random_state feature shuffle); gated by function-equivalence (partition + predictions), NOT recorded feature — conforming would be a circular oracle (Pitfall 4)"
  - "Threshold gated by decision-equivalence not raw float (Open Question 1 / A2): global-unique midpoints route node samples identically to sklearn's node-local midpoints"
  - "Regression variance support added as a witness-LOCAL build_tree_variance (2nd histogram on y^2) rather than mutating the shared Plan-02 build_tree, so Plan 04's build_tree signature is untouched (Rule 3 deviation)"
metrics:
  tasks: 2
  files: 1
  commits: 1
  duration_min: 22
  completed: 2026-06-27
status: complete
---

# Phase 17 Plan 03: Tier-1 sklearn Correctness Witness Summary

A single injected-fixed-index tree, built by composing the Plan-02 cpu-MLIR kernels
(histogram / split-find / relabel) through the host build loop, VALUE-matches sklearn EXACTLY:
the classifier reproduces `DecisionTreeClassifier(gini)` per-node (9 nodes / 5 leaves, exact split
structure + leaf values ≤1e-5) and the regressor reproduces `DecisionTreeRegressor(squared_error)`
as a function (25 nodes / 13 leaves, identical induced partition + regression-mean predictions
≤1e-5), the SparseTreeNode contract (D-02/D-03/D-04) is validated, and the adversarial
forced-pure-leaf + gain-tie backstop is green for both — **answering SC-2, SC-3, and abort signal A5
GREEN** on cpu f64 (the correctness gate) and f32.

## What Was Built

**`crates/mlrs-backend/tests/tree_witness.rs`** (8 tests, all GREEN on cpu f64 + f32):

- **Task 1 — clf + reg VALUE witness** (`tree_witness_{clf,reg}_{f32,f64}_matches_sklearn`):
  - Loads each `.npz` via `mlrs_core::oracle::load_npz` (no Python at test time), reconstructs the
    exact fit matrix `X[bootstrap_idx][:, feature_idx]` + `y[bootstrap_idx]` (D-07), bins each
    feature on decision-exact host quantile edges (midpoints between sorted-unique values, D-10),
    and builds one tree composing the three kernels.
  - **Classifier** builds through the shared Plan-02 `build_tree` (gini) and runs a strict **lockstep
    traversal**: per internal node `colid == sklearn feature` (both subset-indexed), decision-equivalent
    routing of the node's samples, recursion with the D-02 implicit `right = left_child + 1`; per leaf
    the dereferenced `value` offset (D-04) `assert_slice_close` ≤1e-5 vs sklearn's `P(class 1)`.
  - **Regressor** builds through a witness-local `build_tree_variance` (see Deviations) and gates
    **function-equivalence**: identical node/leaf counts, identical induced partition of the 48 training
    rows into leaves, and per-row predictions (dereferenced regression-mean `value` offsets) ≤1e-5.
  - Both leaf shapes (class-probability AND regression-mean) flow through the ONE `value` offset field
    — the **D-09 multiclass-uniform proof**.
  - Every f64 test opens with `capability::skip_f64_with_log()` + a backend/dtype log line; f32 always
    runs.

- **Task 2 — adversarial backstop** (`tree_witness_adversarial_{clf,reg}_{f32,f64}_backstop`):
  - Runs the same witness on the Plan-01 adversarial fixtures (two identical columns → exact gain tie;
    separable target → forced-pure leaves), then asserts explicitly: (1) the gain-TIE root resolves to
    the **lowest feature index (0)** — sklearn's pick, independently verified in `gen_oracle.py` via
    pure-numpy impurity, never conformed to the kernel (Phase-13 CR-01/CR-02, non-circular); (2) both
    children are forced-pure leaves (`colid == -1`) whose dereferenced values match sklearn (clf
    `{0,1}`; reg `{1.0, 5.0}`). This is the explicit **002-B silent-cross-loop-miscompile backstop**.

## SparseTreeNode contract validated (D-02 / D-03 / D-04)

`assert_contract` walks every node: internal nodes have `colid >= 0` and an in-range adjacent right
child (`left_child + 1`, D-02); leaves have the `colid == -1` sentinel, `left_child == -1` (D-03), and
a `value` that is a valid OFFSET into the shared leaf-value buffer, never a scalar (D-04). Plus exact
node-count and leaf-count equality vs the sklearn `tree_` arrays.

## A5 verdict evidence (for Plan 05 VERDICT.md)

| Witness | Backend / dtype | Result |
|---------|-----------------|--------|
| clf(gini) standard, 9 nodes / 5 leaves | cpu f64 + f32 | GREEN — exact per-node feature structure, decision-equivalent, leaf values ≤1e-5 |
| reg(squared_error) standard, 25 nodes / 13 leaves | cpu f64 + f32 | GREEN — counts exact, induced partition identical, regression-mean predictions ≤1e-5 |
| clf adversarial (pure-leaf + tie) | cpu f64 + f32 | GREEN — 002-B backstop, tie→feature 0 (independent rule) |
| reg adversarial (pure-leaf + tie) | cpu f64 + f32 | GREEN — 002-B backstop, tie→feature 0 (independent rule) |

**A5 = NO ABORT.** The histogram/gain/partition MATH is correct: with RNG removed (injected indices,
D-07) a single tree reproduces sklearn. No silent cpu-MLIR miscompile (the adversarial boundary fixture
would have failed here and did not).

### Caveats recorded for VERDICT.md

1. **Threshold = decision-equivalence, not raw float (Open Question 1 / A2 — RESOLVED).** Host binning
   uses global-unique midpoints; a node's binned threshold can differ from sklearn's node-local midpoint
   while routing the node's samples identically. The witness gates the decision boundary, not the raw
   `threshold` value. No divergence observed under this gate.
2. **Regressor split-feature ties are sklearn-RNG (Pitfall 4 — RESOLVED non-circularly).** At minimal
   2-sample regression nodes EVERY feature achieves the identical maximum variance reduction (any feature
   perfectly separates two points). sklearn's `BestSplitter` breaks the tie with its `random_state`
   feature shuffle; the injected-index recipe removes the bagging RNG but not the splitter's internal
   tie-break RNG. Conforming the kernel's pick to sklearn's shuffled choice would be a circular oracle,
   so the regressor is gated on function-equivalence (partition + predictions) — proving the trees are
   the same FUNCTION without chasing an RNG artifact. The classifier had no such ties and passes the
   strict per-node feature lockstep.

## Deviations from Plan

**[Rule 3 — Blocking] Regression variance support added as a witness-local builder.**
- **Found during:** Task 1 (regressor witness).
- **Issue:** the plan requires the regressor to be reproduced via "the SAME witness path", but Plan-02's
  shared `build_tree` computes binary-Gini gain only (it has the histogram's count + sum(y)). Variance
  reduction (`squared_error`) additionally needs the per-cell sum-of-squares, which Gini-only `build_tree`
  cannot produce — the regressor witness cannot pass without it.
- **Fix:** implemented `build_tree_variance` LOCAL to `tree_witness.rs`, composing the SAME three public
  Plan-02 kernel wrappers (`launch_histogram` / `launch_split_find` / `launch_relabel`) and launching the
  histogram a second time on `y²` to obtain the per-cell sum-of-squares, then computing variance-reduction
  gain on the host. The kernels under test are identical to the classifier path; only the host gain formula
  differs. This deliberately AVOIDS mutating the shared `build_tree` so Plan-04's `build_tree` signature
  (which it depends on) is untouched, and keeps all new code in the plan's declared file.
- **Files modified:** `crates/mlrs-backend/tests/tree_witness.rs` only (no Plan-02 artifact changed).
- **Commit:** c69e21b.

**[Plan interpretation — structural gate] Lockstep traversal + function-equivalence, not array-index `assert_eq!`.**
- The plan's Task-1 action says "left_child matches sklearn's children_left layout". sklearn lays nodes
  out depth-first (a parent's right child is NOT `left_child + 1`; the clf root has left=1, right=8),
  while the mlrs contract lays children adjacent (D-02). A raw array-index `assert_eq!` is therefore
  impossible. The witness instead asserts the EQUIVALENT and stronger property: structural correspondence
  via lockstep traversal (clf) / induced-partition equality (reg), exactly the "cross-check by traversing
  the built tree against the sklearn split structure" the research (D-02 validation method 1) prescribes.
  Recorded as an interpretation, not a Rule 1-4 deviation.

## Threat-Model Mitigation

- **T-17-05 (data integrity):** mitigated — VALUE-asserts (exact structure / ≤1e-5 leaf values) on
  clf + reg + adversarial; the adversarial forced-pure-leaf + tie fixture is the 002-B silent-miscompile
  backstop and is green.
- **T-17-06 (circular oracle):** mitigated — the adversarial tie-break is the INDEPENDENT generator-encoded
  rule (Plan 01 pure-numpy impurity), never conformed to the kernel; the regressor's RNG split-feature ties
  are likewise gated on function-equivalence, never on sklearn's RNG pick.
- **T-17-SC (installs):** n/a — zero new packages; pre-existing `mlrs-core` / `cubecl` / `bytemuck` only.

## Verification

- `cargo build -p mlrs-backend --features cpu --test tree_witness` → exit 0, no warnings.
- `cargo test -p mlrs-backend --features cpu --test tree_witness -- --nocapture` →
  `test result: ok. 8 passed; 0 failed` (clf f32/f64, reg f32/f64, adversarial clf/reg f32/f64).
- `cargo test -p mlrs-backend --features cpu --test tree_spike_probes` → still `8 passed; 0 failed`
  (Plan-02 module untouched).
- f64 is the cpu correctness gate and carries `skip_f64_with_log()` (SKIPS-with-log on a no-f64 adapter,
  e.g. rocm); f32 always runs.

## Known Stubs

None. The witness is live (loads real committed sklearn blobs, launches the real kernels, value-asserts
every claim). This is a feasibility spike (D-01); no production `src/` prim is written this phase.

## Self-Check: PASSED

- `crates/mlrs-backend/tests/tree_witness.rs` — FOUND
- Commit `c69e21b` — FOUND
