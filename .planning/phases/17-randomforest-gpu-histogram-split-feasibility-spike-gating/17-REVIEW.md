---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
reviewed: 2026-06-27T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - crates/mlrs-backend/tests/tree_bench.rs
  - crates/mlrs-backend/tests/tree_spike/mod.rs
  - crates/mlrs-backend/tests/tree_spike_probes.rs
  - crates/mlrs-backend/tests/tree_witness.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-06-27
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the Phase-17 RandomForest GPU-tree feasibility spike: three cpu-MLIR-safe
CubeCL kernels + host launch wrappers (`tree_spike/mod.rs`), the host per-level
`build_tree` loop, the value-asserting kernel probes (`tree_spike_probes.rs`), the
Tier-1 sklearn witness (`tree_witness.rs`), the wall-clock A3 cost benchmark
(`tree_bench.rs`), and the sklearn fixture generator (`gen_oracle.py` Phase-17
section).

Overall the math is sound. I traced the histogram → gain → split-find → relabel
composition end-to-end and confirmed: every output cell is written (no
uninitialized `client.empty` reads), empty-child splits resolve to exactly `0.0`
gain via the algebra (`lc == tot` ⇒ `g = parent - parent - 0`), bin-rank routing
(`bv > split_bin`) is equivalent to midpoint-threshold routing, the lowest-index
tie-break is correctly seeded from candidate 0, and `u32` geometry is overflow-
checked before every `unsafe` launch. No correctness BLOCKER found; the suite's
24/24 green status is consistent with the logic as written.

The findings below concern a non-circularity gap in the **adversarial classifier**
gate (it strict-locksteps the split FEATURE on a node that is, by construction, an
RNG-resolved tie), a sizable duplicated build loop, and two precision/robustness
notes appropriate to spike code.

## Warnings

### WR-01: Adversarial classifier witness strict-locksteps the split feature on a deliberate gain TIE — couples the gate to sklearn's RNG tie-break

**File:** `crates/mlrs-backend/tests/tree_witness.rs:688-694` (Clf branch → `compare_rec`), `crates/mlrs-backend/tests/tree_witness.rs:443-448` (`assert_eq!(node.colid, sk_feat)`); `scripts/gen_oracle.py:3078-3100` (adversarial clf generator)

**Issue:** The module doc (lines 39-53) carefully argues that split-feature ties
are sklearn-`random_state`-determined and therefore must be gated as a FUNCTION
(partition + predictions), never by the recorded feature — and applies that
non-circular treatment to the regressor (`assert_function_equiv`). But
`run_witness` dispatches purely on `Kind`, so the **adversarial classifier**
(`check_adversarial::<F>(Kind::Clf)`) still flows through the strict
`compare_rec` path, which asserts `node.colid == sk_tree.feature[sk]` at every
internal node — including the root, which is by construction an EXACT gain tie
between two identical columns. sklearn's `BestSplitter` Fisher-Yates-shuffles the
feature order (even with `max_features=None`) and keeps the first-evaluated
feature on a strict-`>` tie, so which column it records at that tie root is an
RNG outcome. The mlrs kernel deterministically picks feature 0; the witness then
demands sklearn also recorded feature 0. This is exactly the circular oracle the
design forbids for the regressor.

It currently passes only because `random_state=42` happened to make sklearn
record feature 0. The generator (`gen_decision_tree_clf` adversarial) verifies
only that the tie is *genuine* (`abs(imp0 - imp1) < 1e-12`, trivially true for
identical columns) — it never pins or asserts that sklearn recorded the lowest
index. A sklearn version bump that alters the shuffle RNG could regenerate the
committed blob with `feature == 1` at the tie root, which would make `compare_rec`
fail (`split feature mismatch`) even though the tree is functionally identical.

**Fix:** Gate the adversarial classifier the same non-circular way the regressor
is gated. Either route `(Kind::Clf, adversarial=true)` through
`assert_function_equiv` (partition + leaf-value equivalence, feature-index-
independent), or, in the generator, independently assert sklearn recorded the
canonical lowest-index feature and skip the strict per-node feature equality at
known-tie nodes:

```rust
// in run_witness, branch on adversarial as well as kind:
match (kind, adversarial) {
    (Kind::Clf, false) => { /* strict per-node lockstep (no ties) */ }
    // adversarial clf root is a deliberate tie -> gate the FUNCTION, not colid
    (Kind::Clf, true) | (Kind::Reg, _) => {
        assert_function_equiv(&nodes, &leaf_buf, &sk_tree, &x_fit, n, nf);
    }
}
```

(The explicit `nodes[0].colid == 0` check in `check_adversarial` already proves
the kernel's own lowest-index tie-break independently, so dropping the
sklearn-feature equality at the tie node loses no real coverage.)

### WR-02: `build_tree_variance` duplicates ~90% of `build_tree` — divergence hazard across the two frontier loops

**File:** `crates/mlrs-backend/tests/tree_witness.rs:204-381` vs `crates/mlrs-backend/tests/tree_spike/mod.rs:494-666`

**Issue:** `build_tree_variance` is a near-verbatim copy of `build_tree`: the
candidate `(col_of, bin_of)` map construction, the `leaf_placeholder`, the
per-level frontier `while` loop, the D-02 adjacency push, the `split_active/
split_col/split_bin/left_child` frontier-array assembly, the relabel call, and
the final max-depth leaf sweep are all duplicated, with only the host gain
formula (Gini vs variance) and the leaf value (probability vs mean) differing.
That is roughly 100 lines of structural logic kept in two places. The
duplication is deliberate and documented (the doc explains the team did not want
to mutate the shared `build_tree` signature that Plan 04 depends on), but it is a
real maintenance hazard: a fix to the adjacency, leaf, or termination logic in
one copy will silently not propagate to the other, and the two are already
subtly different (the variance copy guards `lc > 0.0 && rc > 0.0` at line 297
while the Gini copy at `mod.rs:577` relies on `tot > 0.0` + algebra to zero out
empty-child gain).

**Fix:** Factor the shared skeleton into a single generic driver parameterized by
a gain closure and a leaf-value closure, e.g.
`build_tree_with(binned, y, edges, …, gain_fn, leaf_fn)`, and have both the Gini
and variance builders call it. This keeps the Plan-04 `build_tree` signature
intact (wrap it) while removing the second copy of the frontier loop.

## Info

### IN-01: f32 witness rebuilds on f32-rounded data while routing against f32-rounded sklearn thresholds — latent structural fragility

**File:** `crates/mlrs-backend/tests/tree_witness.rs:139-195` (`reconstruct` + `make_bins`); `scripts/gen_oracle.py:3110,3119-3137`

**Issue:** The generator fits sklearn on `x_fit` in float64 (`rng.standard_normal`
default dtype) but commits `X`, `threshold`, and `value` cast to the fixture dtype
(f32 for the f32 fixtures). The f32 witness then reconstructs `x_fit` from the
f32-rounded `X`, re-derives unique values / bin edges, rebuilds the tree, and
decision-routes against the f32-rounded `sk_thr`. If f32 rounding ever (a)
collapsed two previously distinct feature values into one unique (changing the
bin layout) or (b) flipped a sample that sits within ~1e-7 of a threshold, the
node-count assertion or `compare_rec` decision-equivalence would fail. It passes
on this well-separated random-normal data, and f64 is the real correctness gate,
so this is a robustness note rather than a defect.

**Fix:** Document explicitly that the f32 path is a companion smoke check (not a
bit-exact sklearn match) and/or generate f32 fixtures by fitting sklearn on the
f32-cast data so the reference structure is derived from the same rounded inputs
the witness sees.

### IN-02: `var` uses the cancellation-prone `E[y²] − E[y]²` formula masked by `.max(0.0)`

**File:** `crates/mlrs-backend/tests/tree_witness.rs:247-254`

**Issue:** `var(sq, sm, c) = (sq/c - m*m).max(0.0)` is the one-pass "computational"
variance, which is subject to catastrophic cancellation, and the `.max(0.0)`
clamp would silently hide a genuinely-negative result produced by a real bug in
the sum-of-squares histogram (the very 002-B miscompile this spike is trying to
catch). On the tiny spike fixtures this is numerically harmless, but the clamp
weakens the kernel's own error-detection.

**Fix:** Either compute variance two-pass on the host (`E[(y−mean)²]`) for the
oracle path, or assert `sq/c - m*m >= -tol` before clamping so a negative result
surfaces as a failure instead of being silently floored to 0.

### IN-03: Tie-genuineness assert in the generator is a tautology for identical columns

**File:** `scripts/gen_oracle.py:3090-3094` and `3164-3168`

**Issue:** `imp0 = _dt_gini_best_impurity(x_fit[:, 0], …)` / `imp1 = …(x_fit[:, 1], …)`
followed by `assert abs(imp0 - imp1) < 1e-12` operates on two columns that were
constructed identically (`x = np.column_stack([base, base])`), so the assert can
never fail — it does not actually prove anything about sklearn's tie behavior or
the canonical pick. The comment claims it proves "the adversarial gain tie is
genuine," but a tie between byte-identical columns is genuine by construction.

**Fix:** If the intent is to guard the adversarial design, assert the stronger,
non-trivial property the witness actually depends on — e.g. that sklearn's
recorded root `feature == 0` (the canonical lowest-index pick) — so the committed
blob is verified against the documented tie-break rule rather than against an
identity that is always true. (See WR-01.)

---

_Reviewed: 2026-06-27_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
