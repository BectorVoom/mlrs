---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
fixed_at: 2026-06-27T00:00:00Z
review_path: .planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/17-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 17: Code Review Fix Report

**Fixed at:** 2026-06-27
**Source review:** .planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/17-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (fix_scope = all → WR + IN)
- Fixed: 5
- Skipped: 0

All fixes were verified by recompiling the affected test targets with the `cpu`
feature AND running the full Phase-17 cpu test suite (8 `tree_witness` + 8
`tree_spike_probes` = 16 tests, all green) after the source-changing fixes. No
committed fixture blobs were regenerated — the generator changes (IN-03) are
generation-time guards that the existing blobs already satisfy (verified against
the installed numpy 2.4.6 / sklearn 1.9.0 without overwriting the committed
fixtures).

## Fixed Issues

### WR-01: Adversarial classifier witness strict-locksteps the split feature on a deliberate gain TIE

**Files modified:** `crates/mlrs-backend/tests/tree_witness.rs`
**Commit:** fb2889f
**Applied fix:** Changed the terminal dispatch in `run_witness` from `match kind`
to `match (kind, adversarial)`. The standard classifier (`(Kind::Clf, false)`)
keeps the strict per-node `compare_rec` lockstep (no ties there); the adversarial
classifier now joins the regressor in the `assert_function_equiv` arm
(`(Kind::Clf, true) | (Kind::Reg, _)`), which gates the tree as a FUNCTION
(induced partition + per-row predictions) and is feature-index-independent. This
removes the circular oracle where the witness demanded sklearn record feature 0
at an RNG-resolved gain-tie root. The kernel's own lowest-index tie-break is still
proven independently by `check_adversarial` (`nodes[0].colid == 0`), so no
coverage is lost. Verified: all 8 cpu witness tests (incl. both adversarial clf
backstops) pass.

### WR-02: `build_tree_variance` duplicates ~90% of `build_tree`

**Files modified:** `crates/mlrs-backend/tests/tree_spike/mod.rs`, `crates/mlrs-backend/tests/tree_witness.rs`
**Commit:** ea659c8
**Applied fix:** Extracted the single histogram → split-find → relabel frontier
skeleton into a new generic `pub fn build_tree_with<F, G, L>(…, level_gain,
leaf_value)` driver in `mod.rs`. `level_gain` returns this level's per-candidate
gain plus a per-node purity flag; `leaf_value(sum_y, total)` maps a leaf's
feature-0 totals to its stored value. `build_tree` is now a thin wrapper supplying
Gini gain + probability leaf (Plan-04 signature unchanged); `build_tree_variance`
supplies variance-reduction gain (with its own second `y²` histogram launch) +
mean leaf. Each criterion's exact prior behavior is preserved verbatim inside its
closure (the Gini `tot > 0.0` guard and the variance `lc > 0.0 && rc > 0.0` guard
are kept distinct, as before). The ~100-line second copy of the loop is gone.
Verified: 16 cpu tests (8 witness + 8 probes, incl. `build_tree` end-to-end and
both regression witnesses) pass — the refactor is behavior-preserving on the real
correctness gate.

### IN-01: f32 witness rebuilds on f32-rounded data while routing against f32-rounded sklearn thresholds

**Files modified:** `crates/mlrs-backend/tests/tree_witness.rs`
**Commit:** 197beac
**Applied fix:** Added a module-doc section ("The f32 path is a COMPANION smoke
check, NOT a bit-exact sklearn match") documenting that the generator fits sklearn
in float64 and commits f32-cast arrays, so the f32 witness reconstructs from
f32-rounded `X` and routes against f32-rounded thresholds. It explains the latent
fragility (a rounding-collapsed unique or a near-threshold flip could fail a
functionally-correct tree) and clarifies that f64 is the real correctness gate and
the f32 run is companion plumbing coverage. Documentation-only (per the review's
own recommendation for spike code).

### IN-02: `var` uses the cancellation-prone `E[y²] − E[y]²` formula masked by `.max(0.0)`

**Files modified:** `crates/mlrs-backend/tests/tree_witness.rs`
**Commit:** 09463d5
**Applied fix:** Added `assert!(v >= -slack, …)` before the `.max(0.0)` clamp in
the variance closure, where `slack = 1e-4 * (|sq/c| + m*m + 1.0)` is RELATIVE to
the cancelling terms. A genuinely-negative variance from a sum-of-squares
miscompile (the 002-B failure this spike targets) now surfaces as a loud failure
instead of being silently floored to 0. The slack is relative rather than absolute
because the histogram sums carry f32 rounding (~eps·E[y²]): an initial absolute
`-1e-6` bound false-positived on the legitimate f32 regression case (observed
`v = -1.3e-6`); the relative bound tolerates that cancellation while still catching
a gross miscompile. Verified: all 8 cpu witness tests pass with the new assert.

_Note: this fix is a robustness guard, not a logic change to the variance math
itself; its effect (catching a future miscompile) cannot be exercised by the
current green fixtures and would only fire on a regressed kernel._

### IN-03: Tie-genuineness assert in the generator is a tautology for identical columns

**Files modified:** `scripts/gen_oracle.py`
**Commit:** 57c9665
**Applied fix:** In both `gen_decision_tree_clf` and `gen_decision_tree_reg`
adversarial branches, reframed the always-true `abs(imp0-imp1)<1e-12` /
`abs(v0-v1)<1e-12` checks as data-construction sanity checks (their comments no
longer over-claim to prove the tie genuine) and added the load-bearing guard the
witness actually depends on: `assert int(clf.tree_.feature[0]) == 0` (and the reg
equivalent), pinning sklearn's canonical lowest-index pick at the gain/variance
tie root. Combined with WR-01 (witness gates the function, not the feature), this
makes the committed blob verified against the documented tie-break rule at
generation time: a sklearn shuffle/RNG change that recorded feature 1 fails loudly
at regeneration. Verified: the new guards pass for f32+f64 clf+reg under the
installed sklearn 1.9.0, run into a throwaway dir so the committed fixtures were
not overwritten. No fixture regeneration required.

---

_Fixed: 2026-06-27_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
