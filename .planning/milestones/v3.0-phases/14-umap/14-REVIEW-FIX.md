---
phase: 14-umap
fixed_at: 2026-06-24T00:00:00Z
review_path: .planning/phases/14-umap/14-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 11
skipped: 1
status: partial
---

# Phase 14: Code Review Fix Report

**Fixed at:** 2026-06-24
**Source review:** .planning/phases/14-umap/14-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 12 (fix_scope = all; 1 critical, 6 warning, 5 info)
- Fixed: 11
- Skipped: 1 (WR-04 — applied then reverted; the chosen fix introduced a test failure)

All fixes were applied in an isolated git worktree, each committed atomically.
Verification was Tier-2 (cargo build of the affected crate) for every fix, plus
targeted runtime gates for the behavioral changes:
- `layout_property_cosine` — PASSES (validates the fit pipeline through the
  IN-02 `map_metric` refactor, the IN-03 renamed RNG constants, and the IN-05
  distinct-row k-means init; also confirmed the WR-04 revert restored the gate).
- `transform_property_cosine` — PASSES (validates CR-01's cosine halving on the
  transform path: the `1−cos`-scale membership graph stays within the relative
  `TRANSFORM_PROPERTY_EPS = 0.15` gate).
- `ab_fit` — PASSES (fast smoke that the rebuilt test binary runs).

The full umap value-gate / reproducibility suite (`smooth_knn_*`, `fuzzy_union_*`,
`spectral_init_*`, `reproducible_*`) was NOT run to completion: under
`--test-threads=1` it exceeds the available per-command time budget (known: the
mlrs-backend cpu suite is slow). The IN-03 change preserves the literal constant
VALUES exactly (only names were introduced), so the D-05 byte-identical contract
is preserved by construction; IN-02 is a behavior-preserving refactor returning
the same `knn_graph::Metric` and the same Minkowski `p`.

## Fixed Issues

### CR-01: Cosine `transform` feeds `2(1−cos)` distances while `fit` feeds `1−cos`

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** 9bc6675
**Status:** fixed — requires human verification (logic/correctness change)
**Applied fix:** Added a `cosine_halve` flag in `query_train_knn` (mirroring
`knn_graph`'s post-GEMM halving at knn_graph.rs:212-219) and multiply the
top-k transform distances by `0.5` for the Cosine metric so the transform KNN
distances are on the same `1−cos` scale as fit before they flow into
`smooth_knn_dist`. `transform_property_cosine` passes after the change.

### WR-01: `compute_membership_strengths` index could OOB-write the dense affinity

**Files modified:** `crates/mlrs-algos/src/error.rs`, `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** 2df53f1
**Applied fix:** Added a new `AlgoError::InvalidGraphInput { estimator, reason }`
variant and a bounds check at the COO-consumption boundary in `run_umap_layout`:
before the `affinity[g_rows[e] * n + g_cols[e]] = g_vals[e]` write, return a
typed error if `g_rows[e] >= n || g_cols[e] >= n`, instead of a silent OOB write.

### WR-02: `transform_new_points` divides by `n_components` with no guard

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** ba8b3d0
**Applied fix:** At the top of `transform_new_points`, return
`AlgoError::InvalidGraphInput` when `n_components == 0`, and when the fitted
embedding length is not a multiple of `n_components` — before the
`/ n_components` division and the downstream `init_graph_transform` indexing.

### WR-03: `smooth_knn_dist` `k == 1` yields a runaway-then-floored σ

**Files modified:** `crates/mlrs-algos/src/manifold/umap_internals.rs`
**Commit:** 4161059
**Applied fix:** Doc-only (the suggestion's "downgrade the doc claim" option,
chosen to preserve umap-learn `range(1, k)` parity and the committed fixtures).
Rewrote the threat-model doc on `smooth_knn_dist` to state the guarantee holds
in the iteration-cap sense and to document the bounded-but-degenerate `k == 1`
case (empty inner sum → `mid` doubles to the floor) as parity-correct.

### WR-05: `make_epochs_per_sample` `w_max` fold is not NaN-safe

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** 965beb5
**Applied fix:** Changed `make_epochs_per_sample` to return
`Result<Vec<f64>, AlgoError>`; it now validates every weight is finite BEFORE
the `w_max` reduction (returning `InvalidGraphInput` on a non-finite weight) so
`f64::max(0.0, NaN) == 0.0` can no longer silently corrupt the schedule. Both
call sites (`transform_new_points`, `run_umap_layout`) use `?`.

### WR-06: eig working-buffer aliasing duplicated in `spectral_init` without the invariant comment

**Files modified:** `crates/mlrs-algos/src/manifold/umap_init.rs`
**Commit:** 5a44db3
**Applied fix:** Doc-only. Replaced the one-line "WR-05 aliasing precedent"
reference in `spectral_init` with the full two-invariant eig-aliasing soundness
comment (verbatim from spectral_embedding.rs:248-259), noting this is the third
copy of the pattern.

### IN-01: Stale/contradictory doc comments

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`, `crates/mlrs-kernels/src/umap_layout.rs`
**Commit:** c1278cd
**Applied fix:** Deleted the contradictory stream-of-consciousness paragraph in
`transform_new_points` (replaced with a concise accurate note that the estimator
retains `x_train_`). Corrected the `umap_layout.rs` module doc from the stale
`fit` `move_other = 1` to the actual owner-only `move_other = 0` (with the
CR-01-option-b rationale).

### IN-02: Redundant dual-carriage of the Minkowski `p`; dead `let _ = p`

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** f48c7a7
**Applied fix:** `map_metric` now returns only `knn_graph::Metric`. Added a
`minkowski_p(knn_graph::Metric) -> f64` helper that reads the exponent from the
enum payload (single source of truth); the fit call site uses it for the
`knn_graph` scalar arg. Removed the dead `let _ = p;` in `query_train_knn`.

### IN-03: Reproducibility-critical mixing constants unnamed

**Files modified:** `crates/mlrs-algos/src/manifold/umap.rs`
**Commit:** ec23ed1
**Applied fix:** Promoted the three magic numbers to named module consts
(`SUBSTREAM_SEED_MULT = 0x9E37_79B9_7F4A_7C15`,
`SUBSTREAM_EPOCH_MULT = 0x1000_0001`, `INIT_SCALE_SEED_XOR = 0x5350_4543`) with
a header comment stating they are fixed D-05 byte-identical stream separators.
The literal VALUES are unchanged, so reproducible output is byte-identical.

### IN-04: `noisy_scale_coords` `n`/`n_components` params are debug-only

**Files modified:** `crates/mlrs-algos/src/manifold/umap_init.rs`
**Commit:** 8c7f5b3
**Applied fix:** Doc-only (the "document" option). Added a note that `n` and
`n_components` exist solely for the debug-build shape assertion and are dead
parameters in release builds.

### IN-05: Test helper `host_kmeans_labels` init not robust to non-distinct rows

**Files modified:** `crates/mlrs-algos/tests/umap_test.rs`
**Commit:** c66a3d9
**Applied fix:** Replaced the literal "first k rows" centroid seeding with a
deterministic distinct-row scan (accept a row only if it differs from every
already-chosen centroid; pad with row 0 if fewer than k distinct rows exist), so
a collapsed/degenerate embedding cannot leave a centroid permanently empty.
Exercised by the passing `layout_property_cosine` ARI path.

## Skipped Issues

### WR-04: Fit/transform `n_epochs` defaults diverge from the oracle's `n_epochs=200`

**File:** `crates/mlrs-algos/src/manifold/umap.rs:1134`;
`crates/mlrs-algos/tests/umap_test.rs:351-361,986-994`; `scripts/gen_oracle.py:946`
**Reason:** skipped — the chosen fix (option a: set `.n_epochs(Some(200))` in the
layout/transform property tests) was applied (commit eb1b7f5) but it made
`layout_property_cosine` FAIL: at 200 epochs the kNN-overlap margin lands at
exactly `0.6433 < 0.6733 − 0.03` (margin == `PROPERTY_EPS = 0.03`), failing the
strict `<` gate. The calibrated `PROPERTY_EPS` was measured against the 500-epoch
fit default, so epoch-matching the tests to the oracle without ALSO recalibrating
`PROPERTY_EPS` (or regenerating the fixtures at the 500-epoch default — the
review's option b) breaks the gate. The fix was reverted (commit 0124c59); the
`layout_property_cosine` gate passes again after the revert.

**Original issue:** The property tests never set `.n_epochs(...)`, so mlrs runs
500 fit epochs against a 200-epoch umap reference; the `PROPERTY_EPS` calibration
was therefore not measured against an epoch-matched layout.

**Recommended follow-up (human):** This is a calibration decision, not a
mechanical edit — pick one: (a) set `.n_epochs(Some(200))` AND recalibrate
`PROPERTY_EPS` against the 200-epoch margins (re-run the calibration sweep and
record in 14-VALIDATION.md), or (b) regenerate the committed fixtures at mlrs's
500-epoch default (requires the oracle-gen venv — see project memory
"oracle fixture regen needs venv"). Either keeps the calibration like-for-like
without weakening the gate.

---

_Fixed: 2026-06-24_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
