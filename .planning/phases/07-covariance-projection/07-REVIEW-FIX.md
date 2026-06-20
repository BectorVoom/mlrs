---
phase: 07-covariance-projection
fixed_at: 2026-06-21T00:00:00Z
review_path: .planning/phases/07-covariance-projection/07-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 10
skipped: 2
status: partial
---

# Phase 7: Code Review Fix Report

**Fixed at:** 2026-06-21
**Source review:** .planning/phases/07-covariance-projection/07-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 12 (7 warning + 5 info; fix_scope = all)
- Fixed: 10
- Skipped: 2 (both reviewer-classified "None required" — intentional design)

All fixes were applied in an isolated git worktree, committed atomically per
finding, fast-forwarded onto `main`, and the worktree torn down via the
transactional cleanup tail. Each Rust fix was compile-checked (`cargo check
--features cpu`); the WR-02 oracle test was additionally run and PASSES
(`empirical_covariance_assume_centered_attrs_{f32,f64} ... ok`). Python edits
were AST-parse checked.

## Fixed Issues

### WR-01: `n_components <= batch_size` is never validated up front

**Files modified:** `crates/mlrs-algos/src/decomposition/incremental_pca.rs`
**Commit:** 9f6deda
**Applied fix:** Added an up-front `self.n_components > batch_size` check in
`fit()` (after `batch_size` is resolved) that returns
`AlgoError::InvalidNComponents { estimator: "incremental_pca", requested, max:
batch_size }`, matching sklearn's "n_components must be <= batch number of
samples" diagnostic instead of the confusing mid-stream `max=batch_row_count`
error. **Requires human verification** — this is a validation/control-flow
change; the syntax compiles but the rejection semantics should be confirmed
against the intended sklearn behavior.

### WR-02: `assume_centered=True` covariance path has zero oracle coverage

**Files modified:** `scripts/gen_oracle.py`,
`crates/mlrs-algos/tests/empirical_covariance_test.rs`,
`tests/fixtures/empirical_covariance_centered_f32_seed42.npz` (new),
`tests/fixtures/empirical_covariance_centered_f64_seed42.npz` (new)
**Commit:** 3f1c0aa
**Applied fix:** Added an `assume_centered=True` fixture emission to
`gen_oracle.py main()` (`kind='centered'`), generated the two committed
`.npz` blobs from `EmpiricalCovariance(assume_centered=True)`, and added
`empirical_covariance_assume_centered_attrs_{f32,f64}` tests that fit
`EmpiricalCovariance::new(true, true)` and value-gate
`covariance_`/`location_`/`precision_` (plus an explicit all-zero `location_`
assertion). This is the only test that exercises the separate
`mle_gram_uncentered` host-Gram branch. **Verified by running:** both tests
pass on cpu within the 1e-5 contract.

### WR-03: `WHITEN_VAR_FLOOR` branch is untested

**Files modified:** `crates/mlrs-algos/src/decomposition/incremental_pca.rs`
**Commit:** 8e406ef
**Applied fix:** Took the "document the floor as defensive-only/unreachable"
option from the review (the alternative — a near-rank-deficient oracle fixture
— would require a new generated blob and is heavier). Expanded the
`whiten_scales` doc comment to state the `ev <= WHITEN_VAR_FLOOR` branch is
defensive-only and unreachable on fitted data (retained components always carry
non-trivial variance), so no committed fixture exercises it.

### WR-04: false `gen_batches`-fidelity claim in the stream tests

**Files modified:** `crates/mlrs-algos/tests/incremental_pca_test.rs`,
`crates/mlrs-backend/tests/incremental_svd_test.rs`
**Commit:** 150638c
**Applied fix:** Corrected both misleading comments to "naive equal chunking
(== sklearn gen_batches ONLY when n % batch_size == 0)", documenting that the
real `gen_batches(min_batch=n_components)` folds the remainder into the prior
batch for non-divisible geometries (the minimum fix the review proposed).

### WR-05: `fit()`/`partial_fit()` reset semantics undocumented

**Files modified:** `crates/mlrs-py/python/mlrs/decomposition.py`
**Commit:** af27742
**Applied fix:** Documented the reset contract on both methods — `fit` builds a
fresh `_mlrs` object (RESET, non-cumulative), `partial_fit` reuses the existing
object (CONTINUES the stream), and the asymmetry when interleaving them (matches
sklearn `IncrementalPCA`). The review's optional pytest pinning
`n_samples_seen_` across interleavings was NOT added (the doc contract is the
core fix; see Partial Notes).

### WR-06: `gaussian_matrix` exact-zero draw injects a ~37σ outlier

**Files modified:** `crates/mlrs-backend/src/prims/rng.rs`
**Commit:** e8fe4c6
**Applied fix:** Changed the Box–Muller `u1` floor from `f64::MIN_POSITIVE`
(~2.2e-308 → r ≈ 37.6σ) to the stream's own quantization resolution `2^-53`,
which bounds the worst-case `r` to ~8.6σ. **Requires human verification** —
this changes the generated random-projection matrix values at the floored draw;
any seed-stable fixture/property test asserting exact moments should be
re-confirmed.

### WR-07: `next_below(0)` degrades to a divide-by-zero panic in release

**Files modified:** `crates/mlrs-backend/src/prims/rng.rs`
**Commit:** 9ef6310
**Applied fix:** Widened the early-return guard from `if bound == 1` to
`if bound <= 1`, so a `bound == 0` returns the empty-range degenerate `0` in
release builds instead of reaching `u64::MAX % bound` (the opaque modulo panic).
The `debug_assert!(bound >= 1)` is retained as documentation.

### IN-01: misleading "RESEARCH Pattern 4" attribution on the Gaussian generator

**Files modified:** `crates/mlrs-backend/src/prims/rng.rs`
**Commit:** f414638
**Applied fix:** Corrected the `gaussian_matrix` doc to cite "RESEARCH
Pattern 5" (the Gaussian `N(0, 1/n_components)` projection, matching its own
estimator at `projection/gaussian.rs:64`) and clarified that Pattern 4 is the
Achlioptas SPARSE matrix.

### IN-03: `decomposition/mod.rs` re-exports only `IncrementalPCA`

**Files modified:** `crates/mlrs-algos/src/decomposition/mod.rs`
**Commit:** e1a901d
**Applied fix:** Added `pub use pca::Pca;` and `pub use
truncated_svd::TruncatedSvd;` at the module root for a consistent surface.
Compiles with no unused/conflict warnings.

### IN-04: `LedoitWolf` keeps an inert `ddof` local

**Files modified:** `crates/mlrs-algos/src/covariance/ledoit_wolf.rs`
**Commit:** dfbe4d5
**Applied fix:** Removed the misleading `let ddof: u32 = 0;` local and inlined
`g / n`, with a comment stating ddof is hard-pinned to 0 for the MLE.

## Skipped Issues

### IN-02: Dead `_y` parameter on every unsupervised `fit`

**File:** `crates/mlrs-algos/src/covariance/empirical_covariance.rs:139` (and
`ledoit_wolf.rs:126`, `projection/gaussian.rs:159`, `projection/sparse.rs:131`)
**Reason:** No action required — the reviewer classified this as intentional
(trait uniformity for Phase-10 MBSGD reuse) and the `_` prefix already documents
intent. The review's own Fix line is "None required".
**Original issue:** Unsupervised estimators take and ignore `_y` per the shared
`Fit` trait; noted only so a reader does not mistake it for unfinished wiring.

### IN-05: host accessors run a device→host copy per call

**File:** `crates/mlrs-algos/src/covariance/empirical_covariance.rs:101-128`
(and `ledoit_wolf.rs:86-115`)
**Reason:** No action required — explicitly out of v1 perf scope; the reviewer's
Fix line is "None required for v1; consider memoizing if profiling shows a
hotspot". Flagged for awareness only.
**Original issue:** `covariance_()`/`location_()`/`precision_()` each run a
device→host copy on every call, which a `@property` in a loop can surprise.

## Partial Notes

- **WR-03** took the documentation option rather than adding a near-rank-deficient
  fixture (both were offered by the review).
- **WR-05** documented the reset contract (the core ask) but did not add the
  optional pytest pinning `n_samples_seen_` across `fit`/`partial_fit`
  interleavings.
- **WR-01, WR-06** are flagged "requires human verification" above: both are
  behavior changes (a new rejection path and a changed RNG draw) that pass
  syntax/compile checks but whose semantics a developer should confirm before
  the phase proceeds to verification.

---

_Fixed: 2026-06-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
