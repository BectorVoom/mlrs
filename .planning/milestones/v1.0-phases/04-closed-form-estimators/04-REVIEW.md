---
phase: 04-closed-form-estimators
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/linear/mod.rs
  - crates/mlrs-algos/src/linear/linear_regression.rs
  - crates/mlrs-algos/src/linear/ridge.rs
  - crates/mlrs-algos/src/decomposition/mod.rs
  - crates/mlrs-algos/src/decomposition/pca.rs
  - crates/mlrs-algos/src/decomposition/truncated_svd.rs
  - crates/mlrs-core/src/error.rs
  - crates/mlrs-kernels/src/cholesky.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-backend/src/prims/cholesky.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-06-12
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Reviewed the four closed-form estimators (LinearRegression / Ridge / PCA /
TruncatedSVD), the new single-cube Cholesky/triangular-solve CubeCL kernel and
its host wrapper, the estimator-facing error enum, and the oracle fixture
generator. The implementation is careful and well-documented: the oracle tests
pass within 1e-5 on the cpu(f64) and rocm(f32) gates, the SVD pseudo-inverse and
Cholesky normal-equations solvers are correctly composed from the validated
Phase-2/3 primitives, the float/runtime-generic contract holds (no hardcoded f32
or backend), and the Ridge buffer-reuse / handle-ref-counting path (the area most
likely to harbor a use-after-free) was traced and found to be correct — the
`drop(gram)`-instead-of-`release_into` decision genuinely avoids a double-file
aliasing bug.

No Critical/Blocker defects were proven. The findings below are robustness,
numerical-edge, and quality concerns. The most material are: a scale-blind
absolute pivot floor in the Cholesky kernel (WR-01), a non-scale-relative
`n_samples ≤ 1` / zero-variance guard that can still admit NaN-producing inputs
the 1e-5 gate does not exercise (WR-02, WR-03), and a deliberate dead
`column_reduce` "key-link" call that costs an extra kernel launch + host
read-back per `fit` on the hot path (WR-04).

## Warnings

### WR-01: Cholesky pivot guard uses a scale-blind absolute floor

**File:** `crates/mlrs-kernels/src/cholesky.rs:107,157`
**Issue:** The non-SPD / NaN guard is `let floor = F::new(1e-12); ... if diag <= floor && spd_ok`.
The floor is an **absolute** constant, not relative to the matrix scale. Two
failure modes follow:
- For a Gram with large-magnitude entries (e.g. Ridge `XᵀX` over unscaled
  features), a *genuinely* indefinite matrix can present a small-negative-or-tiny
  pivot whose magnitude is still well above `1e-12` only because the whole matrix
  is large — the guard fires correctly there, but the symmetric case is the
  problem: a legitimately SPD but tiny-scaled matrix (all entries ~1e-10) has a
  true positive pivot below `1e-12` and is **wrongly rejected** as
  `NotPositiveDefinite`.
- The guard does not protect against catastrophic f32 cancellation at moderate
  scale: a pivot that should be ~`1e-7·‖A‖` can round to a small positive value
  above `1e-12`, get `sqrt`'d, and feed an ill-conditioned solve without flagging.

The committed fixtures are all well-conditioned (`A = MᵀM + n·I`), so this is
latent. sklearn / LAPACK use a relative pivot test (`pivot ≤ eps·max_diag`).
**Fix:** Make the floor relative to the matrix scale, e.g. track
`max_diag = max(A[i][i])` in a first pass (or accept it as a kernel arg computed
host-side like the SVD thresholds in `svd.rs::compute_thresholds`) and test
`if diag <= eps_F * max_diag`. At minimum, document that the absolute `1e-12`
assumes O(1)-scaled inputs and add a scaled-input fixture to the gate.

### WR-02: PCA/TruncatedSVD `n_samples ≤ 1` rejection mislabels the error

**File:** `crates/mlrs-algos/src/decomposition/pca.rs:162-170`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:153-161`
**Issue:** The undefined-variance guard for `n_samples ≤ 1` returns
`AlgoError::Prim(PrimError::ShapeMismatch { rows: n_samples, cols: n_features, len: x.len() })`.
But `x.len()` legitimately equals `n_samples * n_features` here — the shape is
**not** mismatched; the rejection reason is "too few samples for variance," which
the `ShapeMismatch` message (`rows*cols != len`) actively contradicts. A caller
that reads the error will be told the buffer length is wrong when it is correct,
and the displayed message will read `rows(1) * cols(k) = k != len(k)` — a false
statement. **Fix:** Add a dedicated `AlgoError` variant (e.g.
`InsufficientSamples { estimator, n_samples }`) or reuse `InvalidNComponents`-style
typed reporting; do not overload `ShapeMismatch` for a non-shape failure.

### WR-03: Zero-variance denominator guard does not cover NaN/degenerate spectra

**File:** `crates/mlrs-algos/src/decomposition/pca.rs:219-223`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:230-234`
**Issue:** `let total_safe = if total_var.abs() > 0.0 { total_var } else { 1.0 };`.
`total_var` is a sum of squares and is always `≥ 0`, so `.abs()` is dead — the
guard is just `total_var > 0.0`. More importantly, if `total_var` is exactly `0`
(all-constant input columns), `explained_variance_ratio_` is silently set to
`ev/1.0 = 0` for every component, which is a plausible-but-undocumented fabricated
value (sklearn would produce `0/0 = nan`). The guard prevents a div-by-zero but
masks a degenerate input rather than surfacing it. **Fix:** Drop the redundant
`.abs()`; decide explicitly whether an all-zero-variance fit should error
(recommended, mirroring the `n_samples ≤ 1` guard) or document the `ratio = 0`
fallback as intentional. Currently it is neither errored nor tested.

### WR-04: Dead `column_reduce` "key-link" call on the LinearRegression/Ridge fit hot path

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:209-219`, `crates/mlrs-algos/src/linear/ridge.rs:213-223`
**Issue:** Both estimators launch
`column_reduce(.., ScalarOp::Mean, ReducePath::Shared)` on the centered design,
immediately `to_host` the result, discard it (`let _ = _centered_means.to_host(pool);`),
and release it. The comment states the result is "not load-bearing for the solve"
— it exists only to satisfy a "documented key-link prim call." This is dead
computation that costs a full kernel launch **plus a device→host read-back** on
every `fit`. The read-back in particular contradicts the phase's device-residency
/ minimal-read-back intent (D-03) and inflates the very `read_backs` counter the
memory gate asserts on. **Fix:** Remove the call entirely (the load-bearing means
are already computed host-side in the two-pass loop above), or if the key-link
requirement is real, drop the `to_host` so it does not force a read-back, and add
a comment explaining why a discarded device result is retained.

### WR-05: `fit` does not reset prior fitted state on a mid-fit error

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:126-305`, `crates/mlrs-algos/src/linear/ridge.rs:128-310`, `crates/mlrs-algos/src/decomposition/pca.rs:143-271`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:134-265`
**Issue:** `fit` takes `&mut self` and assigns `self.coef_ = Some(..)` etc. only at
the very end. If a second `fit` call fails partway (e.g. `svd` returns
`NotConverged`, or Cholesky returns `NotPositiveDefinite` via `?`), the estimator
**retains the fitted state from the previous successful `fit`**, so a subsequent
`predict`/`transform` silently uses stale `coef_`/`components_` with the new
problem's geometry assumption. sklearn clears/invalidates attributes on a failed
re-fit. For PCA, `self.n_features` is also only updated at the end, so a failed
re-fit leaves a stale `n_features` that `transform`'s geometry check trusts.
**Fix:** Either set all fitted slots to `None` at the top of `fit` before any
fallible work, or assign into locals and only commit to `self` once every step
has succeeded (which also frees the old device buffers deterministically).

### WR-06: Stale fitted device buffers are leaked into the pool on re-fit

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:302-303`, `crates/mlrs-algos/src/linear/ridge.rs:307-308`, `crates/mlrs-algos/src/decomposition/pca.rs:264-269`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:259-263`
**Issue:** When `fit` is called a second time on an already-fitted estimator,
`self.coef_ = Some(new_coef)` overwrites the `Option`, dropping the old
`DeviceArray`. Because `DeviceArray` has no `Drop` impl that returns the handle to
the pool (release is only via the by-value `release_into`), the old buffer's
pool accounting is never decremented — `live_bytes` stays elevated and the buffer
is never re-added to the free-list, defeating the buffer-reuse design (D-11) on
the re-fit path. This is a pool-accounting/efficiency leak, not a memory-safety
bug (the GPU buffer is freed when the last handle clone drops). **Fix:** Before
overwriting a fitted slot, take the old value and `release_into(pool)` it (e.g.
`if let Some(old) = self.coef_.take() { old.release_into(pool); }`), or pair this
with the WR-05 "reset at top of fit" fix so old buffers are released exactly once.

## Info

### IN-01: `host_to_f64` / `f64_to_host` duplicated verbatim across six files

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:374-389`, `crates/mlrs-algos/src/linear/ridge.rs:378-393`, `crates/mlrs-algos/src/decomposition/pca.rs:389-404`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:315-330`, `crates/mlrs-backend/src/prims/cholesky.rs:241-247`, `crates/mlrs-backend/src/prims/svd.rs:446-461`
**Issue:** The identical bytemuck-based `F ↔ f64` reinterpret helpers are copy-pasted
into six modules. A bug fixed in one (or a future `f16`/`bf16` extension) must be
fixed in all six, and the `unreachable!` arms drift independently. **Fix:** Hoist a
single `pub(crate) fn host_to_f64<F: Pod>` / `f64_to_host<F: Pod>` pair into a
shared `mlrs-core` (or `mlrs-backend`) utility module and import it everywhere.

### IN-02: SVD-relabel comments contain an abandoned half-sentence

**File:** `crates/mlrs-backend/src/prims/svd.rs:331-332`
**Issue:** The wide-path relabel comment reads `// Original A is (n × m) where here
rows = n (=orig cols), cols = m... no:` — an in-line self-correction left in the
source. While not in the primary review scope, this file is a direct dependency of
the in-scope PCA/TruncatedSVD wide path; the dangling "... no:" is confusing for
the next reader tracing the wide-case index math. **Fix:** Replace with the single
corrected statement that the lines below already give.

### IN-03: `AlgoError::InvalidNComponents` doc references a nonexistent decision tag

**File:** `crates/mlrs-algos/src/error.rs:29`
**Issue:** The doc comment cites "(D-06 — v1 takes an int `k ≤ min(m, n)`)" but D-06
in this phase's summaries is "no transpose buffers." The n_components-as-int
decision is referenced elsewhere as the minimal-surface choice. A wrong decision
tag in a doc comment misleads anyone cross-referencing the planning log. **Fix:**
Correct the tag (or drop the parenthetical) to the actual decision that scoped
`n_components` to an integer.

### IN-04: Estimator geometry checks use unchecked `n_samples * n_features`

**File:** `crates/mlrs-algos/src/linear/linear_regression.rs:144`, `crates/mlrs-algos/src/linear/ridge.rs:147`, `crates/mlrs-algos/src/decomposition/pca.rs:172`, `crates/mlrs-algos/src/decomposition/truncated_svd.rs:163`
**Issue:** The validate-before-launch shape checks compute `n_samples * n_features`
(and `n_samples * n_components`) with plain `*`, unlike the primitive layer
(`gemm.rs`, `cholesky.rs`, `svd.rs`) which uses `checked_mul`. On 64-bit with the
MAX_DIM/MAX_ROWS caps this cannot overflow in practice, but the estimators accept
the geometry *before* those caps are enforced by the downstream prim, so the
inconsistency is a latent footgun if a caller passes adversarial `usize` shapes.
**Fix:** Mirror the prim layer and use `checked_mul(..).map(..).unwrap_or(true)`
for the estimator-level length checks.

### IN-05: `info_out` doc says "length 2" / "stay 0" inconsistently with the length-3 contract

**File:** `crates/mlrs-backend/src/prims/cholesky.rs:135`
**Issue:** The `SAFETY` comment enumerates the validated element counts as
"(n*n, n*rhs, n*rhs, n*n, 2)" — listing the `info` array as length **2**, while the
actual contract (and the `pool.acquire(3 * elem)` / `from_raw_parts(.., 3)` calls)
is length **3** `[flag, pivot_index, pivot_value]` after the Plan-02 sign-decode
fix. The stale `2` in the safety rationale undercuts the audit trail for the unsafe
launch. **Fix:** Update the comment to `(n*n, n*rhs, n*rhs, n*n, 3)`.

---

_Reviewed: 2026-06-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
