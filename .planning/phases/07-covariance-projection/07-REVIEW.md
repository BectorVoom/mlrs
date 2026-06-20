---
phase: 07-covariance-projection
reviewed: 2026-06-21T00:00:00Z
depth: standard
files_reviewed: 32
files_reviewed_list:
  - crates/mlrs-algos/src/covariance/empirical_covariance.rs
  - crates/mlrs-algos/src/covariance/ledoit_wolf.rs
  - crates/mlrs-algos/src/covariance/mod.rs
  - crates/mlrs-algos/src/decomposition/incremental_pca.rs
  - crates/mlrs-algos/src/decomposition/mod.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/projection/gaussian.rs
  - crates/mlrs-algos/src/projection/mod.rs
  - crates/mlrs-algos/src/projection/sparse.rs
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/tests/empirical_covariance_test.rs
  - crates/mlrs-algos/tests/incremental_pca_test.rs
  - crates/mlrs-algos/tests/ledoit_wolf_test.rs
  - crates/mlrs-algos/tests/random_projection_test.rs
  - crates/mlrs-backend/src/prims/incremental_svd.rs
  - crates/mlrs-backend/src/prims/kmeans.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/rng.rs
  - crates/mlrs-backend/tests/incremental_svd_test.rs
  - crates/mlrs-backend/tests/rng_test.rs
  - crates/mlrs-py/python/mlrs/__init__.py
  - crates/mlrs-py/python/mlrs/covariance.py
  - crates/mlrs-py/python/mlrs/decomposition.py
  - crates/mlrs-py/python/mlrs/random_projection.py
  - crates/mlrs-py/src/errors.rs
  - crates/mlrs-py/src/estimators/covariance.rs
  - crates/mlrs-py/src/estimators/decomposition.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/projection.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/pyclass_smoke_test.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 7
  info: 5
  total: 12
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Status:** issues_found

## Summary

Phase 7 adds two covariance estimators (EmpiricalCovariance, LedoitWolf), the
streaming IncrementalPCA + its incremental-SVD merge primitive, two random
projection transformers + a host RNG primitive, and the full PyO3/Python surface
for all five.

I verified the core numerical kernels against scikit-learn directly rather than
trusting the in-tree fixtures: the LedoitWolf β/δ/μ closed form, the
IncrementalPCA/incremental-SVD merge finalize (`explained_variance_ = S²/(n−1)`,
ratio denom `= n_total·Σvar_`), the EmpiricalCovariance MLE (BOTH `assume_centered`
arms), the `johnson_lindenstrauss_min_dim` bound (including the sklearn
`int64`-truncation vs Rust `floor` equivalence for positive values), and the
sklearn `gen_batches` port (the `continue`-without-advance + trailing-fold loop) —
all reproduce sklearn's arithmetic. The constructor argument order across the
three layers (Python `__init__` → Rust `#[new]` → algos `new`) is consistent for
every estimator I traced (EmpiricalCovariance, LedoitWolf, IncrementalPCA,
Gaussian/Sparse RP). The `partial_fit` mismatched-dtype-stream path correctly
returns a typed `ValueError` and does NOT panic.

No blocker-class defects (wrong numerical results, crashes, security gaps) were
found. The findings are validation-completeness gaps that diverge from sklearn's
up-front checks, oracle-coverage holes that leave real code paths unverified by
the committed suite, and one test that misrepresents what it exercises.

## Warnings

### WR-01: `n_components <= batch_size` is never validated up front; a valid `batch_size` yields a misleading mid-stream error

**File:** `crates/mlrs-algos/src/decomposition/incremental_pca.rs:280-308`
**Issue:** `fit()` validates `n_components <= min(n_samples, n_features)` and
`batch_size >= 1`, but NOT `n_components <= batch_size`. With e.g.
`batch_size = 1, n_components = 3`, `gen_batches(n, 1, min_batch=3)` emits leading
size-1 batches (verified output: `[(0,1),(1,2),...,(7,10)]` for n=10). The first
size-1 batch reaches `validate_batch`, where `max_nc = b.min(p) = 1`, so it returns
`InvalidNComponents { requested: 3, max: 1 }` — an error whose `max=1` references
the *batch row count*, not the `n_components`/`batch_size` relationship the caller
got wrong. sklearn rejects this up front with "n_components=L must be ≤ batch
number of samples B". Behavior is not *wrong* (it errors rather than mis-computing),
but the diagnostic is confusing and the validation is incomplete vs sklearn.
**Fix:** In `fit()`, after resolving `batch_size`:
```rust
if self.n_components > batch_size {
    return Err(AlgoError::InvalidNComponents {
        estimator: "incremental_pca",
        requested: self.n_components,
        max: batch_size,
    });
}
```

### WR-02: `assume_centered=True` covariance path has zero oracle coverage

**File:** `scripts/gen_oracle.py:1116-1155,1387-1390` (path under test:
`empirical_covariance.rs:178-230`, the `mle_gram_uncentered` host Gram)
**Issue:** `gen_empirical_covariance` accepts `assume_centered`, but `main()` only
emits `assume_centered=False` fixtures. The dedicated `mle_gram_uncentered` branch
(a SEPARATE implementation from the centered `covariance` prim — it builds the
uncentered `Xᵀ·X/n` on the host) is therefore never value-gated against sklearn. I
verified it matches sklearn manually (`EmpiricalCovariance(assume_centered=True)`),
but the committed suite cannot catch a future regression in that branch.
**Fix:** Emit an `assume_centered=True` fixture
(`gen_empirical_covariance(..., assume_centered=True, kind='centered')`) and add a
test fitting `EmpiricalCovariance::new(true, true)` that compares
`covariance_`/`location_`/`precision_`.

### WR-03: IncrementalPCA `whiten=True` re-checks identical attrs; the `WHITEN_VAR_FLOOR` branch is untested

**File:** `crates/mlrs-algos/tests/incremental_pca_test.rs:494-524`
**Issue:** `incremental_pca_transform_whiten_*` calls `assert_attrs` against the
whiten fixture, but those fitted attrs (`components_`/`explained_variance_`/…) are
byte-identical to the nowhiten fixture (whiten only changes `transform`), so this
re-asserts the same numbers. The whitening math is exercised only through
`transform`/`inverse_transform`, and the `WHITEN_VAR_FLOOR` degenerate branch
(`ev <= 1e-12 → scale 1.0`, `incremental_pca.rs:458-464`) is never reached by any
fixture (all retained components have non-trivial variance).
**Fix:** Add a near-rank-deficient design so a retained component has ~0 explained
variance and assert the whitened transform stays finite, OR document the floor as
defensive-only/unreachable-on-fitted-data.

### WR-04: `gen_batches`-fidelity claim in the IncrementalPCA / incremental-SVD tests is false for non-divisible n

**File:** `crates/mlrs-algos/tests/incremental_pca_test.rs:211-222` and
`crates/mlrs-backend/tests/incremental_svd_test.rs:84-99`
**Issue:** Both helpers comment "Stream `gen_batches(n, batch_size)` exactly as
sklearn fit()" but actually do naive `batch_size.min(n - start)` chunking. These
match `gen_batches` ONLY because the fixture is 30×6 with `batch_size=10`
(30 % 10 == 0). For any `n` not divisible by `batch_size`, naive chunking emits a
SHORT trailing batch while the real `gen_batches(min_batch=n_components)` FOLDS the
remainder into the prior batch — a different stream, a different merged state. The
explicit-stream test would silently stop mirroring `fit()` if the geometry changed,
weakening the cross-check it claims to provide.
**Fix:** Drive the explicit stream through the real `gen_batches` port (or call the
estimator's `fit()`), so the test mirrors the one-shot path on any geometry. At
minimum correct the comment to "naive equal chunking (== gen_batches only when n %
batch_size == 0)".

### WR-05: `fit()` then `partial_fit()` silently continues the stream; the wrapper's reset semantics are asymmetric and undocumented

**File:** `crates/mlrs-py/python/mlrs/decomposition.py:96-120`
**Issue:** `IncrementalPCA.fit` builds a fresh `_mlrs` object each call (correct
sklearn reset). `partial_fit` reuses `self._mlrs_obj` when present (line 114). So
`fit(X1); partial_fit(X2)` MERGES X2 into the X1-fitted running state and bumps
`n_samples_seen_` past `len(X1)` — which is sklearn's documented behavior, but here
it is silent and the two entry points have asymmetric reset semantics
(`partial_fit` continues, `fit` resets) that a caller mixing them can easily
misjudge.
**Fix:** Document the reset contract on both methods to match sklearn and add a
pytest pinning `n_samples_seen_` across `fit`/`partial_fit` interleavings.

### WR-06: `gaussian_matrix` exact-zero uniform draw injects a ~37σ outlier

**File:** `crates/mlrs-backend/src/prims/rng.rs:148-159`
**Issue:** The Box–Muller floor is `if u1 <= f64::MIN_POSITIVE { u1 = f64::MIN_POSITIVE }`.
`next_f64()` quantizes to multiples of `2^-53`, so its smallest nonzero value
(~1.1e-16) is ~10^292× larger than `f64::MIN_POSITIVE` (~2.2e-308). The guard thus
only fires on an exact `0.0` draw, and when it does, `r = sqrt(-2·ln(2.2e-308)) ≈
37.6` — a 37-sigma entry injected into the projection matrix. Deterministic and
rare, but it skews the very moment statistics the property tests assert.
**Fix:** Floor at the stream's resolution instead, e.g.
`if u1 < 2.0_f64.powi(-53) { u1 = 2.0_f64.powi(-53); }` (bounds worst-case r to
~8.6σ), or redraw on the zero.

### WR-07: `next_below(0)` is reachable only via a future caller and degrades to a divide-by-zero panic in release builds

**File:** `crates/mlrs-backend/src/prims/rng.rs:86-102`
**Issue:** `next_below` guards its contract with `debug_assert!(bound >= 1)` (a
no-op in release) and special-cases `bound == 1`, but a `bound == 0` in a release
build skips the assert, skips the `== 1` branch, and reaches `u64::MAX % bound`
(line 95) → an arithmetic panic. All current callers (`permutation`,
`kmeanspp_sample`) pass `>= 1`, so it is not presently reachable, but the public
`pub fn` exposes a foot-gun whose failure mode is an opaque modulo panic rather
than a typed error.
**Fix:** Early-return `0` for `bound == 0`, or change the signature to return a
`Result`/take a `NonZeroU64`, so the contract is enforced in release too.

## Info

### IN-01: Misleading "RESEARCH Pattern 4" attribution on the Gaussian generator

**File:** `crates/mlrs-backend/src/prims/rng.rs:109-113`
**Issue:** The Gaussian-matrix doc cites "RESEARCH Pattern 4", which the Achlioptas
section (line 167) also cites. The Gaussian case is the `N(0, 1/n_components)`
pattern, not the Achlioptas sparse one; one attribution is wrong.
**Fix:** Point the Gaussian doc at its own pattern reference.

### IN-02: Dead `_y` parameter on every unsupervised `fit`

**File:** `crates/mlrs-algos/src/covariance/empirical_covariance.rs:139`,
`ledoit_wolf.rs:126`, `projection/gaussian.rs:159`, `projection/sparse.rs:131`
**Issue:** These unsupervised estimators take and ignore `_y` per the shared `Fit`
trait. Intentional (trait uniformity for Phase-10 MBSGD reuse, documented in
`traits.rs`) — noted only so a reader does not mistake it for unfinished wiring.
**Fix:** None required; the `_` prefix already documents intent.

### IN-03: `decomposition/mod.rs` re-exports only `IncrementalPCA`

**File:** `crates/mlrs-algos/src/decomposition/mod.rs:25`
**Issue:** Only `IncrementalPCA` is `pub use`-d at the module root; `Pca` /
`TruncatedSvd` need the full path. Inconsistent surface, harmless.
**Fix:** Optionally add `pub use pca::Pca; pub use truncated_svd::TruncatedSvd;`.

### IN-04: `LedoitWolf` keeps an inert `ddof` local for documentation

**File:** `crates/mlrs-algos/src/covariance/ledoit_wolf.rs:193-194`
**Issue:** `let ddof: u32 = 0;` then `g / (n - ddof as f64)` is just `g / n`. Reads
as if ddof were configurable when it is hard-pinned to the MLE.
**Fix:** Inline `g / n` with the existing comment, or hoist to a clearly-const
`const DDOF: f64 = 0.0;`.

### IN-05: `EmpiricalCovariance`/`LedoitWolf` mutate device state but expose host accessors taking `&self` + `&BufferPool` — fine, but `to_host` is non-trivial work behind a getter

**File:** `crates/mlrs-algos/src/covariance/empirical_covariance.rs:101-128`,
`ledoit_wolf.rs:86-115`
**Issue:** `covariance_()`, `location_()`, `precision_()` each run a device→host
copy on every call (the Python `@property` accessors invoke them repeatedly). Not a
correctness bug, but a `@property` that downloads from the device each access can
surprise callers in a loop. Out of v1 perf scope; flagged for awareness.
**Fix:** None required for v1; consider memoizing the host copy if profiling shows
repeated-access hotspots.

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
