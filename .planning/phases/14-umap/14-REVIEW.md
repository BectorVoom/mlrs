---
phase: 14-umap
reviewed: 2026-06-23T22:26:14Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - crates/mlrs-algos/src/manifold/umap.rs
  - crates/mlrs-algos/src/manifold/umap_internals.rs
  - crates/mlrs-algos/src/manifold/umap_init.rs
  - crates/mlrs-algos/src/manifold/mod.rs
  - crates/mlrs-algos/src/cluster/spectral.rs
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-algos/tests/umap_test.rs
  - crates/mlrs-kernels/src/umap_layout.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-backend/tests/umap_layout_test.rs
  - scripts/gen_oracle.py
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-23T22:26:14Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed the Phase-14 UMAP implementation: the estimator/builder/typestate surface
(`umap.rs`), the host numeric stages (`umap_internals.rs`, `umap_init.rs`), the one
new device kernel (`umap_layout.rs`), the shared spectral recovery refactor
(`spectral.rs` + the two spectral estimators), the test suites, and the oracle
generator. The code is heavily documented and the determinism / cpu-MLIR-safety
discipline is real. The review surfaced one BLOCKER (a correctness divergence
between the `fit` and `transform` distance scales for the Cosine metric), plus
several WARNINGs around silent panics on adversarial input, an unguarded transform
path, and schedule/epoch inconsistencies that are currently masked by relative test
gates.

The dominant risk pattern: correctness defects hidden behind *relative-to-oracle*
property gates (`PROPERTY_EPS`, `TRANSFORM_PROPERTY_EPS`) that cannot detect a
systematic bias because both sides shift together or the slack is wide. Several
findings below are real behavior divergences that the test harness, by design,
will not catch.

## Critical Issues

### CR-01: Cosine `transform` feeds `2(1−cos)` distances while `fit` feeds `1−cos` — divergent membership graph

**File:** `crates/mlrs-algos/src/manifold/umap.rs:951` (and 1009-1015)
**Issue:** In the fit path, `knn_graph` post-scales the Cosine GEMM result from the
squared-Euclidean-of-unit-vectors value `2(1−cos)` back to the true cosine distance
`1−cos` (`cosine_halve`, confirmed in `knn_graph.rs:155,213-215`). The transform
path's `query_train_knn` does **not** apply this halving: `needs_sqrt` is set only
for Euclidean (line 951), and there is no cosine branch, so `top_k` returns the raw
`2(1−cos)` values which flow into `smooth_knn_dist` as `knn_dist`.

The inline comment (lines 946-950) claims "the absolute distance scale is irrelevant
to the membership stage." That is only approximately true: while the membership
exponent `exp(-(d−ρ)/σ)` is scale-invariant under a uniform 2× (ρ and σ both scale),
`smooth_knn_dist` is NOT purely scale-invariant. The σ floor
`MIN_K_DIST_SCALE * mean_ith` (umap_internals.rs:132-136) and the `rho ≤ 0` global
fallback `MIN_K_DIST_SCALE * mean_distances` are absolute floors against the *scaled*
mean, and the binary-search tolerance `SMOOTH_K_TOLERANCE` is compared against a
dimensionless `psum` — these interact non-linearly when σ is near the floor. The net
effect is that for Cosine, `transform` builds a directed-membership graph from a
2×-inflated distance matrix that does not match the graph `fit` would build for the
same neighbor geometry. The two paths are specified to be the same membership
computation (UMAP-04 / D-03); they are not, for Cosine.

This is masked in `transform_property_cosine` because the gate is relative
(`TRANSFORM_PROPERTY_EPS = 0.15`); the calibration doc even records cosine as "mlrs
BEATS umap" — i.e. the bias is large enough to change the score yet the slack
swallows it.

**Fix:** Mirror `knn_graph`'s cosine halving in `query_train_knn` so the transform
KNN distances are on the same `1−cos` scale as fit:
```rust
let needs_sqrt = matches!(knn_metric, knn_graph::Metric::Euclidean);
let cosine_halve = matches!(knn_metric, knn_graph::Metric::Cosine);
// ... after top_k ...
let (tk_val, tk_idx) = top_k::<F>(pool, &dist, m, n, k, needs_sqrt, None, None)?;
let knn_dist_host: Vec<f64> = tk_val
    .to_host(pool)
    .iter()
    .map(|&v| {
        let d = host_to_f64(v);
        if cosine_halve { 0.5 * d } else { d }
    })
    .collect();
```

## Warnings

### WR-01: `compute_membership_strengths` indexes `cols` straight from KNN — host panic (not typed error) on an out-of-range index

**File:** `crates/mlrs-algos/src/manifold/umap_internals.rs:185` and `umap.rs:1116`
**Issue:** `compute_membership_strengths` does `let col = knn_idx[p].round() as usize;`
and stores it; the doc asserts the indices are "already validated `< n`". The value
is then used in the fit path to index a dense affinity:
`affinity[g_rows[e] * n + g_cols[e]] = g_vals[e];` (umap.rs:1116). If any KNN index
were ever ≥ n (a future KNN-prim regression, or a NaN float-encoded index where
`round()` yields a huge value), this is a silent OOB write / panic rather than a
typed error. The "Phase-13 prim guarantees `< n`" claim is an unchecked cross-module
invariant carried across a host round-trip.
**Fix:** Add a bounds check at the COO-consumption boundary — error/clamp in
`compute_membership_strengths` when `col >= n`, or assert `g_cols[e] < n && g_rows[e] < n`
before the affinity write in `run_umap_layout`.

### WR-02: `transform_new_points` divides by `n_components` with no guard — divide-by-zero / shape-trust on the fitted buffer

**File:** `crates/mlrs-algos/src/manifold/umap.rs:582-593`
**Issue:** `let n = embedding_train.len() / n_components;` (line 593) trusts that
`cfg.n_components >= 1` and that `embedding_.len() == n * n_components` exactly, with
no assertion. `fit` guards `n_components >= 1` at `build()` and `n_components < n` at
`fit`, so this is currently unreachable, but the transform path itself performs no
defensive check before the division and before `init_graph_transform` indexes
`embedding_train[col * n_components + d]`. Any future path that constructs a
`Fitted` with `n_components == 0` (or a partially-built buffer) turns this into a
panic deep inside transform rather than a typed error at the boundary.
**Fix:** Assert `n_components >= 1` and `embedding_train.len() % n_components == 0`
at the top of `transform_new_points`, returning a typed error instead of relying on
the `/ n_components` to panic.

### WR-03: `smooth_knn_dist` `k == 1` yields a runaway-then-floored σ; "no non-termination" framing overstated

**File:** `crates/mlrs-algos/src/manifold/umap_internals.rs:97-125`
**Issue:** The σ binary search sums `for j in 1..k`. When `k == 1` (reachable: fit
clamps `k = n_neighbors.min(n-1).max(1)`, so `n == 2, n_neighbors == 1` → `k == 1`),
the inner sum is empty, `psum == 0 < target`, and the `else` branch doubles `mid`
every iteration for all `SMOOTH_N_ITER` iterations, producing σ ≈ `2^64` before the
floor clamps it. The result is finite (no panic) but numerically degenerate. This
matches umap's own `range(1, k)` (parity-correct), but the threat-model claim
(T-14-03/04, "no NaN / non-termination on pathological input") holds only in the
weak "terminates at the iteration cap" sense.
**Fix:** Special-case `k <= 1` to set σ directly to the per-row floor, or downgrade
the doc claim to acknowledge the bounded-but-degenerate `k == 1` case.

### WR-04: Fit/transform `n_epochs` defaults diverge from the committed oracle's `n_epochs=200`

**File:** `crates/mlrs-algos/src/manifold/umap.rs:1134`; `tests/umap_test.rs:351-361,986-994`; `scripts/gen_oracle.py:946,1214,1259`
**Issue:** `run_umap_layout` defaults `n_epochs` to `if n <= 10_000 { 500 } else { 200 }`
when `cfg.n_epochs` is None. The oracle fixtures (`gen_umap_layout` /
`gen_umap_transform`) were generated with `UMAP_N_EPOCHS = 200`. The property tests
(`fit_embedding`, `run_transform_property`) never set `.n_epochs(...)`, so mlrs runs
**500** fit epochs against a **200**-epoch umap reference. The relative gate hides
this, but the "calibrated worst margins" recorded in the `PROPERTY_EPS` doc were
measured against a layout that is NOT epoch-matched to the oracle, weakening the
calibration's reproducibility claim.
**Fix:** Set `.n_epochs(Some(200))` in the layout/transform property tests to match
the oracle, OR regenerate the fixtures at mlrs's 500-default, so the calibration is
like-for-like.

### WR-05: `make_epochs_per_sample` `w_max` fold is not NaN-safe; a NaN weight silently corrupts the schedule

**File:** `crates/mlrs-algos/src/manifold/umap.rs:1027-1045`
**Issue:** `w_max = weights.iter().cloned().fold(0.0_f64, f64::max)`. `f64::max(0.0, NaN)`
returns `0.0`, so a NaN weight is silently treated as 0 and may corrupt `w_max`,
yielding a wrong per-edge sampling schedule for every edge with no error. Membership
values come from `exp` of finite inputs so NaN should not arise today, but there is
no finiteness validation of `g_vals` before the schedule is built, and the failure
mode (silent schedule corruption) is invisible.
**Fix:** Validate `g_vals`/`weights` are all-finite before scheduling (typed error on
violation), or use a NaN-aware reduction.

### WR-06: eig working-buffer aliasing duplicated a third time in `spectral_init` without the load-bearing WR-05 invariant comment

**File:** `crates/mlrs-algos/src/manifold/umap_init.rs:266-268`
**Issue:** `spectral_init` threads `l.handle().clone()` through eig's `out` exactly
like `spectral_embedding.rs:260-264` and `spectral_clustering.rs:285-288`, but those
two carry the full WR-05 soundness comment documenting the two eig-internal
invariants (eig never writes `a_in`; it acquires its w/V/info outputs before
releasing the working buffer) that make the `from_raw`-over-shared-handle aliasing
sound. The umap site repeats the aliasing with only a one-line "WR-05 aliasing
precedent" reference. This unsafe-adjacent pattern now lives in three places; if
eig's internals change, this is the easiest call site to miss.
**Fix:** Copy the full WR-05 invariant comment to `spectral_init`, or factor "eig
with working-buffer reuse" into one helper so the soundness argument lives once.

## Info

### IN-01: Stale/contradictory doc comments diverge from the implemented behavior

**File:** `crates/mlrs-algos/src/manifold/umap.rs:596-602`; `crates/mlrs-kernels/src/umap_layout.rs:23-24`
**Issue:** The block comment inside `transform_new_points` (596-602) is a
stream-of-consciousness "…but the fitted shell does not keep X… Instead transform
receives X_new only…" that contradicts the code, which DOES keep and use `x_train_`
three lines later. Separately, `umap_layout.rs:23-24` states "`fit` launches with
`owners = all n`, `move_other = 1` (two-sided)", but the fit path now uses
`FIT_MOVE_OTHER = 0` (owner-only). Both comments are stale relative to the code.
**Fix:** Delete the contradictory paragraph in `transform_new_points`; correct the
`umap_layout.rs` module doc to `move_other = 0` for the fit path.

### IN-02: Redundant dual-carriage of the Minkowski `p`; dead `let _ = p` in the transform KNN

**File:** `crates/mlrs-algos/src/manifold/umap.rs:890-898,944,1005`
**Issue:** `map_metric` returns `(knn_graph::Metric, f64)` but the exponent already
travels inside `knn_graph::Metric::Minkowski { p }`. `query_train_knn` immediately
discards the tuple `p` (`let _ = p;`, line 1005). The duplication is redundant and
the discard is dead code.
**Fix:** Have `map_metric` return only `knn_graph::Metric` and read `p` from the enum
at the single fit call site (`run_umap_layout:1063`) that needs the scalar.

### IN-03: Reproducibility-critical mixing constants are unnamed magic numbers

**File:** `crates/mlrs-algos/src/manifold/umap.rs:784-787,1257-1260`; `umap_init.rs:1131` (call at umap.rs:1131)
**Issue:** The per-(seed, epoch, edge) substream uses literals `0x9E37_79B9_7F4A_7C15`,
`0x1000_0001`, and the init scale uses `seed ^ 0x5350_4543`. These VALUES are part of
the D-05 byte-identical contract, but they are unnamed and `0x1000_0001` is an
unusual (non-standard) multiplier. A future "cleanup" could change them and silently
break every committed-seed output without a compile error.
**Fix:** Promote them to named `const`s with a comment stating the values are fixed
stream separators whose change alters every reproducible output.

### IN-04: `noisy_scale_coords` `n` / `n_components` params are debug-only

**File:** `crates/mlrs-algos/src/manifold/umap_init.rs:306-314`
**Issue:** `n` and `n_components` are used only in `debug_assert_eq!(coords.len(), n * n_components)`
and nowhere else; in release builds both are dead parameters. Minor API-surface
smell (callers pass redundant shape info).
**Fix:** Drop the params, or document that they exist solely for the debug shape
assertion.

### IN-05: Test helper `host_kmeans_labels` init not robust to non-distinct first-k rows

**File:** `crates/mlrs-algos/tests/umap_test.rs:278-285`
**Issue:** The deterministic test k-means seeds centroids from "the first k rows". On
a degenerate/collapsed embedding the first `k` rows may coincide, leaving a cluster
permanently empty and skewing the downstream-ARI gate (test-harness only, not shipped
code), making the gate flaky rather than failing loudly.
**Fix:** Seed from `k` rows chosen to be distinct/separated, or assert the chosen
init rows are distinct.

---

_Reviewed: 2026-06-23T22:26:14Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
