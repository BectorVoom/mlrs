---
phase: 09-spectral-family
reviewed: 2026-06-21T04:30:00Z
depth: standard
files_reviewed: 18
files_reviewed_list:
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/src/cluster/spectral.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/tests/spectral_clustering_test.rs
  - crates/mlrs-algos/tests/spectral_embedding_test.rs
  - crates/mlrs-backend/src/prims/laplacian.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/tests/laplacian_test.rs
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-py/Cargo.toml
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/spectral.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/spectral_smoke_test.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 9: Code Review Report

**Reviewed:** 2026-06-21
**Depth:** standard
**Files Reviewed:** 18 (the spectral-family slice plus the shared `cluster/spectral.rs` helper pulled in via `mod.rs`, and the three test files)
**Status:** issues_found

## Summary

The Phase-9 spectral family (`SpectralEmbedding` SPECTRAL-01, `SpectralClustering`
SPECTRAL-02, the `laplacian` PRIM-09 primitive, the three new map kernels
`zero_diag_copy`/`degree_guard`/`laplacian_map`, the shared host-recovery helper
`cluster/spectral.rs`, and the PyO3 wrappers) was reviewed at standard depth. Both
`mlrs-algos` and `mlrs-py` type-check clean under `--features cpu --tests`.

The numerical core is sound. I traced the validate-before-launch ordering (the
`n_samples > 64` cap and geometry guards genuinely precede every affinity / Laplacian
/ eig / KMeans launch in both estimators), the typed-zero `degree_guard` (no
division by zero, no NaN/inf on isolated nodes — confirmed against the
`laplacian_map` divisor `dd[i]*dd[j]` which is always `>= 1*1`), the `/dd`
recovery + deterministic sign flip + drop-first order in `recover`, and the WR-05
eig out-buffer aliasing (sound against the eig.rs read-only-input / acquire-before-
release invariants). The oracle fixtures exist and the generators match the estimator
defaults (SE rbf gamma `1/n_features`, SC rbf gamma `1.0` literal). No injection,
secret, or unsafe-launch defect found — hence no Critical (BLOCKER).

The four warnings are robustness/correctness-adjacent. The two highest-value: WR-01
(the mutex-poison recovery preserves memory safety but leaks the `live_bytes`
accounting the memory-gate invariant depends on, contradicting its own doc), and
WR-02 (re-`fit` of an already-fitted PyO3 wrapper silently discards the user's
hyperparameters and reverts to hardcoded defaults).

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: Mutex-poison recovery silently corrupts pool accounting (contradicts its own safety doc)

**File:** `crates/mlrs-py/src/lib.rs:96-115`
**Issue:** `lock_pool()` recovers a poisoned mutex via `poisoned.into_inner()`. The
doc asserts: *"The pool data is NOT left torn by a panicked compute call … so
`into_inner()` recovery is safe."* That holds for **memory safety**, not for the
pool's accounting invariant. `BufferPool::acquire` (`pool.rs:107-118`) bumps
`live_bytes`/`peak_bytes` at acquisition time, and a spectral `fit` acquires many
buffers (affinity `A`, `L`, `dd`, eig `w`/`V`/`info`, `maps`, the inner KMeans
buffers) and releases them incrementally across the function. If a panic unwinds the
`py.detach` closure mid-`fit` while the guard is held (a device fault, eig assertion,
OOM, or the cpu-MLIR SharedMemory panic the project memory notes warn about), every
buffer acquired-but-not-yet-released leaves its bytes permanently added to
`live_bytes`. After recovery, `live_bytes` is monotonically inflated for the rest of
the process — the exact FOUND-05 conservation property the `laplacian_test.rs`
`memory_gate` and the kmeans/embedding re-fit gates assert. The recovery thus turns a
recoverable device error into permanent silent corruption of the leak-detection
metric, directly contradicting the "not left torn" claim.
**Fix:** Either qualify the doc (state that accounting counters may be left inflated
after a recovered poison and must not be trusted), or reconcile on recovery — rebuild
`live_bytes` from the live free-list/handle truth rather than the running counter:
```rust
Err(poisoned) => {
    let mut guard = poisoned.into_inner();
    guard.reconcile_live_bytes(); // recompute from live handles, not the counter
    guard
}
```

### WR-02: Re-`fit` of a fitted PyO3 wrapper discards user hyperparameters and reverts to defaults

**File:** `crates/mlrs-py/src/estimators/spectral.rs:122-130` and `272-289`
**Issue:** Both `fit` methods read hyperparameters out of `self.inner` only in the
`Unfit` arm; the catch-all arm fabricates the sklearn defaults:
```rust
_ => (2, "nearest_neighbors".to_string(), None, 10),          // SE
_ => (8, None, "rbf".to_string(), 1.0, 10, 0),                // SC
```
The catch-all is reached whenever `self.inner` is already `F32(..)`/`F64(..)` — i.e.
on a **second** `fit` of the same object. The user's constructor arguments
(`n_components`, `affinity`, `gamma`, `n_clusters`, `seed`, …) are then silently
discarded and the estimator re-fits with the defaults, producing a wrong model with
no error. sklearn semantics require `est.fit(X1); est.fit(X2)` to honor the
constructor params on every call. This follows a pre-existing pattern in
`kernel.rs:179`/`kernel.rs:370`, so it is a latent issue carried into the new
wrappers rather than introduced fresh — but the compiled surface is wrong on its own
terms regardless of whether the Python shim happens to always construct fresh objects.
**Fix:** Persist the constructor hyperparameters on the `#[pyclass]` struct (not only
in the `Unfit` enum arm) so they survive into the fitted arms, and read them from
there on every `fit`. At minimum the catch-all must not fabricate defaults — re-extract
from a persisted params field or return a typed error rather than silently mis-fitting.

### WR-03: `fit_predict` round-trips labels host→device→host→device needlessly

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:323-332`
**Issue:** `fit` already materializes `labels_host` on the host (line 301), uploads it
to `labels_dev` (lines 309-310), and stores it. `fit_predict` then calls
`self.labels(pool)` which copies that device buffer **back** to the host (line 330),
and `DeviceArray::from_host` re-uploads it (line 331) — two extra full round-trips of
the label vector per call. The project treats copy efficiency as a first-class,
per-phase-verified constraint (CLAUDE.md "Memory: zero-copy, buffer reuse, minimal
copies … verified per phase"). Not a correctness bug, but it violates the stated
efficiency contract on a convenience path.
**Fix:** Have `fit_predict` build the returned device buffer directly from the
`labels_host` `fit` already holds (e.g. an internal helper that returns a device clone
of `self.labels_`), avoiding the host hop.

### WR-04: Inconsistent pool-lock helper undermines the WR-01 poison recovery

**File:** `crates/mlrs-py/src/estimators/spectral.rs:132, 163, 171, 291, 328` vs `crates/mlrs-py/src/estimators/kernel.rs:183, 375` and `crates/mlrs-py/src/dispatch.rs:33, 184`
**Issue:** The poison-recovering `lock_pool()` was added (prior WR-02 fix) so a
panicked `fit` cannot permanently brick the interpreter. The new spectral wrappers
correctly use `crate::lock_pool()` everywhere. But the recovery only delivers its
benefit if **every** lock site uses it: a single surviving
`global_pool().lock().expect("pool mutex")` will panic-on-poison and re-brick. Grep
shows `kernel.rs` and the canonical `dispatch.rs` doc example still use the panicking
`.lock().expect(...)` form, so the codebase is inconsistent about which lock helper is
authoritative and the brick-prevention is only partial.
**Fix:** Make `lock_pool` the single sanctioned lock path project-wide — replace the
remaining `global_pool().lock().expect(...)` sites (including `kernel.rs` and the
`dispatch.rs` doc skeleton) — or explicitly document that mixing lock helpers defeats
the recovery. Pairs with WR-01's accounting reconcile.

## Info

### IN-01: Unused `rng` binding on the degenerate SpectralEmbedding fixture path

**File:** `scripts/gen_oracle.py:1631-1643`
**Issue:** `rng = np.random.default_rng(seed)` is created unconditionally, but the
`degenerate=True` branch builds `x` from `np.linspace`/`np.cos`/`np.sin` and never
uses `rng`. Dead assignment on that path (harmless; the non-degenerate path uses it).
**Fix:** Move the `rng` creation into the `else` branch, or comment that it is
intentionally seeded-but-unused for the degenerate geometry.

### IN-02: `knn_connectivity_affinity` duplicated verbatim across the two estimators

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:340-384` and `crates/mlrs-algos/src/cluster/spectral_embedding.rs:296-340`
**Issue:** `knn_connectivity_affinity` is byte-for-byte identical between the two
estimators (the comment literally says "Mirrors SpectralEmbedding verbatim"). WR-06
already factored the host recovery math into the shared `cluster/spectral.rs` module
*precisely* because verbatim duplication risks silent desync on a future fix; this
connectivity builder is the remaining duplicated block of the same character.
**Fix:** Move it into `crate::cluster::spectral` as a free function
`knn_connectivity_affinity::<F>(pool, x, n, d, k)` and call from both estimators, so
the two affinity builders cannot drift.

### IN-03: SC gamma underflow accepted (sklearn-parity question)

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:202-207`, `crates/mlrs-algos/src/cluster/spectral_embedding.rs:184-189`
**Issue:** The rbf branch rejects `!(gamma64 > 0.0)` (catching `0`, NaN, ±inf — good),
but a finite-positive f32 gamma that underflows to effective-zero in
`F::exp(-gamma*dist)` yields a near-constant all-ones affinity (the same degenerate
graph the `gamma == 0` rejection targets). The exact-zero boundary is guarded;
effective-zero underflow is not. sklearn also only rejects `gamma <= 0`, so this is
likely intended parity, not a defect.
**Fix:** No action required if sklearn parity is the contract (leave as-is). Flagged
so the contract is an explicit decision rather than an accident.

---

_Reviewed: 2026-06-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
