---
phase: 09-spectral-family
fixed_at: 2026-06-21T00:00:00Z
review_path: .planning/phases/09-spectral-family/09-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 9: Code Review Fix Report

**Fixed at:** 2026-06-21
**Source review:** .planning/phases/09-spectral-family/09-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (WR-01..WR-06; 0 Critical)
- Fixed: 6
- Skipped: 0

All five required correctness gates stayed green on `--features cpu` after every
fix:
`laplacian_test` (4), `spectral_embedding_test` (5), `spectral_clustering_test` (3),
`spectral_smoke_test` (2), `kmeans_test` (6). No numerical output changed; the
exact-label and embedding value/subspace gates still pass.

## Fixed Issues

### WR-01: Inner KMeans device buffers leak the pool accounting in `SpectralClustering::fit`

**Files modified:** `crates/mlrs-algos/src/cluster/kmeans.rs`, `crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** 7f620bf
**Applied fix:** Added a `KMeans::release_into(pool)` teardown that consumes
`self` and returns `cluster_centers_` + `labels_` to the pool free-list (both
`Option`-guarded; `DeviceArray` has no `Drop`). `SpectralClustering::fit` now
calls `kmeans.release_into(pool)` after copying the labels to the host and before
the function-local KMeans drops, so the inner KMeans' acquired bytes no longer leak
`live_bytes` across re-fits (the FOUND-05 invariant). `kmeans_test` (incl.
predict/inertia) and `spectral_clustering_test` both stay green — no regression.

### WR-02: A device error during `fit` poisons the process-global pool mutex permanently

**Files modified:** `crates/mlrs-py/src/lib.rs`, `crates/mlrs-py/src/estimators/spectral.rs`
**Commit:** 9c00da3
**Applied fix:** Added a `pub(crate) fn lock_pool()` helper next to `global_pool()`
that recovers from poisoning via `match global_pool().lock() { Ok(g) => g, Err(p) =>
p.into_inner() }`. Replaced all 5 `global_pool().lock().expect("pool mutex")` sites
in `spectral.rs` (`fit` ×2, the `embedding_f32`/`_f64` and `labels_` accessors) with
`crate::lock_pool()`. A panicking device call inside `py.detach` no longer bricks the
interpreter session — the pool data is not torn (ref-counted handles unwind before
any half-write), so `into_inner()` recovery is sound. Scoped to the spectral module
per the review (the other estimators' `.expect("pool mutex")` sites were left
unchanged — out of this finding's scope).

### WR-03: `n_neighbors > n_samples` is a hard error; sklearn silently clamps

**Files modified:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs`, `crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** d36c46f
**Applied fix:** The `"nearest_neighbors"` path now clamps `let k =
self.n_neighbors.min(n_samples)` and only rejects `k < 1` (instead of hard-erroring
on `n_neighbors > n_samples`). Threaded the clamped `k` into
`knn_connectivity_affinity` as a parameter (was reading `self.n_neighbors`). Matches
sklearn's `kneighbors_graph` "use min(n_neighbors, n_samples)" behavior, so the SE
default constructor (`n_neighbors=10`) now works for all `n_samples` in `[1, 64]`.
`knn_affinity` and the SE/SC value gates stay green.

### WR-04: `gamma == 0.0` (and negative gamma) pass the `is_finite()` validation

**Files modified:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs`, `crates/mlrs-algos/src/cluster/spectral_clustering.rs`, `crates/mlrs-algos/src/error.rs`
**Commit:** f863f15
**Applied fix:** Tightened the rbf gamma guard in both estimators from
`!gamma64.is_finite()` to `!(gamma64 > 0.0)` (which also subsumes the non-finite
case — NaN and ±inf both fail `> 0.0`), reusing `AlgoError::InvalidGamma` per
sklearn's `Interval(Real, 0, None, closed='neither')` contract. Broadened the
`InvalidGamma` error message to "must be a finite value > 0" so it reads correctly
for both the spectral (`> 0`) and the pre-existing KernelRidge (finite) callers.

### WR-05: eig `out`-reuse aliases the same handle as `&l` — sound today, silently breakable

**Files modified:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs`, `crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** 2a7d420
**Applied fix:** Chose review option (b) — documentation-only, zero behavior change
(the safest choice for the verified phase; option (a)'s extra n² allocation was
avoided). Added a comment block at both call sites naming the two load-bearing
eig-internal invariants the aliasing reuse depends on: (1) eig READS `a_in` and never
writes it, and (2) eig ACQUIRES its `w/V/info` outputs BEFORE releasing the `out`
working buffer. Both were verified against the current `eig.rs` source (the
`a_in_owned.release_into(pool)` happens after the `pool.acquire` calls; the kernel
only reads `a_in`). Corrected the misleading `drop(l)` comment — eig releases the
CLONE threaded through `out`, and `drop(l)` releases `l`'s remaining handle clone.

### WR-06: `recover_embedding` / `recover_maps` are duplicated verbatim except one slice bound

**Files modified:** `crates/mlrs-algos/src/cluster/spectral.rs` (new), `crates/mlrs-algos/src/cluster/mod.rs`, `crates/mlrs-algos/src/cluster/spectral_embedding.rs`, `crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** baf21e6
**Applied fix:** Created a new `pub(crate)` module `cluster/spectral.rs` holding a
single `recover(v_host, dd, n, n_components, drop_first)` helper (the `drop_first:
bool` is the only real difference — `m = n_components + 1` & row offset 1 for SE,
`m = n_components` & offset 0 for SC) plus the shared `host_to_f64`/`f64_to_host`
bytemuck pair. Both estimators now import and call the shared helper; their local
`recover_embedding`/`recover_maps` and the duplicated conversion helpers were
removed. The load-bearing recovery ORDER (slice ascending → `/dd` → sign-flip →
drop-first/transpose) is now defined once, so the embedding and clustering paths
stay bit-identical by construction. Registered the module as `pub(crate) mod
spectral;` in `cluster/mod.rs`. The SE/SC value + exact-label gates confirm identical
numerical output (no behavior change).

---

_Fixed: 2026-06-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
