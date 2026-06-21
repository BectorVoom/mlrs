---
phase: 09-spectral-family
reviewed: 2026-06-21T03:32:40Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/laplacian.rs
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/spectral.rs
  - crates/mlrs-py/src/lib.rs
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 9: Code Review Report

**Reviewed:** 2026-06-21T03:32:40Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the Phase-09 "Spectral Family" source: the `laplacian` prim (PRIM-09)
plus its three new elementwise kernels, the `SpectralEmbedding` /
`SpectralClustering` estimators (SPECTRAL-01/02), the `NSamplesExceedsMaxDim`
error variant, and the two PyO3 wrappers. The numerical core — the scipy
`_laplacian_dense` reproduction, the typed-zero degree guard, the pinned
sklearn `_spectral_embedding` recovery order, and the descending-to-ascending
eig slice — is implemented carefully and is well-corroborated by the committed
oracle gates (f64 ≈1e-15, f32 inside band, exact labels up to permutation).

No BLOCKER-class correctness, security, or data-loss defects were proven. The
findings below are robustness, buffer-accounting, and sklearn-parity concerns.
Two are worth fixing before this ships beyond the n≤64 cap: an inner-KMeans
buffer-accounting leak in `SpectralClustering::fit` (WR-01) and the
mutex-poisoning failure mode where a single device error during `fit`
permanently bricks the process-global pool (WR-02). The remaining warnings are
sklearn-semantics divergences (n_neighbors clamp, gamma=0) and a fragile
aliased-handle eig-reuse pattern that is sound today but easy to break.

## Warnings

### WR-01: Inner KMeans device buffers leak the pool accounting in `SpectralClustering::fit`

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:268-274`
**Issue:** `let mut kmeans = KMeans::<F>::new(...)` is a function-local that, after
`kmeans.fit(...)`, owns fitted device buffers — `cluster_centers_`
(`k × n_components`) and its internal `labels_` (`DeviceArray<.., i32>`, length
`n`) — see `kmeans.rs:92-95`. `DeviceArray` has no `Drop` impl
(`device_array.rs`), so when `kmeans` falls out of scope at the end of `fit`
those buffers are dropped WITHOUT `release_into(pool)`. Their `acquire`d bytes
are never returned to the free-list and `live_bytes` is never decremented, so a
re-fit (or a long-lived estimator looped over data) monotonically grows
`live_bytes` and forfeits buffer reuse — the FOUND-05 memory invariant this
phase is supposed to uphold. Every other buffer in `fit` (`a`, `dd`, `l`,
`maps_dev`, the prior `labels_`) is explicitly released; only the inner KMeans'
state is silently dropped.
**Fix:** Release the inner KMeans' device state before it drops, e.g. take its
labels and centers and `release_into(pool)` (or give `KMeans` a `release(pool)`
teardown that the composing estimator calls). Concretely, after copying labels:
```rust
let labels_host = kmeans.labels(pool)?;
kmeans.release_into(pool); // new: returns cluster_centers_/labels_ to the pool
```
Confirm with the PoolStats `live_bytes`-conserved gate across a repeated
`SpectralClustering::fit`, mirroring the laplacian memory gate.

### WR-02: A device error during `fit` poisons the process-global pool mutex permanently

**File:** `crates/mlrs-py/src/estimators/spectral.rs:132,162,171,291,328`
**Issue:** Every accessor and `fit` body locks the process-global pool with
`crate::global_pool().lock().expect("pool mutex")`. The new `fit` paths can
panic while holding that lock — `laplacian.rs:128-129`
`row_reduce(...).expect("shared path is never plane-gated to None")`,
`device_array.rs:117` `read_one(...).expect("device read-back ...")`, and the
several `unsafe { ArrayArg::from_raw_parts(...) }` launches — on a real device
fault, OOM, or unsupported-op. A panic inside the `py.detach` closure while the
`MutexGuard` is held POISONS the mutex; thereafter every `.expect("pool mutex")`
in the whole module (all estimators, not just spectral) panics, converting one
recoverable device error into a permanent process-wide brick. This is a
robustness/DoS-class regression that the new infinity-free/SharedMemory-free
kernels make MORE likely to surface (the cpu-MLIR backend panics on unsupported
shapes per the project memory notes).
**Fix:** Recover from poisoning instead of `.expect()`: `match
global_pool().lock() { Ok(g) => g, Err(p) => p.into_inner() }`, or translate the
poison into a typed `PyErr`. The pool data itself is not left in a torn state by
a panicked compute call (handles are ref-counted), so `into_inner()` recovery is
safe and keeps a single bad `fit` from killing the interpreter session.

### WR-03: `n_neighbors > n_samples` is a hard error; sklearn silently clamps

**File:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs:195-201` and
`crates/mlrs-algos/src/cluster/spectral_clustering.rs:214-220`
**Issue:** The `"nearest_neighbors"` path rejects `self.n_neighbors > n_samples`
with `InvalidK`. sklearn's `kneighbors_graph` does NOT error here — for the
default `n_neighbors=10` and any `n_samples <= 10` it effectively uses all
available neighbors (NearestNeighbors caps at `n_samples`). With the SE default
`n_neighbors=10`, ANY dataset with `n_samples <= 10` (well within the n≤64 cap)
that a user runs through the default constructor will raise instead of producing
the sklearn-equivalent embedding. The committed oracle (n=12) dodges this by one
row, so the gate does not catch it. This is a real default-path parity divergence
for small inputs, not a theoretical edge.
**Fix:** Clamp rather than reject: `let k = self.n_neighbors.min(n_samples);` and
validate only `k >= 1`. Match sklearn's "use min(n_neighbors, n_samples)"
behavior so the default constructor works for all `n_samples` in `[1, 64]`.

### WR-04: `gamma == 0.0` (and negative gamma) pass the `is_finite()` validation

**File:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs:176-181` and
`crates/mlrs-algos/src/cluster/spectral_clustering.rs:195-200`
**Issue:** The rbf gamma guard only rejects non-finite values (`!gamma64.is_finite()`).
`gamma = 0.0` passes, producing `exp(-0·d²) = 1` for ALL pairs — a constant
all-ones affinity → a degenerate fully-connected graph whose normalized
Laplacian spectrum is meaningless (every off-diagonal `−1/(n-1)`), silently
yielding garbage embeddings/labels rather than an error. A negative gamma
(`exp(+|γ|·d²)`) blows the affinity up monotonically with distance — also
silently wrong. sklearn's contract for the kernel coefficient is
`Interval(Real, 0, None, closed='neither')` (strictly positive), which the
sibling `KernelRidge`/`KernelDensity` guards in `error.rs` cite but this path
does not enforce. `SpectralEmbedding`'s `None → 1/n_features` default is always
positive, but a user-supplied or `SpectralClustering` literal gamma can be `0`
or negative.
**Fix:** Tighten the guard to sklearn's contract: reject `gamma <= 0.0` (not just
non-finite). Reuse `InvalidGamma` with a message naming the `> 0` requirement, or
add the positivity check alongside the finiteness one.

### WR-05: eig `out`-reuse aliases the same handle as `&l` — sound today, silently breakable

**File:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs:225-230` and
`crates/mlrs-algos/src/cluster/spectral_clustering.rs:244-249`
**Issue:** `l_out = DeviceArray::from_raw(l.handle().clone(), n*n)` then
`eig(pool, &l, n, Some(l_out))` passes TWO `DeviceArray`s (`&l` and `l_out`)
that wrap the SAME ref-counted cubecl handle. eig reads `a` (=`&l`) in
`compute_thresholds` AND uses `out` (=`l_out`) as the kernel working input, then
`l_out.release_into(pool)` files that handle into the free-list; the subsequent
`drop(l)` drops the second clone. This is correct ONLY because (a) eig's kernel
reads `a_in` and never writes it, and (b) eig acquires its `w/v/info` outputs
BEFORE releasing `l_out`, so the freed handle is not re-handed mid-call. Both are
load-bearing internal invariants of `eig.rs` that are undocumented at this call
site; if eig ever writes its working input in place, or reorders the
acquire/release, this becomes a use-after-free / aliased-write with no
compile-time signal. The `drop(l); // released by eig` comment also
mis-describes the mechanics (eig releases the CLONE `l_out`, not `l`; `l`'s
handle clone is merely dropped).
**Fix:** Either (a) stop aliasing — pass `None` for `out` and let eig read `&l`
directly (the n≤64 working buffer is tiny; the `out`-reuse saves one `n²`
allocation that the memory gate does not require here), or (b) add an assertion
/ comment block documenting the "eig reads-only its input and acquires outputs
first" invariant this reuse depends on, and correct the misleading `drop(l)`
comment.

### WR-06: `recover_embedding` / `recover_maps` are duplicated verbatim except one slice bound

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:362-417` vs
`crates/mlrs-algos/src/cluster/spectral_embedding.rs:321-377`
**Issue:** The two host recovery helpers (slice-smallest → `/dd` → sign-flip →
transpose) plus the `host_to_f64`/`f64_to_host` bytemuck pair are copy-pasted
across both files; they differ ONLY by `m = n_components + 1` (drop_first=true)
vs `m = n_components` and the kept-row offset. The sign-flip, the `col = n-1-r`
descending-slice math, and the `/dd` step — the exact pieces the SUMMARY calls
"load-bearing, a wrong order fails the value match" — now exist in two places
that must be kept bit-identical by hand. A future fix to the pinned sklearn order
(or a `dd==0` edge) applied to one and not the other silently desynchronizes the
embedding vs the clustering path. The 09-04 SUMMARY acknowledges the deliberate
duplication for file-disjointness, but the parallel-safety rationale expired once
both files landed.
**Fix:** Factor the shared recovery into one `pub(crate)` helper taking a
`drop_first: bool` (the only real difference), e.g. in a `cluster/spectral.rs`
or as `recover(v, dd, n, n_components, drop_first)`. Collapse the two
`host_to_f64`/`f64_to_host` copies into a shared `mlrs-algos` util (they are
already triplicated with `eig.rs`).

## Info

### IN-01: Stale module doc claims KMeans init is "injected for the oracle"

**File:** `crates/mlrs-algos/src/cluster/mod.rs:9-10`
**Issue:** The doc says `KMeans` uses "k-means++ init (injected for the oracle,
D-09)". Phase-9 D-10 explicitly REJECTS init-injection for SpectralClustering and
uses `KMeans::new` (no injection); the comment predates Phase 9 and now
contradicts the spectral usage a reader arrives from.
**Fix:** Note that SpectralClustering reuses `KMeans::new` with the non-injected
kmeans++ path (D-10), or drop the "(injected for the oracle)" parenthetical.

### IN-02: Stale "Wave-0 scaffold status" block describes `todo!()` bodies that are now filled

**File:** `crates/mlrs-py/src/estimators/spectral.rs:26-33`
**Issue:** The module doc still says "This is the 09-01 Wave-0 COMPILING STUB ...
every device-compute body delegates to the algos `fit`/accessor stubs, which are
`todo!()` until the Wave-2/3 plans". Both algos bodies and the PyO3 bodies are
fully implemented as of 09-03/09-04; the doc misleads anyone auditing whether
the surface is live.
**Fix:** Update the status block to "filled (09-03/09-04)" and remove the
`todo!()` language.

### IN-03: `error.rs` enum-level doc omits the Phase-9 variant from its summary

**File:** `crates/mlrs-algos/src/error.rs:18-28`
**Issue:** The `AlgoError` doc comment enumerates the failure classes
(InvalidNComponents, InvalidAlpha, InvalidK, ... NotConverged) but never mentions
`NSamplesExceedsMaxDim`, added in this phase. Minor doc drift; the variant itself
is well-documented inline.
**Fix:** Add `NSamplesExceedsMaxDim` (the spectral dense-eig cap) to the
enum-level summary list.

### IN-04: `prims/mod.rs` / `lib.rs` comments still describe `laplacian` as a `todo!()` stub

**File:** `crates/mlrs-backend/src/prims/mod.rs:25-29` and
`crates/mlrs-py/src/lib.rs:62,134` ("12 estimator wrappers")
**Issue:** `prims/mod.rs:28` says the `laplacian` "compute path `todo!()` until
09-02"; it is filled. Separately, `lib.rs:62`/`:134` and the `estimators/mod.rs`
header still say "12 `#[pyclass]` estimator wrappers" while the module now
registers 19 (the original 12 plus Phase-7/8/9 additions, including the two
spectral classes registered at `lib.rs:176-177`). Comment-vs-code count drift.
**Fix:** Update the `laplacian` stub comment to "filled (09-02)" and correct the
estimator count (or make it non-numeric, e.g. "all estimator wrappers").

### IN-05: `fit_predict` round-trips labels host→device→host needlessly

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs:287-296`
**Issue:** `fit_predict` calls `self.fit(...)`, then `self.labels(pool)?` (a
device→host read-back), then `DeviceArray::from_host(pool, &labels)` (a
host→device upload) to return a fresh device buffer. The labels were already
device-resident in `self.labels_` immediately after `fit`. The extra read-back +
re-upload is wasted work (and an extra unmetered copy) on every `fit_predict`.
Not incorrect — the values are identical — but avoidable.
**Fix:** Return a clone/handle of the already-resident `labels_` (or document why
a detached fresh buffer is required for the sklearn `fit_predict` contract).

---

_Reviewed: 2026-06-21T03:32:40Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
