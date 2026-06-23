---
phase: 14-umap
reviewed: 2026-06-24T00:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - crates/mlrs-algos/src/cluster/spectral.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/src/manifold/mod.rs
  - crates/mlrs-algos/src/manifold/umap.rs
  - crates/mlrs-algos/src/manifold/umap_init.rs
  - crates/mlrs-algos/src/manifold/umap_internals.rs
  - crates/mlrs-algos/tests/umap_test.rs
  - crates/mlrs-backend/tests/umap_layout_test.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-kernels/src/umap_layout.rs
  - scripts/gen_oracle.py
findings:
  critical: 3
  warning: 5
  info: 4
  total: 12
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-24
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed the Phase-14 UMAP implementation: the host orchestration (`umap.rs`),
host numeric stages (`umap_internals.rs`, `umap_init.rs`), the one new device
kernel (`umap_layout.rs`), the shared spectral recovery (`spectral*.rs`), and the
oracle generator + tests. The deterministic host stages (smooth-kNN, membership,
fuzzy union, a/b LM fit) are faithful ports and are value-gated against committed
umap-learn fixtures; those look sound.

The serious problems are in the **stochastic SGD layout** and **its interaction
with the device kernel**. Three of them are correctness/determinism blockers:

1. The fit-path kernel launches one cube per owner **concurrently** while
   `move_other=1` has each cube write its *neighbour's* coordinates — a
   cross-cube read-write data race that produces non-deterministic results on any
   genuinely parallel backend (wgpu/rocm/cuda) and silently breaks the D-05
   byte-identical reproducibility contract that `reproducible_f64` asserts.
2. `Umap::fit` never validates `n_components < n`; with spectral init and a small
   `n` this underflows `usize` inside `recover` (`n - 1 - r`) and panics.
3. The negative-sample target index in the fit path is drawn against `n`
   (owner-local count) but is interpreted by the kernel as a **global vertex
   index**, which is correct for fit only because `n == n_vertices` there — but
   the symmetric-graph double-update plus `move_other` semantics double-count
   every undirected edge relative to umap-learn's single-pass loop (a structural
   divergence masked by the relative property gate).

The reproducibility test only runs on the cpu-MLIR gate (sequential execution),
so the race in #1 is invisible in CI today; it will surface the moment the gate
includes a parallel backend.

No structural pre-pass (`<structural_findings>`) was supplied, so this report is
entirely narrative.

## Critical Issues

### CR-01: Concurrent cross-cube write race in the fit-path layout kernel breaks determinism on parallel backends

**File:** `crates/mlrs-kernels/src/umap_layout.rs:153-157`, driven from `crates/mlrs-algos/src/manifold/umap.rs:1265` and `:1277-1294`

**Issue:** `host_epoch_driver` launches the kernel with
`CubeCount::Static(n as u32, 1, 1)` — one cube per owner, all running
concurrently — and passes `move_other = 1`. Inside the kernel each owner cube
writes BOTH `embedding[cur_base + d1]` (its own coords) AND
`embedding[other_base + d1]` (the positive neighbour's coords) when
`move_other == 1u32`:

```rust
embedding[(cur_base + d1) as usize] = cur_d + grad_d * alpha;
if move_other == 1u32 {
    embedding[(other_base + d1) as usize] = other_d - grad_d * alpha;  // writes a DIFFERENT vertex
}
```

Because the fuzzy graph is symmetric, vertex `c` is simultaneously an owner in
its own cube AND a write target of vertex `r`'s cube. Two concurrently-scheduled
cubes therefore read-modify-write the same `embedding[c]` slots with no
synchronization. On a parallel backend (wgpu/rocm/cuda) this is an unsynchronized
data race: the result is non-deterministic between runs. That directly violates
the D-05 "byte-identical per (backend, dtype)" contract and the `reproducible_f64`
test (`umap_test.rs:884-893`) would become flaky/failing on any parallel gate.
The test passes today only because it runs on the sequential cpu-MLIR backend.

**Fix:** Make the fit path race-free. Either (a) split into a two-pass /
read-snapshot scheme where attractive updates are accumulated against a read-only
copy of the previous epoch's coordinates and applied in a separate pass (the
`sgd.rs` two-pass GATHER idiom this kernel cites), or (b) run the fit path
owner-only (`move_other = 0`) over the *symmetric* edge set so each undirected
pair is updated from both owners without any cube writing a foreign vertex — the
symmetric graph already contains both `(r,c)` and `(c,r)`, so owner-only updates
cover both endpoints with no cross-cube write. Option (b) is the smaller change
and removes the race entirely:

```rust
// host_epoch_driver fit launch — symmetric graph already has both directions,
// so owner-only updates touch every endpoint without a foreign-vertex write.
1u32 // move_other  -->  0u32
```

Whichever path is chosen, add a parallel-backend determinism test (not just the
cpu gate) so the contract is actually exercised.

### CR-02: `Umap::fit` never validates `n_components < n_samples` — spectral init underflows `usize` and panics

**File:** `crates/mlrs-algos/src/manifold/umap.rs:443-450` (fit), `:1091-1106` (spectral path), root cause `crates/mlrs-algos/src/cluster/spectral.rs:90-95`

**Issue:** `UmapBuilder::build` only rejects `n_components == 0` and
`n_neighbors == 0` (umap.rs:382-395), and `validate_geometry` only checks
`n>0 && p>0 && len`. Nothing enforces `n_components < n`. With `Init::Spectral`
(the default) and `n <= 64`, `run_umap_layout` calls
`umap_init::spectral_init` → `recover` with `drop_first = true`, so `m =
n_components + 1`. `recover` then computes:

```rust
for r in 0..m {
    let col = n - 1 - r;   // spectral.rs:91 — usize underflow when r >= n
    ...
}
```

When `n_components + 1 > n` (e.g. `n = 2, n_components = 2` → `m = 3`, `r = 2`
gives `2 - 1 - 2`), `n - 1 - r` underflows `usize` and panics (or, in release
without overflow checks, indexes wildly out of bounds → OOB read). By contrast
`SpectralEmbedding::fit` DOES guard this (`spectral_embedding.rs:155`:
`self.n_components + 1 > n_samples`), so the UMAP path is the unguarded one.

**Fix:** Validate `n_components < n` (or `n_components + 1 <= n` for the spectral
path) at the start of `fit`, before any launch, returning a typed error:

```rust
// in Umap::fit, after validate_geometry
if self.n_components >= n {
    return Err(AlgoError::InvalidNComponents {
        estimator: "umap",
        requested: self.n_components,
        max: n.saturating_sub(1),
    });
}
```

Add a test that constructs `Umap` with `n_components >= n` and asserts the typed
error instead of a panic.

### CR-03: Symmetric-graph + `move_other` double-counts every undirected edge vs umap-learn's single-pass loop

**File:** `crates/mlrs-algos/src/manifold/umap.rs:1190-1227` (per-owner CSR build), `:1293` (`move_other = 1`)

**Issue:** `fuzzy_union` emits the symmetric graph containing BOTH `(r,c)` and
`(c,r)` for each undirected edge (`umap_internals.rs:244-248`). `host_epoch_driver`
then builds one positive edge per COO entry keyed by `owner = head[e]`
(umap.rs:1201-1203), so the undirected pair `{r,c}` produces two positive edges:
one owned by `r` (target `c`) and one owned by `c` (target `r`). With
`move_other = 1`, EACH of those edges pushes both endpoints. The result is that
every undirected pair is attracted **four times** per due-epoch (r-owner moves
r&c, c-owner moves c&r), and the negative-sample schedule is likewise doubled
(each direction draws its own `epochs_per_negative`). umap-learn's
`optimize_layout_euclidean` iterates the symmetric graph's `head`/`tail` once and
relies on `move_other` to update the partner — it does NOT additionally process
the reverse edge as a separate owner step in the way this driver does, so the
effective attractive/repulsive force per pair diverges by ~2× here.

This is masked by the relative property gates (`PROPERTY_EPS = 0.02`,
`trustworthiness >= umap - eps`) and the calibration was tuned to whatever this
code produces, so the test is GREEN — but the calibration is fitting the bug, not
validating against umap's actual force schedule. Combined with the CR-01 race,
the layout dynamics are not a faithful port.

**Fix:** Decide on ONE convention and make it match umap-learn:
- If keeping `move_other = 1`, drive the layout over the DIRECTED edge set (one
  representative per undirected pair, e.g. `r < c`), not the doubled symmetric
  COO, so each pair is processed once.
- If keeping the symmetric COO, use `move_other = 0` (owner-only) so each pair is
  updated once per direction with no double endpoint move (this also fixes
  CR-01).

Re-derive the property-gate calibration AFTER the schedule is corrected, and add
an assertion comparing the per-pair sample count against the expected
`epochs_per_sample` to catch future regressions.

## Warnings

### WR-01: `make_epochs_per_sample` divides by `negative_sample_rate as f64` without guarding zero

**File:** `crates/mlrs-algos/src/manifold/umap.rs:1177-1180` and `:732-735`

**Issue:** Both `host_epoch_driver` and `transform_epoch_driver` compute
`epochs_per_negative = e / negative_sample_rate as f64`. `negative_sample_rate`
is a public builder field (`negative_sample_rate(usize)`, umap.rs:348) with no
validation. If a caller sets it to `0`, this is a division by zero producing
`inf`/`NaN` in the negative-sample clock, which then poisons `n_neg` and the
whole layout. umap-learn requires `negative_sample_rate >= 1`.

**Fix:** Validate `negative_sample_rate >= 1` in `UmapBuilder::build`, or clamp
with `.max(1)` at use. Prefer a typed build-time rejection for parity with
umap-learn.

### WR-02: `transform` reduced-epoch default seed differs from fit, undocumented divergence

**File:** `crates/mlrs-algos/src/manifold/umap.rs:663-665`

**Issue:** In `transform_new_points`, `n_epochs = cfg.n_epochs.unwrap_or(100)`
and `seed = cfg.random_state.unwrap_or(42)`. The fit path uses
`unwrap_or(if n <= 10_000 { 500 } else { 200 })` for epochs and the same `42`
default seed. When `random_state` is `None`, BOTH fit and transform fall back to
the literal `42`, so a `None`-seed fit followed by transform reuses the same seed
stream pattern across two unrelated stochastic processes. That is not wrong per
se, but the hardcoded `42` fallback in two places is a magic-number duplication
that will silently desync if one is changed. More importantly, a `None`
`random_state` is documented as "no seed" yet here it is a fixed deterministic
`42` — callers expecting run-to-run variation get identical output.

**Fix:** Hoist the default-seed (`42`) and default-epoch constants to named
`const`s shared by fit and transform, and document that `random_state = None`
maps to a fixed deterministic seed (not entropy) so the D-05 contract holds —
this is a behavioural decision that should be explicit, not buried in two
`unwrap_or(42)` calls.

### WR-03: `fit` performs a full host round-trip of `x` purely to retain training rows, doubling device traffic

**File:** `crates/mlrs-algos/src/manifold/umap.rs:456-457`

**Issue:** `let x_train_host: Vec<F> = x.to_host(pool); let x_train =
DeviceArray::from_host(pool, &x_train_host);` reads the entire input matrix back
to the host and re-uploads it solely to obtain an owned device copy for
`transform`. The caller already holds `x` on device. This is a correctness-safe
but wasteful device→host→device copy of the full design matrix on every fit. (Per
v1 scope performance is out of scope, but this is also a robustness concern: for
large `n*p` it can OOM the host with a redundant full copy.)

**Fix:** If `DeviceArray` supports a device-to-device clone / buffer retain,
acquire a device copy directly instead of round-tripping through host memory; or
document why an owned device copy cannot be taken without the host hop.

### WR-04: `fit_ab` singularity test uses `f64::MIN_POSITIVE`, far too small to catch ill-conditioning

**File:** `crates/mlrs-algos/src/manifold/umap_init.rs:120-125`

**Issue:** The LM step solves a 2×2 system and treats it as singular only when
`det.abs() < f64::MIN_POSITIVE` (~2.2e-308). A genuinely ill-conditioned but
non-subnormal `det` (e.g. 1e-280) passes the guard and yields an astronomically
large `da_step`/`db_step`, which is then rejected by the SSE check and the loop
raises `lambda` — so it usually recovers, but the guard as written never actually
fires for realistic near-singular systems and relies entirely on the SSE-reject
fallback. The comment claims it guards the singular case; effectively it does not.

**Fix:** Use a relative/scaled threshold tied to the matrix magnitude, e.g.
`det.abs() <= 1e-12 * (h00 * h11).abs().max(1.0)`, so genuine near-singularity
raises damping deterministically rather than relying on the step-reject path.

### WR-05: kernel attractive branch computes `pow_bm1 = powf(dist_sq, b - 1)` which can overflow for tiny `dist_sq`

**File:** `crates/mlrs-kernels/src/umap_layout.rs:131-137`

**Issue:** For the attractive coefficient, when `dist_sq` is small and positive
and `b - 1 < 0` (umap's typical `b ≈ 0.79..0.9`, so `b - 1 ≈ -0.1..-0.2`),
`powf(dist_sq, b - 1)` = `dist_sq^(negative)` grows without bound as `dist_sq →
0+`. umap-learn guards the gradient via the `(0.001 + dist²)` denominator in the
repulsive term and clips the per-dim delta to ±4; the attractive `grad_coeff`
here is NOT denominator-floored, only the post-multiply `grad_d` is clipped. For
`dist_sq` near the smallest positive float, `pow_bm1` can reach `inf`, making
`grad_coeff` `inf`/`NaN` BEFORE the clip (clip compares `grad_d > 4.0`, but
`NaN > 4.0` is false and `NaN < -4.0` is false, so a `NaN` passes through the
clip unchanged and corrupts the coordinate). The `dist_sq > 0` guard does not
prevent a near-zero-but-positive `dist_sq` from overflowing.

**Fix:** Floor `dist_sq` before the `powf` (e.g. compute on `max(dist_sq, eps)`
with a finite literal `eps`), matching umap's effective behaviour, or guard the
resulting `grad_coeff` for finiteness with a statement-form `if` before applying
it (NaN-safe clamp). The current clip is not NaN-safe.

## Info

### IN-01: `query_train_knn` discards `p` after computing it, with a no-op `let _ = p`

**File:** `crates/mlrs-algos/src/manifold/umap.rs:927`, `:988`

**Issue:** `let (knn_metric, p) = map_metric(metric);` then `let _ = p;` at line
988 — `p` is never used because the Minkowski exponent flows through the enum
payload instead. The binding and the suppression are dead. Minor readability
noise.

**Fix:** Destructure as `let (knn_metric, _p) = map_metric(metric);` and drop the
trailing `let _ = p;`, or change `map_metric` callers that need only the metric
to ignore the second field at the call site.

### IN-02: Duplicated symmetric-graph computation in the spectral oracle generator

**File:** `scripts/gen_oracle.py:1108` and `:1118-1119`

**Issue:** `gen_umap_spectral` computes `g = graph.maximum(graph.transpose()).tocoo()`
at line 1108 and then independently computes `sym = graph.maximum(graph.transpose())`
at line 1118 for the `spectral_layout` call. The two are the same matrix computed
twice; `g` is used only for the stored COO and `sym` only for the solver. Wasteful
and a desync risk if one is edited.

**Fix:** Compute the symmetric graph once and derive both the COO and the solver
input from it.

### IN-03: `transform_new_points` carries a long stale narrative comment describing an abandoned design

**File:** `crates/mlrs-algos/src/manifold/umap.rs:578-585`

**Issue:** The block comment ("…but the fitted shell does not keep X. Instead
transform receives X_new only…") narrates a design that was superseded — the code
immediately below DOES read `cfg.x_train_`. The comment contradicts the code and
will mislead future readers.

**Fix:** Replace with a one-line statement of the actual behaviour: "Read host
f64 copies of the new query rows and the retained training rows (`x_train_`) for
the query-vs-train KNN."

### IN-04: `umap_internals.rs` module doc still describes the file as an "EMPTY stub"

**File:** `crates/mlrs-algos/src/manifold/umap_internals.rs:3-12`

**Issue:** The module-level doc says "This module is an EMPTY stub created in Plan
14-01…" even though the file now contains the full smooth-kNN / membership /
fuzzy-union / init-graph-transform implementations. Stale scaffolding doc.

**Fix:** Update the module doc to describe the implemented stages; drop the
"EMPTY stub" framing.

---

_Reviewed: 2026-06-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
