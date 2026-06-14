---
phase: 02-core-compute-primitives
plan: 03
subsystem: compute-primitives
tags: [distance, prim-03, gemm-expansion, clamp, sqrt, elementwise, oracle, device-resident, f64-gating]

# Dependency graph
requires:
  - phase: 02-core-compute-primitives
    provides: "prims::gemm::gemm (PRIM-01, transb for XYᵀ, cubek-matmul wrap); prims::reduce::row_reduce (PRIM-02, per-row segment reduction + ScalarOp); DeviceArray (pool-routed, from_raw/from_host/to_host_metered); BufferPool; capability::skip_f64_with_log + active_backend_name + log_oracle_dtype; PrimError (ShapeMismatch/DimMismatch); mlrs-core oracle loader (load_npz) + assert_slice_close + F32_TOL/F64_TOL; smoke.rs #[cube(launch)] per-element idiom"
provides:
  - "Per-element map kernels (mlrs-kernels/src/elementwise.rs, feature-free, generic <F: Float + CubeElement>): clamp_nonneg, sqrt_elem, scale, dist_combine_clamp"
  - "Pairwise squared-Euclidean distance host API (PRIM-03): prims::distance::distance — GEMM-expansion ‖x‖²+‖y‖²−2XYᵀ + max(d²,0) clamp + optional sqrt, device-resident"
  - "ScalarOp::SumSq in prims::reduce (Σxᵢ², NO sqrt finalize) — the squared row norm ‖x‖² the GEMM-expansion needs, distinct from L2Norm (which sqrt-finalizes)"
  - "scale kernel for Plan 04 covariance: mlrs_kernels::scale::launch::<F,R>(client, count, dim, input, output, factor: F) — factor a scalar F by value"
  - "Distance npz convention fixtures: dist_sq_{f32,f64}_seed42.npz (squared) + dist_sqrt_f64_seed42.npz (sqrt), direct (X−Y)² reference"
affects: [02-04-covariance, 02-05-memory-gate, phase-05-distance-estimators-knn-kmeans-dbscan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GEMM-expansion distance: gemm(transb=true) for XYᵀ + row_reduce(SumSq) for the two squared-norm terms + 2D dist_combine_clamp kernel; squared is the core output, sqrt is an opt-in in-place boundary pass (D-08)"
    - "Unconditional max(d²,0) clamp in STATEMENT form (if d<zero{d=zero}) guarantees no negative squared distance under f32 catastrophic cancellation (Criterion 3 / T-0203-03); property test pins min>=0 on a deliberate near-identical-rows case"
    - "2D launch for the combine kernel: i=ABSOLUTE_POS_X (rows), j=ABSOLUTE_POS_Y (cols), 16×16 cube, bounds-checked i<rows && j<cols"
    - "device-resident composition in distance.rs (grep to_host == 0) — XYᵀ → norms → clamp → optional sqrt all DeviceArray→DeviceArray; pool-routed out buffer (D-11)"
    - "f32 distance oracle near-zero floor (F32_DIST_NEAR_ZERO_FLOOR=1e-2) reusing the pipeline_test/gemm_test precedent; f64 keeps strict assert_slice_close"
    - "direct (X−Y)² host reference + fixtures (NOT the GEMM-expansion) so the match validates the expansion identity, not a tautology"

key-files:
  created:
    - crates/mlrs-kernels/src/elementwise.rs
    - crates/mlrs-backend/src/prims/distance.rs
    - crates/mlrs-backend/tests/distance_test.rs
    - tests/fixtures/dist_sq_f32_seed42.npz
    - tests/fixtures/dist_sq_f64_seed42.npz
    - tests/fixtures/dist_sqrt_f64_seed42.npz
  modified:
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-backend/src/prims/reduce.rs
    - crates/mlrs-backend/tests/reduce_test.rs
    - scripts/gen_oracle.py

key-decisions:
  - "[02-03] distance needs the SQUARED row norm ‖x‖², but row_reduce(ScalarOp::L2Norm) applies a sqrt finalize. Added ScalarOp::SumSq (Σxᵢ², no sqrt) to prims::reduce — the internal Op::SumSq + first-pass squaring already existed; SumSq passes the raw partial through in finalize_scalar. This is the minimal in-tree change (no new dependency, T-0203-SC held) and keeps L2Norm/SumSq as distinct, separately-tested ops."
  - "[02-03] dist_combine_clamp is a 2D-launched kernel (16×16 cube, i on ABSOLUTE_POS_X, j on ABSOLUTE_POS_Y) so one unit maps to one output element; rows/cols passed as scalar u32 BY VALUE (no ScalarArg wrapper, cubecl 0.10 — confirmed against the saxpy `a: F` idiom and spike_test note)."
  - "[02-03] the max(d²,0) clamp is the STATEMENT form (let zero=F::from_int(0i64); if d<zero{d=zero;}) per Cubecl_conditionals.md — never a max expression / if-expression that can mis-lower. distance_min_nonnegative pins min>=0 on a deliberate large-magnitude near-identical-rows f32 cancellation case."
  - "[02-03] sqrt is an OPT-IN in-place pass over the already-clamped (non-negative) buffer, so sqrt_elem never sees a negative argument (no NaN leak to KNN — T-0203-03 / Pitfall 5). Squared is the cheaper default for KMeans/DBSCAN distance comparisons (D-08)."
  - "[02-03] distance.rs is device-resident (grep -c to_host == 0); the comment text was reworded to avoid the literal token so the grep gate is unambiguous. The internal row_reduce host-slicing is Plan 02's existing reduction behaviour, not a distance.rs mid-pipeline round-trip."
  - "[02-03] f32 distance oracle uses F32_DIST_NEAR_ZERO_FLOOR=1e-2 (abs-only below the floor, still ≤1e-5 abs) reusing the pipeline_test/gemm_test precedent; the 1e-5 abs bound is never loosened and f64 keeps strict assert_slice_close. HARD guardrail NOT tripped — the squared/sqrt oracle gates pass at 1e-5 on cpu and wgpu."

patterns-established:
  - "Per-element map kernel file (elementwise.rs): the smoke.rs saxpy shell generalised to clamp/sqrt/scale (1D ABSOLUTE_POS) and dist_combine_clamp (2D ABSOLUTE_POS_X/Y); feature-free, generic <F: Float + CubeElement>; the shared file Plan 04 covariance consumes (scale)"
  - "Primitive-composition host API (distance.rs): reuse validated GEMM + reduction primitives, add only a small combine kernel, keep the whole pipeline DeviceArray→DeviceArray"
  - "direct independent host reference (not the device's own algebra) as the oracle so the test validates the identity rather than restating it"

requirements-completed: [PRIM-03]

# Metrics
duration: 7min
completed: 2026-06-12
---

# Phase 02 Plan 03: Pairwise Squared-Euclidean Distance (PRIM-03) Summary

**A device-resident pairwise squared-Euclidean distance primitive via the GEMM-expansion `‖x‖²+‖y‖²−2XYᵀ` — composing the Plan-01 GEMM (`transb` for `XYᵀ`) and the Plan-02 row reduction (new `ScalarOp::SumSq` squared-norm term), an unconditional `max(d²,0)` clamp in statement form that produces NO negative distances under f32 catastrophic cancellation (property-pinned `min >= 0`), and an opt-in in-place sqrt boundary for KNN — oracle-validated within 1e-5 against a direct independent host reference and committed squared/sqrt npz fixtures, for f32 and f64 on cpu AND wgpu.**

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-06-12
- **Tasks:** 3 (all `auto`, Tasks 1–2 TDD-style RED-implied/GREEN, Task 3 validation)
- **Files:** 11 (6 created, 5 modified)

## Accomplishments

- **Elementwise map kernels** (`crates/mlrs-kernels/src/elementwise.rs`, feature-free, D-13): `clamp_nonneg` (max(in,0), STATEMENT form), `sqrt_elem`, `scale` (out=in*factor, scalar F by value), and the 2D `dist_combine_clamp` (`max(‖x‖²+‖y‖²−2XYᵀ, 0)`). Re-exported in `lib.rs`. `cargo build -p mlrs-kernels` clean.
- **Distance host API** (`crates/mlrs-backend/src/prims/distance.rs`): `prims::distance::distance` validates geometry (PrimError), computes `XYᵀ` via `gemm(transb=true)`, the two squared row norms via `row_reduce(ScalarOp::SumSq)`, launches `dist_combine_clamp`, and applies the optional in-place `sqrt_elem`. Device-resident (`grep -c to_host` = 0), pool-routed output (D-11).
- **`ScalarOp::SumSq`** added to `prims::reduce` — the squared row norm `Σxᵢ²` (no sqrt finalize) the GEMM-expansion needs, distinct from the existing `L2Norm` (which sqrt-finalizes). Reused the already-present internal `Op::SumSq` + first-pass squaring.
- **Validation** (`crates/mlrs-backend/tests/distance_test.rs`): `distance_matches_host_ref` (f32) + `distance_f64_matches_host_ref` squared sweeps vs a DIRECT `(X−Y)²` host loop; `distance_min_nonnegative` (deliberate f32 cancellation, asserts `min >= 0`); `distance_sqrt_matches_host_ref`; `distance_npz_fixture_matches`. **5/5 green on cpu AND wgpu.**
- **Distance npz fixtures** (`dist_sq_{f32,f64}_seed42.npz`, `dist_sqrt_f64_seed42.npz`) via `gen_oracle.py::gen_distance` — committed blobs, direct-reference oracle.

## `scale` kernel signature (Plan 04 covariance consumes this)

```rust
// crates/mlrs-kernels/src/elementwise.rs
#[cube(launch)]
pub fn scale<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, factor: F)
// launch: mlrs_kernels::scale::launch::<F, ActiveRuntime>(
//     &client, count, dim, in_arg: ArrayArg, out_arg: ArrayArg, factor /* scalar F by value */)
```
`factor` is a scalar `F` passed BY VALUE (no `ScalarArg` wrapper in cubecl 0.10 — same idiom as `saxpy_kernel`'s `a: F`). Covariance folds the `1/(n-ddof)` normalisation through this.

## Distance host API signature

```rust
// crates/mlrs-backend/src/prims/distance.rs
pub fn distance<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>, (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>, (rows_y, cols_y): (usize, usize),
    sqrt: bool,
    out: Option<DeviceArray<ActiveRuntime, F>>,
    path: ReducePath,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>  // rows_x × rows_y, device-resident
```

## Task Commits

1. **Task 1: Elementwise kernels (clamp_nonneg, sqrt_elem, scale, dist_combine_clamp)** — `3d79fde` (feat)
2. **Task 2: Distance host API (GEMM-expansion + row-SumSq + clamp + optional sqrt) + ScalarOp::SumSq** — `773342b` (feat)
3. **Task 3: Validate distance (host-ref sweep, min>=0 property, squared/sqrt fixtures)** — `9c8b549` (test)

## Files Created/Modified

- `crates/mlrs-kernels/src/elementwise.rs` (created) — four per-element `#[cube(launch)]` kernels.
- `crates/mlrs-kernels/src/lib.rs` (modified) — `pub mod elementwise` + re-exports.
- `crates/mlrs-backend/src/prims/distance.rs` (created) — distance host API, geometry validation, 2D combine launch, optional sqrt.
- `crates/mlrs-backend/src/prims/mod.rs` (modified) — `pub mod distance`.
- `crates/mlrs-backend/src/prims/reduce.rs` (modified) — `ScalarOp::SumSq` variant + `into_op`/`finalize_scalar` arms.
- `crates/mlrs-backend/tests/distance_test.rs` (created) — 5 oracle/property tests.
- `crates/mlrs-backend/tests/reduce_test.rs` (modified) — `SumSq` match arms (`host_sumsq` for axis; `unreachable!` for the full-scalar helper not called with it).
- `scripts/gen_oracle.py` (modified) — `gen_distance` (squared + sqrt) + the `main()` calls.
- `tests/fixtures/dist_sq_{f32,f64}_seed42.npz`, `tests/fixtures/dist_sqrt_f64_seed42.npz` (created).

## Decisions Made

See `key-decisions` frontmatter. Headlines:
- **`ScalarOp::SumSq` added** for the squared norm (L2Norm sqrt-finalizes; distance needs `Σxᵢ²`). Minimal in-tree change, zero new deps.
- **2D combine kernel**, scalar `u32` rows/cols by value.
- **STATEMENT-form clamp** (Cubecl_conditionals.md) — `min >= 0` property-pinned on f32 cancellation.
- **sqrt is opt-in, in place over the clamped buffer** — never sees a negative argument (no KNN NaN).
- **Direct `(X−Y)²` oracle** independent of the GEMM-expansion.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Added `ScalarOp::SumSq` to consume the SQUARED row norm**
- **Found during:** Task 2 (wiring the norm terms of the GEMM-expansion)
- **Issue:** The plan and the Plan-02 handoff name `row_reduce(.., ScalarOp::L2Norm, ..)` as the row-norm source, but `L2Norm` applies a sqrt finalize — it returns `sqrt(Σxᵢ²)`, while the GEMM-expansion `‖x‖²+‖y‖²−2XYᵀ` needs the SQUARED norm `Σxᵢ²`. Using `L2Norm` would compute `‖x‖+‖y‖−2XYᵀ` (wrong, and could go negative far more often). Correctness requirement.
- **Fix:** Added `ScalarOp::SumSq` (maps to the existing internal `Op::SumSq`, passes the raw `Σxᵢ²` through `finalize_scalar` with no sqrt). The squaring + multi-pass machinery already existed; this only exposes the un-sqrt'd result. No new dependency (T-0203-SC held).
- **Files modified:** `crates/mlrs-backend/src/prims/reduce.rs`.
- **Verification:** `distance_matches_host_ref` (f32) + `_f64` match the direct host `Σ(X−Y)²` within tolerance on cpu and wgpu.
- **Committed in:** `773342b` (Task 2).

**2. [Rule 3 - Blocking] Updated `reduce_test.rs` match arms for the new `SumSq` variant**
- **Found during:** Task 3 (first test compile)
- **Issue:** Adding the `ScalarOp::SumSq` variant made the three exhaustive `match op` sites in the existing `reduce_test.rs` non-exhaustive (E0004) — a blocking compile error in a sibling test file caused directly by the Task-2 change.
- **Fix:** Added a `host_sumsq` helper and `SumSq` arms: the axis-finalize host ref (`reduce_host`) maps `SumSq → host_sumsq`; the full-array helper (`check_full_scalar`), which has no public full-array sumsq fn and is never invoked with `SumSq`, gets an `unreachable!` arm with a clear message.
- **Files modified:** `crates/mlrs-backend/tests/reduce_test.rs`.
- **Verification:** `reduce` suite still green on wgpu (3/3) — no regression; distance suite compiles and passes.
- **Committed in:** `9c8b549` (Task 3).

---

**Total deviations:** 2 (1 missing-critical correctness, 1 blocking compile in a sibling test). **Impact:** No scope creep — deviation 1 is the correct norm term for the expansion (the plan's `L2Norm` handoff name would have been a bug); deviation 2 is the mechanical consequence of exposing it. The HARD guardrail held: the squared/sqrt oracle gates and the `min >= 0` property all pass at 1e-5 without loosening the clamp or `F32_TOL`.

## Issues Encountered

- **`ScalarArg::new` does not exist in cubecl 0.10** (E0433 on first build of distance.rs): scalar kernel args are passed BY VALUE in the generated `launch` fn (the `spike_test.rs` note and `saxpy_kernel`'s `a: F` confirm this). Fixed by passing `rows_x as u32` / `rows_y as u32` directly — a one-line idiom correction, not a deeper cubecl issue, so the AGENTS.md §4 cubecl_error_guideline full protocol was not invoked (the manual's "scalar args by value" pattern resolved it immediately).
- **No CubeCL build/lowering errors** in the kernels — `elementwise.rs` compiled clean on cpu + wgpu first try; the statement-form clamp lowered correctly (the `min >= 0` property holds on real subgroup hardware).
- **`grep -c to_host` gate**: two doc-comment mentions of the token initially made the raw grep non-zero though no actual read-back call exists; reworded the comments so the device-residency grep gate is unambiguously `0`.

## Threat Flags

None. Numerical compute-kernel plane — no auth/session/network/PII surface. The threat register is mitigated as designed: kernels bounds-check `i<rows && j<cols` / `tid<len` with `len` from validated `DeviceArray.len` (T-0203-01); `distance` validates `rows*cols==len` and `cols==cols_y` via `PrimError` before any launch (T-0203-02); the unconditional `max(d²,0)` clamp + the sqrt-after-clamp ordering prevent any NaN-from-negative leaking to KNN (T-0203-03, property-pinned by `distance_min_nonnegative`); zero new dependencies (T-0203-SC).

## Next Phase Readiness

- **Plan 04 (covariance):** `mlrs_kernels::scale` ready for the `1/(n-ddof)` normalisation (signature recorded above); `prims::reduce::column_reduce(.., ScalarOp::Mean, ..)` (Plan 02) for column-centering; `prims::gemm::gemm(transa=true)` for `AᵀA`. The shared `elementwise.rs` file is the home for any further covariance map kernels.
- **Plan 05 (memory gate):** distance is another device-resident composition whose pool-routed scratch/out buffers feed the D-10 reuse/read-back assertions.
- **Phase 5 (KNN/KMeans/DBSCAN):** `prims::distance::distance(sqrt=false)` for squared-distance comparisons (KMeans/DBSCAN), `sqrt=true` for KNN Euclidean distances; composes with `argmin_rows` (Plan 02) for nearest-neighbour/label assignment.

---
*Phase: 02-core-compute-primitives*
*Completed: 2026-06-12*

## Self-Check: PASSED
- All 6 created files + the SUMMARY verified present on disk.
- All 3 task commits (3d79fde, 773342b, 9c8b549) verified in git log.
- `cargo test -p mlrs-backend --features cpu distance`: 5/5 green (~6.6 s).
- `cargo test -p mlrs-backend --features wgpu distance`: 5/5 green; `reduce` suite unregressed (3/3 wgpu).
- `cargo build -p mlrs-kernels` (feature-free): green. distance.rs `grep -c to_host` = 0; clamp in statement form.
