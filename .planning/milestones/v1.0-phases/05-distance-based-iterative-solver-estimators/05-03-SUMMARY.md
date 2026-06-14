---
phase: 05-distance-based-iterative-solver-estimators
plan: 03
subsystem: kernels
tags: [kmeans, lloyd, inertia, kmeans-plus-plus, d2-sampling, host-seeded-rng, splitmix64, cubecl, cpu-mlir, gather-no-scatter, primitive-first]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "kmeans.rs/prims/kmeans.rs/{kmeanspp,lloyd}_test.rs stubs + lib.rs/prims/mod.rs registrations; kmeans_{f32,f64}_seed42.npz fixtures (injected init, D-09)"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 02
    provides: "the cubecl-cpu SharedMemory-free #[cube] kernel pattern (F/u32 accumulators + if-guards; no SharedMemory/mutable-bool/F::INFINITY/descending-shift) proven to LAUNCH on the cpu(f64) MLIR gate"
  - phase: 02-foundational-primitives
    provides: "prims::distance (GEMM-expansion squared-Euclidean), prims::reduce::argmin_rows (nearest-centroid assignment, REUSED not rebuilt)"
provides:
  - "mlrs_kernels::kmeans::{centroid_sumcount, inertia_rows} — feature-free #[cube] GATHER kernels (per-(c,j) sum+count; per-row squared dist to assigned center), cubecl-cpu MLIR safe"
  - "mlrs_backend::prims::kmeans::{lloyd_update, inertia, kmeanspp_sample} — validate-before-launch wrappers; empty-cluster relocation; host SplitMix64 PRNG over device D² weights (init-only)"
  - "lloyd_test.rs (centers + inertia within 1e-5 vs sklearn on cpu(f64) + empty-cluster relocation) + kmeanspp_test.rs (distinct/in-range + seed-reproducible invariant) GREEN"
affects: [05-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-output GATHER #[cube] (one unit reads many inputs) instead of scatter+atomic — cubecl-cpu lowers it directly; centroid_sumcount gathers Σ X[i,j] over label==c rows, inertia_rows gathers Σ(X[i,j]−center[label_i,j])² per row"
    - "Host-side documented seeded PRNG (SplitMix64, Steele-Lea-Flood 2014) for k-means++ — no rand crate, never OsRng (ASVS V6); D² weights device-computed via the distance prim + read back ONCE PER CENTER at init only (D-09c), never on-device RNG (backend-divergent)"
    - "Lloyd math granularity (RESEARCH Open Q1): device handles the n-heavy per-output gather (sums/counts/per-row d²); host handles the small-k finalize (divide→mean, Σ→inertia, empty-cluster relocation)"

key-files:
  created: []
  modified:
    - "crates/mlrs-kernels/src/kmeans.rs (filled the 05-01 stub: centroid_sumcount + inertia_rows #[cube] GATHER kernels)"
    - "crates/mlrs-backend/src/prims/kmeans.rs (filled the 05-01 stub: lloyd_update + inertia + kmeanspp_sample + SplitMix64; validate-before-launch)"
    - "crates/mlrs-backend/tests/lloyd_test.rs (de-#[ignore]d: sklearn centers+inertia 1e-5 oracle + empty-cluster relocation case)"
    - "crates/mlrs-backend/tests/kmeanspp_test.rs (de-#[ignore]d: distinct/in-range + seed-reproducibility invariant)"

key-decisions:
  - "centroid_sumcount written as a per-(c,j) GATHER (each output unit scans all n rows) rather than a per-row scatter-with-atomic: cubecl-cpu does not reliably lower cross-unit atomics, and the gather has no scatter race — the host then divides sum/count to get the mean"
  - "empty-cluster relocation is HOST-side (small-k): an empty cluster is moved to the not-yet-claimed sample with the MAX squared distance to the nearest NON-EMPTY new center (the worst-served point) — sklearn _k_means_common relocation, never a divide-by-zero NaN (T-05-03-02). The previous-iteration center is unavailable to this stateless prim, so 'farthest from nearest new center' is the standalone-prim analog"
  - "k-means++ RNG is a hand-rolled host SplitMix64 (no rand crate is in the workspace) — fully deterministic, documented, seed-reproducible; the D² weights are device-computed (distance prim, sqrt=false) and read back once per center (D-09c, init only)"
  - "bad k / geometry reported as PrimError::ShapeMismatch (operand \"k\"/\"x\"/\"labels\"/\"centers\") — PrimError has no InvalidK variant, matching the distance.rs/topk.rs convention"

patterns-established:
  - "cubecl-cpu-safe Lloyd idiom: per-output gather kernels (no scatter, no atomic, no SharedMemory) for the n-heavy sums/inertia; host finalize for the small-k mean/sum/relocation"
  - "host-seeded SplitMix64 weighted-draw idiom for any init-time D²/probability sampling — device computes the weights, host owns the RNG (reproducible, backend-agnostic)"

requirements-completed: [CLUSTER-01]

# Metrics
duration: 14min
completed: 2026-06-13
---

# Phase 5 Plan 03: KMeans Lloyd + k-means++ Primitives (D-01) Summary

**The genuinely-new KMeans device primitives: two feature-free `#[cube]` GATHER kernels (`centroid_sumcount`, `inertia_rows`) plus their `lloyd_update` / `inertia` / `kmeanspp_sample` validate-before-launch wrappers — Lloyd centroid means (with sklearn empty-cluster relocation) and the Σ-squared-distance inertia GREEN within 1e-5 vs sklearn on cpu(f64), and the host-seeded-PRNG k-means++ D²-sampler proven valid + seed-reproducible — all standalone before the KMeans estimator (07) consumes them (D-01 primitive-first).**

## Performance

- **Duration:** ~14 min
- **Tasks:** 2 (Task 1 TDD)
- **Files modified:** 4 (kernel + prim + 2 tests — all 05-01 stubs filled; zero shared-file edits)

## Accomplishments
- Filled `mlrs_kernels::kmeans` with two SharedMemory-free `#[cube]` GATHER kernels: `centroid_sumcount` (one unit per `(centroid c, feature j)` output scans all `n` rows accumulating `Σ X[i,j]` over `label==c` rows + the per-centroid count) and `inertia_rows` (one unit per row gathers `Σ_j (X[i,j] − centers[label_i,j])²`). Both use only `F`/`u32` accumulators + ascending `while` scans + `if` guards — the cubecl-cpu MLIR-safe pattern from plan 05-02 — and a per-output GATHER layout (no scatter, no atomic) that cpu lowers directly.
- Filled `mlrs_backend::prims::kmeans` with `lloyd_update` (device gather → host divide-by-count mean + sklearn empty-cluster relocation to the worst-served sample), `inertia` (device gather → host f64-accumulate sum, squared/no-sqrt per D-08), and `kmeanspp_sample` (host SplitMix64 PRNG drawing each next center ∝ device-computed D² weights, read back once per center at init only — D-09c). Every entry point validates `n*d==x.len()`, `1<=k<=n`, and label-in-`0..k` → `PrimError::ShapeMismatch` BEFORE any unsafe launch (T-05-03-01 / ASVS V5). Assignment reuses the Phase-2 distance prim (not rebuilt).
- De-`#[ignore]`d `lloyd_test.rs`: from `kmeans_{f32,f64}_seed42.npz` it runs `lloyd_update` on the fixture labels and asserts the centers equal sklearn `cluster_centers_` within 1e-5, runs `inertia` and asserts it equals `inertia_` within 1e-5 (f64 cpu-gated), plus a constructed empty-cluster case asserting cluster 2 is relocated to a finite real data row (no divide-by-zero NaN).
- De-`#[ignore]`d `kmeanspp_test.rs` as an INVARIANT test (no bit-for-bit sklearn ref, D-09): k sampled indices distinct + in `0..n`, same-seed reproducible, and at least one alternate seed differs (proving the host RNG drives the draw).
- Verified the full gate: `cargo build -p mlrs-kernels` green; `cargo test --features cpu -p mlrs-backend --test lloyd_test --test kmeanspp_test` 7/7 green (incl. both f64 cases + relocation + reproducibility); `cargo build -p mlrs-backend --features rocm --tests` green.

## Task Commits

1. **Task 1: fill kmeans `#[cube]` kernels + lloyd_update/inertia wrappers + lloyd_test oracle** — `d3fc967` (feat)
2. **Task 2: de-ignore the k-means++ D²-sampling invariant test** — `2d8743c` (feat)

> Note: `kmeanspp_sample` (the Task-2 prim function) lives in the same `prims/kmeans.rs` file as Task 1's wrappers, so it landed in the Task-1 commit; Task 2's commit is the `kmeanspp_test.rs` de-ignore that exercises it. This keeps each commit atomic on a single file boundary.

## Files Created/Modified
- `crates/mlrs-kernels/src/kmeans.rs` — `centroid_sumcount` + `inertia_rows` `#[cube]` GATHER kernels; `pub use … as kmeans_centroid_sumcount`/`kmeans_inertia_rows` inside the file. (lib.rs untouched.)
- `crates/mlrs-backend/src/prims/kmeans.rs` — `lloyd_update` / `inertia` / `kmeanspp_sample` + `device_d2_to_center` helper + host `SplitMix64` PRNG + `validate_geometry`. (prims/mod.rs untouched.)
- `crates/mlrs-backend/tests/lloyd_test.rs` — `check_lloyd` oracle body + 4 tests (fixture_loads, sklearn f32, inertia f64, empty-cluster relocation).
- `crates/mlrs-backend/tests/kmeanspp_test.rs` — `sample`/`assert_valid` helpers + 3 tests (fixture_loads, valid+reproducible f32, reproducible f64).

## Decisions Made
- **GATHER, not scatter+atomic:** `centroid_sumcount` was written as a per-`(c,j)`-output gather (each output unit scans all `n` rows) rather than a per-row scatter accumulating into shared per-centroid sums. The gather has no cross-unit write race, so it needs no atomics — which cubecl-cpu does not lower reliably — and it launched cleanly on cpu(f64) first try.
- **Empty-cluster relocation is the standalone-prim analog of sklearn's:** sklearn relocates an empty cluster to the point farthest from ITS CURRENT (previous-iteration) center. This stateless prim has no previous center, so it relocates to the not-yet-claimed sample with the max squared distance to its nearest NON-EMPTY new center (the worst-served point) — finite, distinct, and never a divide-by-zero NaN (T-05-03-02).
- **Hand-rolled host SplitMix64 (no `rand` crate):** the workspace has no `rand` dependency and adding one is out of scope (RESEARCH audit: zero new packages). SplitMix64 is a small, well-documented, fully-deterministic seeded PRNG — never `OsRng` (ASVS V6 / T-05-03-03) — so the sampler is seed-reproducible across runs and backends.

## Deviations from Plan

None — plan executed exactly as written. (No deviation rules triggered: the kernel granularity was Claude's-discretion per RESEARCH Open Q1; the GATHER-vs-scatter and SplitMix64 choices are implementation decisions within the plan's stated latitude, documented above. Both feature builds + all 7 tests pass.)

## Known Stubs

None. Both stub files were fully implemented and the oracle/invariant tests exercise real device output (no hardcoded/empty values flow to the assertions).

## Issues Encountered

None. The cubecl-cpu MLIR gate accepted both gather kernels on the first launch — the plan 05-02 SharedMemory-free pattern (F/u32 accumulators + if-guards, no atomics) held, so the "compiles but panics at launch" failure mode never materialised.

## Next Phase Readiness
- **Plan 05-07 (KMeans estimator) unblocked:** `lloyd_update` returns device-resident `k × d` centers (empty-cluster-safe), `inertia` returns the scalar `Σ d²`, and `kmeanspp_sample` returns the `k` default-init indices — all validate-before-launch and seed-reproducible. Assignment is the Phase-2 `distance` + `argmin_rows` (already device-resident). The standalone oracle is green within 1e-5, satisfying the D-01 primitive-first gate.
- No blockers. cpu(f64) full + rocm(f32) test-target build both green; lib.rs/prims/mod.rs untouched so the sibling Wave-2 prim plans stay file-disjoint.

## Threat Flags

None — no new network/auth/file surface. The trust boundaries are exactly the threat register's: `lloyd_update`/`inertia`/`kmeanspp_sample` validate `1<=k<=n`/`n*d==x.len()`/label-in-`0..k` → `PrimError` before any unsafe launch (T-05-03-01), empty clusters are relocated before the mean (T-05-03-02, no NaN), and the RNG is a documented host-seeded SplitMix64 with a pinned same-seed reproducibility invariant (T-05-03-03). Zero new dependencies (T-05-03-SC).

## Self-Check: PASSED

- All modified files verified present (kmeans.rs kernel, prims/kmeans.rs wrapper, lloyd_test.rs, kmeanspp_test.rs, this SUMMARY).
- Both task commits verified in git history (`d3fc967`, `2d8743c`).
- `cargo test --features cpu -p mlrs-backend --test lloyd_test --test kmeanspp_test` 7/7 green (incl. both f64 + relocation + reproducibility); `cargo build -p mlrs-kernels` + `-p mlrs-backend --features rocm --tests` green; lib.rs/prims/mod.rs untouched.

---
*Phase: 05-distance-based-iterative-solver-estimators*
*Completed: 2026-06-13*
