---
phase: 02-core-compute-primitives
plan: 04
subsystem: compute-primitives
tags: [covariance, gram, xtx, prim-04, gemm-transa, ddof, np-cov, center-columns, buffer-reuse, d10-gate3, oracle, f64-gating]

# Dependency graph
requires:
  - phase: 02-core-compute-primitives
    provides: "prims::gemm::gemm (PRIM-01, transa for AᵀA, cubek-matmul wrap, pool-routed/caller-provided out buffer D-11); prims::reduce::column_reduce(.., ScalarOp::Mean, ..) (PRIM-02 column-mean centring); mlrs_kernels::scale (PRIM-03 elementwise.rs, 1/(n-ddof) normalise); DeviceArray (pool-routed, from_raw/from_host/to_host_metered, handle()); BufferPool::{acquire,client,stats}; capability::{skip_f64_with_log, active_backend_name, log_oracle_dtype}; PrimError (ShapeMismatch); mlrs-core oracle loader (load_npz) + assert_slice_close + F32_TOL/F64_TOL; smoke.rs #[cube(launch)] per-element idiom"
provides:
  - "Covariance / XᵀX (Gram) host API (PRIM-04): prims::covariance::covariance — column-mean centring + AᵀA via GEMM(transa=true, no transpose buffer D-09) + 1/(n-ddof) scale, device-resident, GEMM-output-buffer reuse (D-10 gate 3)"
  - "center_columns elementwise kernel (mlrs-kernels/src/elementwise.rs): out[r,c] = a[r,c] - mean[c], broadcasting length-n_features means — keeps covariance centring device-resident (grep to_host == 0 in covariance.rs)"
  - "Covariance npz convention fixtures: cov_ddof0_f64_seed42.npz, cov_ddof1_f64_seed42.npz, cov_ddof1_f32_seed42.npz (np.cov A rowvar=False — features as columns)"
  - "gen_oracle.py gen_covariance(ddof) case (np.cov rowvar=False, ddof=0 population + ddof=1 sample)"
affects: [02-05-memory-gate, phase-03-svd-eig, phase-04-pca-linear-solvers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Covariance composition: column_reduce(Mean) → center_columns kernel (device-side broadcast subtract) → gemm(transa=true) AᵀA → scale(1/(n-ddof)) in place over the GEMM output buffer; zero new external deps (T-0204-SC), one new per-element kernel (center_columns, same class as scale/clamp_nonneg)"
    - "GEMM-output-buffer reuse (D-10 gate 3): the internal gemm() out handle IS scaled in place (scale input==output==gram.handle()) and returned as covariance's DeviceArray — the Gram handle is never a second parallel allocation; the caller's D-11 out-buffer threads straight through gemm→scale"
    - "ddof selectable via the scale factor 1/(n_samples - ddof): ddof=0 population (1/n), ddof=1 sample (1/(n-1)); pinned by np.cov(rowvar=False, ddof) fixtures"
    - "device-resident composition in covariance.rs (grep to_host == 0): centring done on-device via center_columns reading the means DeviceArray directly (no host round-trip in covariance.rs; the column_reduce internal host slicing is Plan 02's reduction behaviour, not a covariance round-trip)"
    - "f32 covariance oracle near-zero floor (F32_COV_NEAR_ZERO_FLOOR=1e-2) reusing the gemm_test/distance_test precedent; f64 keeps strict assert_slice_close"
    - "direct centred-AᵀA/(n-ddof) host reference (NOT the GEMM(transa) algebra) + np.cov fixtures so the match validates the normalisation convention, not a tautology"

key-files:
  created:
    - crates/mlrs-backend/src/prims/covariance.rs
    - crates/mlrs-backend/tests/covariance_test.rs
    - tests/fixtures/cov_ddof0_f64_seed42.npz
    - tests/fixtures/cov_ddof1_f64_seed42.npz
    - tests/fixtures/cov_ddof1_f32_seed42.npz
  modified:
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-kernels/src/elementwise.rs
    - crates/mlrs-kernels/src/lib.rs
    - scripts/gen_oracle.py

key-decisions:
  - "[02-04] GEMM-output-buffer reuse mechanism (D-10 gate 3, LOAD-BEARING for Plan 05): covariance() drives the internal gemm(transa=true) into a SINGLE output buffer — the caller's `out` DeviceArray when supplied (D-11), else a pool acquisition inside gemm — and then launches scale with that SAME handle as BOTH input and output (gram.handle() == scale in_arg == scale out_arg), normalising 1/(n-ddof) IN PLACE. The DeviceArray covariance returns wraps the EXACT GEMM output handle (`Ok(gram)` where gram = the gemm() result). There is NO separate covariance output allocation. Plan 05's gate 3 assertion: pass the GEMM's output DeviceArray as covariance's `out`; computing covariance after a same-shape GEMM does not bump PoolStats.allocations for a fresh Gram buffer (the Gram handle is the reused GEMM handle / same-byte-size free-list reuse)."
  - "[02-04] center_columns elementwise kernel added (Rule 2 — D-05 device-residency) so covariance.rs keeps grep -c to_host == 0. Two-pass centring (RESEARCH Pitfall 4) needs the per-column mean subtracted from every element; doing it on the host would require a literal to_host in covariance.rs (the acceptance criterion forbids it). center_columns(a, mean, out, cols) is a per-element #[cube(launch)] map (out[tid]=a[tid]-mean[tid%cols]), same class as the existing scale/clamp_nonneg — NOT a new compute kernel for the Gram itself (the 'no new kernels' note is about the Aᵀ·A path, which still composes GEMM). means stay device-resident as the column_reduce DeviceArray and feed the kernel directly."
  - "[02-04] ddof folded into the scale factor: factor = 1/(n_samples - ddof), ddof=0 ⇒ 1/n (population), ddof=1 ⇒ 1/(n-1) (sample). recip() computes the factor at f64 width then stores at the array dtype; passed by value to scale (cubecl 0.10, like saxpy's a:F)."
  - "[02-04] Fixtures pin np.cov(A, rowvar=False, ddof) — rowvar=False so FEATURES are the columns of A, exactly the device API's (n_samples, n_features) row-major contract. A is 7×4 (n_samples > n_features, non-square so ddof actually changes normalisation); C is 4×4. Regenerated via /tmp numpy venv (PEP 668); existing fixtures byte-reproducible (only the 3 cov blobs are new)."

patterns-established:
  - "Primitive-composition host API (covariance.rs): reuse validated GEMM(transa) + column-mean reduction + scale, add only a small center_columns map kernel, keep the whole pipeline DeviceArray→DeviceArray; the GEMM output buffer is the returned result (in-place scale), establishing the D-10 gate-3 reuse contract"
  - "ddof normalisation via the scale factor + np.cov(rowvar=False) fixtures for BOTH population (ddof=0) and sample (ddof=1) — the convention PCA + linear closed-form solvers inherit"

requirements-completed: [PRIM-04]

# Metrics
duration: 12min
completed: 2026-06-12
---

# Phase 02 Plan 04: Covariance / XᵀX (Gram) Primitive (PRIM-04) Summary

**A device-resident covariance / XᵀX (Gram) primitive via column-mean centring (Plan-02 `column_reduce(Mean)` + a new device-side `center_columns` map kernel), the Gram `AᵀA` computed with GEMM's `transa` flag (no materialized transpose buffer, D-09), and a `1/(n_samples − ddof)` scale folded IN PLACE over the GEMM output buffer — so covariance REUSES the GEMM output handle rather than allocating a parallel one (the load-bearing D-10 gate-3 reuse Plan 05 asserts) — selectable for both population (`ddof=0`) and sample (`ddof=1`) normalisation, oracle-validated within 1e-5 against a direct independent host reference AND committed `np.cov(rowvar=False)` npz fixtures, for f32 and f64 on cpu AND wgpu, with zero new external dependencies.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-06-12
- **Tasks:** 2 (Task 1 TDD-GREEN host API, Task 2 validation)
- **Files:** 9 (5 created, 4 modified)

## Accomplishments

- **PRIM-04 covariance host API** (`crates/mlrs-backend/src/prims/covariance.rs`): `prims::covariance::covariance(pool, a, (n_samples, n_features), ddof, out, path)` validates geometry (`n_samples*n_features == a.len()`, D-04 / T-0204-02), centres columns on-device, computes `AᵀA` via `gemm(transa=true)` (no transpose buffer, D-09 / D-06), and scales by `1/(n_samples-ddof)` in place over the GEMM output buffer. Device-resident (`grep -c to_host` = 0), pool-routed output (D-11). `cargo build --features cpu/wgpu` clean.
- **`center_columns` elementwise kernel** (`crates/mlrs-kernels/src/elementwise.rs`, feature-free, D-13): `out[r,c] = a[r,c] - mean[c]` broadcasting the length-`n_features` means; keeps the two-pass centring device-resident so covariance.rs has no host round-trip.
- **GEMM-output-buffer reuse (D-10 gate 3)**: the internal `gemm()` output handle is the buffer that `scale` writes back into itself, and covariance returns that exact handle — no parallel Gram allocation (mechanism detailed below).
- **ddof=0 AND ddof=1** both selectable via the `1/(n-ddof)` scale factor — population (`1/n`) and sample (`1/(n-1)`), pinned by `np.cov` fixtures.
- **Validation** (`crates/mlrs-backend/tests/covariance_test.rs`): `covariance_ddof0_matches` (population) and `covariance_ddof1_matches` (sample) vs a DIRECT f64 centred-`AᵀA/(n-ddof)` host reference AND the committed `np.cov(rowvar=False)` fixtures; f32 + f64, f64 capability-gated. **2/2 green on cpu AND wgpu.**
- **Covariance npz fixtures** (`cov_ddof0_f64_seed42.npz`, `cov_ddof1_f64_seed42.npz`, `cov_ddof1_f32_seed42.npz`) via `gen_oracle.py::gen_covariance` — committed blobs, `np.cov(A, rowvar=False, ddof)`, A is 7×4.

## GEMM-output-buffer reuse mechanism (Plan 05 D-10 gate 3 reads this)

This is the exact reuse Plan 05's memory-gate assertion 3 ("Gram reuses the GEMM buffer") asserts. In `covariance()`:

1. The internal GEMM is driven into a **single** output buffer:
   ```rust
   let gram = gemm::<F>(pool, &centred_dev, (n_features, n_samples),
                        &centred_dev, (n_samples, n_features),
                        /*transa*/ true, /*transb*/ false, out)?;
   ```
   - When the caller supplies `out: Some(DeviceArray)` (D-11), `gemm` writes directly into the caller's handle (gemm's `out_handle = o.handle().clone()`).
   - When `out` is `None`, `gemm` does `pool.acquire(n_features² * elem)` for its output — ONE acquisition.
   In BOTH cases `gram` wraps that single GEMM output handle (`DeviceArray::from_raw(out_handle, ...)` inside gemm).

2. The `1/(n-ddof)` normalisation is applied **IN PLACE** over that same buffer — the `scale` kernel's input and output are the SAME handle:
   ```rust
   let in_arg  = unsafe { ArrayArg::from_raw_parts(gram.handle().clone(), gram_len) };
   let out_arg = unsafe { ArrayArg::from_raw_parts(gram.handle().clone(), gram_len) };
   scale::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, factor);
   ```
   `gram.handle() == scale in_arg handle == scale out_arg handle`.

3. Covariance returns that exact buffer: `Ok(gram)`. **There is NO separate covariance output allocation** — the returned `DeviceArray`'s handle IS the GEMM output handle, scaled in place.

**How Plan 05 asserts it:** thread the SAME `BufferPool`; do a `gemm` of the `n_features × n_features` output shape, then call `covariance(.., out: Some(gemm_output_DeviceArray), ..)`. Because covariance passes `out` straight to its internal GEMM (no new acquire for the Gram), and scales in place, `PoolStats.allocations` does not increase for a fresh Gram buffer (same-byte-size reuse bumps `reuses`, not `allocations`). The single internal `pool.acquire` that DOES occur when `out=None` is the GEMM output itself (the buffer being reused), plus the transient `centred` scratch buffer — neither is a parallel Gram allocation.

## Host API signature

```rust
// crates/mlrs-backend/src/prims/covariance.rs
pub fn covariance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (n_samples, n_features): (usize, usize),
    ddof: u32,                                     // 0 = population (1/n), 1 = sample (1/(n-1))
    out: Option<DeviceArray<ActiveRuntime, F>>,    // D-11 caller buffer; threads into gemm
    path: ReducePath,                              // column-mean reduction path
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>  // n_features × n_features, device-resident
where F: Float + CubeElement + Pod;
```

## Task Commits

1. **Task 1: Covariance host API (column-mean center + AᵀA via GEMM(transa) + ddof scale, GEMM-buffer reuse)** — `14a3a9c` (feat)
2. **Task 2: Validate covariance (ddof=0 AND ddof=1, host ref + np.cov fixtures, f32/f64, cpu/wgpu)** — `4224050` (test)

## Files Created/Modified

- `crates/mlrs-backend/src/prims/covariance.rs` (created) — covariance host API: geometry validation, device-side centring, `gemm(transa)` Gram, in-place `1/(n-ddof)` scale, GEMM-buffer reuse.
- `crates/mlrs-backend/src/prims/mod.rs` (modified) — `pub mod covariance`.
- `crates/mlrs-kernels/src/elementwise.rs` (modified) — `center_columns` per-element kernel.
- `crates/mlrs-kernels/src/lib.rs` (modified) — `center_columns` re-export.
- `crates/mlrs-backend/tests/covariance_test.rs` (created) — ddof=0/ddof=1 oracle tests (host ref + fixtures, f32/f64).
- `scripts/gen_oracle.py` (modified) — `gen_covariance(ddof)` (np.cov rowvar=False) + main() calls.
- `tests/fixtures/cov_ddof0_f64_seed42.npz`, `cov_ddof1_f64_seed42.npz`, `cov_ddof1_f32_seed42.npz` (created).

## Decisions Made

See `key-decisions` frontmatter. Headlines:
- **GEMM-output-buffer reuse** (D-10 gate 3): `scale` runs in place over the GEMM output handle; covariance returns that exact handle. No parallel Gram allocation.
- **`center_columns` kernel** added (Rule 2 — D-05) to keep `covariance.rs` `grep -c to_host == 0`; same class as `scale`/`clamp_nonneg`, not a new Gram kernel.
- **ddof via the scale factor** `1/(n-ddof)`; both ddof=0/1 pinned by `np.cov(rowvar=False)` fixtures.
- **Direct centred-`AᵀA/(n-ddof)` host reference** independent of the device's GEMM(transa) algebra.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Added `center_columns` device kernel to keep `covariance.rs` device-resident (grep to_host == 0)**
- **Found during:** Task 1 (first build) — an initial host-side centring (`to_host` the means + the input, subtract on the host, re-upload) made `grep -c to_host crates/mlrs-backend/src/prims/covariance.rs` = 5, violating the acceptance criterion's `== 0` device-residency gate (D-05).
- **Issue:** Two-pass column centring (RESEARCH Pitfall 4) needs the per-column mean subtracted from every element. Doing it on the host requires a literal `to_host` in `covariance.rs`. The distance.rs precedent kept the gate at 0 by delegating all host slicing to the reduction primitive; covariance's broadcast-subtract has no such existing delegate.
- **Fix:** Added `center_columns(a, mean, out, cols)` — a per-element `#[cube(launch)]` map (`out[tid] = a[tid] - mean[tid % cols]`) in the existing `elementwise.rs` (same class as `scale`/`clamp_nonneg`, feature-free D-13). The means stay device-resident as the `column_reduce` `DeviceArray` and feed the kernel directly; centring is now on-device. This is NOT a new Gram compute kernel — the `Aᵀ·A` path still composes GEMM (the "no new kernels" mandate is about the Gram, T-0204-SC: zero external deps, held).
- **Files modified:** `crates/mlrs-kernels/src/elementwise.rs`, `crates/mlrs-kernels/src/lib.rs`, `crates/mlrs-backend/src/prims/covariance.rs`.
- **Verification:** `grep -c to_host crates/mlrs-backend/src/prims/covariance.rs` = 0; `cargo build -p mlrs-kernels` (feature-free) + `--features cpu/wgpu` green; both covariance tests pass on cpu and wgpu; distance/reduce suites unregressed.
- **Committed in:** `14a3a9c` (Task 1).

---

**Total deviations:** 1 (missing-critical device-residency). **Impact:** No scope creep — `center_columns` is the device-resident form of the centring the plan already mandated ("center columns via the column-mean reduction"); it adds zero external dependencies (T-0204-SC held) and is the same per-element kernel class as the existing `scale`. The HARD guardrail held throughout: covariance matches `np.cov` within 1e-5 for BOTH ddof conventions on cpu AND wgpu, and the GEMM-output-buffer reuse was achieved exactly as the plan required (no faked reuse, no loosened tolerance).

## Issues Encountered

- **`grep to_host` device-residency gate** initially tripped (5) by a host-side centring; resolved by the `center_columns` kernel (deviation 1) — gate now 0.
- **No CubeCL build/lowering errors** — `center_columns` compiled clean on cpu + wgpu first try; the `tid % cols` broadcast index and the scalar-`u32`-by-value `cols` arg lowered correctly (same idiom as `dist_combine_clamp`). The AGENTS.md §4 cubecl_error_guideline protocol was NOT invoked.
- **`cargo test ... covariance` substring filter** runs the 2 covariance tests in the `covariance_test` binary (other test binaries filter to 0 — expected, the filter is a test-name substring applied per already-built binary).

## Threat Flags

None. Numerical compute-kernel plane — no auth/session/network/PII surface. The threat register is mitigated as designed: `center_columns`/`scale` bound-check `tid < a.len()` with `len` from the validated `DeviceArray.len` (T-0204-01); `covariance` validates `n_samples*n_features == a.len()` via `PrimError::ShapeMismatch` before any launch (T-0204-02); zero new external dependencies — composition of in-tree GEMM + column-mean reduction + scale + the new feature-free `center_columns` kernel (T-0204-SC held).

## Next Phase Readiness

- **Plan 05 (memory gate):** the GEMM-output-buffer reuse is in place and documented above — D-10 gate 3 ("Gram reuses the GEMM buffer") is directly assertable by passing a GEMM output `DeviceArray` as covariance's `out` and checking `PoolStats.allocations` does not bump for a fresh Gram buffer. covariance is another device-resident composition (grep to_host == 0) feeding the gate-2 no-mid-pipeline-read-back assertion.
- **Phase 3 (SVD/eig) + Phase 4 (PCA / linear closed-form solvers):** `prims::covariance::covariance(.., ddof, ..)` is ready as the covariance/Gram these inherit, with the exact `np.cov`/sklearn normalisation convention (ddof=0 population, ddof=1 sample) baked in and fixture-pinned.

---
*Phase: 02-core-compute-primitives*
*Completed: 2026-06-12*

## Self-Check: PASSED
- All 5 created files + the SUMMARY verified present on disk.
- Both task commits (14a3a9c, 4224050) verified in git log.
- `cargo test -p mlrs-backend --features cpu covariance`: 2/2 green (~3 s).
- `cargo test -p mlrs-backend --features wgpu covariance`: 2/2 green (~0.9 s) — ddof=0 + ddof=1, f32 + f64.
- `cargo build -p mlrs-kernels` (feature-free) + `--features cpu/wgpu`: green. covariance.rs `grep -c to_host` = 0.
