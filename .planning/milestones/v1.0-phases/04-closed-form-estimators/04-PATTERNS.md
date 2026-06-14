# Phase 4: Closed-Form Estimators - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 14 new + 3 modified
**Analogs found:** 14 / 14 (every new file has a strong in-tree analog from Phase 1–3)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-kernels/src/cholesky.rs` (NEW) | kernel (`#[cube(launch)]`) | transform (in-kernel iterative solve) | `crates/mlrs-kernels/src/jacobi_eig.rs` | exact (single-cube, all-LDS, n≤MAX_DIM blueprint) |
| `crates/mlrs-backend/src/prims/cholesky.rs` (NEW) | prim launch wrapper / service | request-response (validate→launch→device result) | `crates/mlrs-backend/src/prims/eig.rs` | exact (square `n×n`, info-array convergence/flag, `out=Some` reuse) |
| `crates/mlrs-backend/src/prims/mod.rs` (MODIFY) | config (module index) | — | existing `pub mod` list | exact |
| `crates/mlrs-core/src/error.rs` (MODIFY) | model (error enum) | — | `PrimError` variants (`NotConverged`/`NotSquare`) | exact (add `NotPositiveDefinite`) |
| `crates/mlrs-algos/src/lib.rs` (MODIFY) | config (re-exports + module decl) | — | `crates/mlrs-backend/src/prims/mod.rs` | role-match (doc-comment + `pub mod` index) |
| `crates/mlrs-algos/src/traits.rs` (NEW) | trait surface (Fit/Predict/Transform) | request-response | sklearn mixin pattern (no in-tree analog) | NO ANALOG (use RESEARCH D-04 shape) |
| `crates/mlrs-algos/src/linear/linear_regression.rs` (NEW) | estimator (service-over-prims) | transform (SVD pseudo-inverse) | `crates/mlrs-backend/src/prims/svd.rs` host orchestration | role-match (host composition over `svd`+`gemm`+`reduce`) |
| `crates/mlrs-algos/src/linear/ridge.rs` (NEW) | estimator | transform (Cholesky normal-eq) | `crates/mlrs-backend/src/prims/covariance.rs` (Gram reuse) + `svd.rs` (host arithmetic) | role-match |
| `crates/mlrs-algos/src/decomposition/pca.rs` (NEW) | estimator | transform (center→SVD) | `crates/mlrs-backend/src/prims/svd.rs` + `sign_flip::align_rows` | role-match |
| `crates/mlrs-algos/src/decomposition/truncated_svd.rs` (NEW) | estimator | transform (uncentered SVD) | `crates/mlrs-algos/.../pca.rs` (sibling) + `svd.rs` | role-match |
| `crates/mlrs-backend/tests/cholesky_test.rs` (NEW) | test (prim oracle + invariant) | — | `crates/mlrs-backend/tests/svd_test.rs` | exact |
| `crates/mlrs-algos/tests/{linear_regression,ridge,pca,truncated_svd}_test.rs` (NEW) | test (estimator oracle) | — | `crates/mlrs-backend/tests/svd_test.rs` | exact |
| `crates/mlrs-algos/tests/memory_gate_test.rs` (NEW) OR extend `mlrs-backend/tests/memory_gate_test.rs` | test (PoolStats gate) | — | `crates/mlrs-backend/tests/memory_gate_test.rs` (D-11 §) | exact |
| `scripts/gen_oracle.py` (MODIFY) | utility (fixture generator) | batch (numpy/sklearn → `.npz`) | existing `gen_saxpy` / SVD generators in same file | exact |
| `crates/mlrs-algos/Cargo.toml` (MODIFY) | config | — | `crates/mlrs-backend/Cargo.toml` deps | exact |

## Pattern Assignments

### `crates/mlrs-kernels/src/cholesky.rs` (NEW kernel, in-kernel iterative solve)

**Analog:** `crates/mlrs-kernels/src/jacobi_eig.rs` (the single-cube, all-shared-memory, in-kernel-loop blueprint — RESEARCH Pattern 2).

**Comptime LDS cap pattern** (`jacobi_eig.rs:92-97`):
```rust
/// Comptime dimension cap for the shared tiles (MAX_DIM × MAX_DIM). At f32 this
/// is 64·64·4 = 16 KiB ... within gfx1100's 64 KiB LDS.
pub const MAX_DIM: u32 = 64;
```
Reuse `MAX_DIM` (already exported from `mlrs-kernels` and consumed by `eig.rs`); the host rejects `n > MAX_DIM` before launch. A 64×64 f64 L factor = 32 KiB still fits (RESEARCH A3).

**Kernel signature + shared-memory staging** (`jacobi_eig.rs:119-155`): copy the `#[cube(launch)]` attribute, `<F: Float + CubeElement>` bound, `&Array<F>` in / `&mut Array<F>` out params, the `info_out` length-2 array, the `SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize)` tile sized to the comptime cap, and the `let i = UNIT_POS_X;` + `if i < n { ... }` active-region staging:
```rust
#[cube(launch)]
pub fn cholesky_solve<F: Float + CubeElement>(
    a_in: &Array<F>,       // row-major n×n SPD (XᵀX + αI)
    b_in: &Array<F>,       // n×rhs (Xᵀy)
    x_out: &mut Array<F>,  // n×rhs solution
    info_out: &mut Array<F>, // [0] = non-SPD flag (negative pivot)
    n: u32, rhs: u32,
) {
    let mut l_sh = SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize);
    let i = UNIT_POS_X;
    // ... stage rows of A into l_sh (mirror jacobi_eig.rs:146-155) ...
    sync_cube();
    // phase 1: Cholesky-Banachiewicz; phase 2: forward solve; phase 3: back solve
}
```

**Acting-unit idiom + `sync_cube()` between phases** (`jacobi_eig.rs:179-244`): start with the simplest correct version — unit 0 does the whole factorization/solve while others idle, with `sync_cube()` between the three phases (RESEARCH Open Q2 recommends unit-0 sequential since n≤64). This is the EXACT idiom at `jacobi_eig.rs:189 if i == 0u32 { ... } sync_cube();`.

**CRITICAL CubeCL constraints copied from `jacobi_eig.rs`:**
- `continue` is NOT supported in `#[cube]` → use `if`-wrap (jacobi_eig.rs:198 — guard the diagonal sqrt arg: `if arg > floor { sqrt } else { set info flag }`).
- `SharedMemory::new(N)` needs a COMPILE-TIME size → size to `MAX_DIM*MAX_DIM`, bound active loops by runtime `n`.
- Generic constants via `F::from_int(0i64)` / `F::new(..)`; `Float` methods `.abs()` / `.sqrt()` (jacobi_eig.rs:139-141, 203, 283).
- NO hardcoded plane width / 32 — use the shared-memory layout, not a plane path (jacobi_eig.rs:83-84 anti-pattern; carried no-hardcoded-plane-width rule).
- Non-SPD pivot writes `info_out` (mirror the sweep-count/residual write at `jacobi_eig.rs:304-307`), surfaced as `PrimError::NotPositiveDefinite` on the host (RESEARCH Pitfall 4).

**File-level doc + AGENTS.md §2 note** (`jacobi_eig.rs:1-88`): copy the module-doc style — what/why, layout, CubeCL expression notes, and the trailing `// tests live in crates/mlrs-backend/tests/cholesky_test.rs` line (NO in-source `mod tests`).

---

### `crates/mlrs-backend/src/prims/cholesky.rs` (NEW launch wrapper, request-response)

**Analog:** `crates/mlrs-backend/src/prims/eig.rs` (square-`n×n` prim, info-array convergence read, `out=Some` buffer reuse) + `svd.rs` host helpers.

**Validate-before-launch (ASVS V5)** — mirror `eig.rs:92` / `eig.rs:210-233` `validate_geometry`:
```rust
validate_geometry(a.len(), n, out.as_ref().map(DeviceArray::len))?;
// reject n*n != a.len() → PrimError::NotSquare; n > MAX_DIM → NotSquare; also
// validate b.len() == n*rhs, alpha ≥ 0 at the Ridge call site.
```

**Signature shape (D-08 conventions)** — copy the `eig.rs:75-92` / `covariance.rs:89-98` form: `pool: &mut BufferPool<ActiveRuntime>`, device-array inputs, explicit `(rows, cols)` / `n`, `out: Option<DeviceArray<..>>` for buffer reuse (D-11), `Result<.., PrimError>`, `where F: Float + CubeElement + Pod`.

**`out=Some` Gram-buffer reuse (memory gate)** — `eig.rs` threads the caller's `out` straight through as the kernel working input; do the SAME so Ridge can pass the Gram `XᵀX` buffer through (D-11 gate 2 / RESEARCH Pattern 3). The Gram itself comes from `gemm(transa=true)` (RESEARCH Open Q1 — call gemm directly for the RAW `XᵀX`, NOT scaled `covariance`).

**Info-array read + typed error** — mirror `eig.rs:153-167` / `svd.rs:231-250`:
```rust
let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 2);
let info = info_dev.to_host(pool);
info_dev.release_into(pool);
if host_to_f64(info[0]) < 0.0 {  // non-SPD pivot flag
    return Err(PrimError::NotPositiveDefinite { operand: "cholesky", .. });
}
```

**SAFETY / ArrayArg pattern** — copy the `svd.rs:206-209` `unsafe { ArrayArg::from_raw_parts(handle.clone(), validated_len) }` with the carried/validated element count (never raw caller geometry), single-cube `CubeCount::Static(1,1,1)` + `CubeDim { x: n, .. }` (svd.rs:194-200).

**Host float-bridge helpers** — reuse the `host_to_f64` / `f64_to_host` `bytemuck` pair verbatim from `svd.rs:445-461`.

---

### `crates/mlrs-algos/src/linear/linear_regression.rs` (NEW estimator, SVD pseudo-inverse)

**Analog:** `crates/mlrs-backend/src/prims/svd.rs` host orchestration (the closest "compose prims + host arithmetic" pattern in-tree).

**Prim call (VERIFIED signature, `svd.rs:90`):**
```rust
let (u, s, vt) = svd::<F>(pool, &x_centered_dev, (n_samples, n_features))?;
// coef = V · diag(σ⁺) · Uᵀ · y_centered ;  σ⁺ = 1/σ if σ > cond·σ_max else 0
```

**Small-σ cutoff (RESEARCH Pitfall 1 + `svd.rs:72` precedent):** reuse the `NEAR_ZERO_FLOOR = 1e-8` floor idea from `svd.rs:68-72` as a fallback; primary cutoff is `cond·σ_max` (`cond ≈ eps·max(m,n)`, Claude's Discretion D-02). The `svd.rs:289` `if sj > NEAR_ZERO_FLOOR { 1/sj } else { 0 }` guard is the exact column-skip pattern to copy for `σ⁺`.

**Centering / intercept (D-05)** — `column_reduce(.., ScalarOp::Mean, ReducePath::Shared)` from `reduce.rs` (VERIFIED `reduce.rs:263 Mean`); intercept recovered host-side as `ȳ − x̄·coef`. The two-pass center pattern is in `covariance.rs:103-119`.

**GEMM products** — `gemm::<F>(pool, a, (m,k), b, (k,n), transa, transb, out)` (VERIFIED `gemm.rs:54`); transpose flags are zero-copy logical (D-06) — use `transb=true` to read a `Vᵀ`-stored buffer as `V` exactly as `svd.rs:258-267`.

**Device-resident fitted state (D-03):** store `coef_` / `intercept_` as `DeviceArray<ActiveRuntime, F>` fields; `predict` = `gemm` on-device; host materialize only at accessor/oracle time.

---

### `crates/mlrs-algos/src/linear/ridge.rs` (NEW estimator, Cholesky normal-eq)

**Analog:** `crates/mlrs-backend/src/prims/covariance.rs` (Gram + buffer-reuse contract) consuming the NEW `prims::cholesky`.

**Raw Gram (RESEARCH Open Q1 — NOT scaled covariance):**
```rust
// covariance() centers + scales by 1/(n-ddof); Ridge wants RAW XᵀX → call gemm directly:
let gram = gemm::<F>(pool, &x_c, (n_features, n_samples), &x_c, (n_samples, n_features),
                     /*transa*/ true, /*transb*/ false, None)?; // n_features²
// add alpha to the diagonal (NOT to intercept — D-05); then prims::cholesky(gram_as_out, xty)
```
The `covariance.rs:19-29` doc explains the GEMM-output-buffer reuse contract verbatim — thread the Gram handle through `cholesky`'s `out` so no parallel `n²` alloc (memory gate, D-11 gate 2).

**Intercept (D-05, RESEARCH Pitfall 5):** center BOTH X and y, solve on centered, recover `intercept_ = ȳ − x̄·coef`; α NEVER penalizes the intercept. Same `column_reduce(ScalarOp::Mean)` path as LinearRegression.

**Anti-pattern:** do NOT unify with LinearRegression's SVD solver (RESEARCH Anti-Patterns; LINEAR-01 pins SVD, LINEAR-02 pins Cholesky).

---

### `crates/mlrs-algos/src/decomposition/pca.rs` (NEW estimator, center→SVD)

**Analog:** `crates/mlrs-backend/src/prims/svd.rs` + `crates/mlrs-core/src/sign_flip.rs`.

**sklearn `_fit_full` arithmetic (RESEARCH VERIFIED):** `mean_ = col means` → center → `svd(centered)` → `components_ = Vᵀ[:n_components]` after flip → `explained_variance_ = S²/(n−1)` → `explained_variance_ratio_ = ev / ev.sum()` (sum over ALL S — compute BEFORE truncation, RESEARCH Pitfall 6).

**svd_flip applied by the ESTIMATOR (D-01/D-03):** `align_rows` from `sign_flip.rs:60` == sklearn `svd_flip(u_based_decision=False)` (CONFIRMED). Apply to the `Vᵀ` rows; the row-split helper is `rows_of` in `svd_test.rs:169-173`. Primitive stays RAW (anti-pattern: never flip inside `prims::svd`).

**Truncation (Pitfall 6):** total variance from ALL S first, THEN keep top `n_components`; validate `n_components ≤ min(m,n)` at fit (new error variant).

---

### `crates/mlrs-algos/src/decomposition/truncated_svd.rs` (NEW estimator, uncentered SVD)

**Analog:** sibling `pca.rs` — SAME `svd` + `align_rows` skeleton, three documented differences (RESEARCH Pattern 1 + Pitfall 2):
- NO centering (feed uncentered X).
- `explained_variance_ = var(transform(X) columns)`, NOT `S²/(n−1)`.
- `explained_variance_ratio_` denominator = sum of per-feature variances of ORIGINAL X.
- Oracle fixture uses `algorithm='arpack'` (deterministic), NOT `'randomized'` (D-07).

---

## Shared Patterns

### Capability gating (every f64 oracle test)
**Source:** `crates/mlrs-backend/tests/svd_test.rs:209-225` + `capability` API (`capability.rs:132,147`).
**Apply to:** all 5 new test files (cholesky + 4 estimators), every f64 case.
```rust
let backend = capability::active_backend_name();
capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
if capability::skip_f64_with_log() {
    println!("<op> f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
    return;
}
```
f64 validates on cpu, skips-with-log on rocm; f32 validates on rocm (D-07, supersedes cpu+wgpu).

### svd_flip sign alignment (every decomposition/SVD compare)
**Source:** `crates/mlrs-core/src/sign_flip.rs:60` (`align_rows`) + `svd_test.rs:159-173` (`columns` / `rows_of` splitters).
**Apply to:** PCA, TruncatedSVD, LinearRegression oracle compares — sign-align BOTH device output AND fixture before `assert_close` (idempotent, RESEARCH Pitfall 3).

### Tolerance + near-zero-floored compare
**Source:** `crates/mlrs-backend/tests/svd_test.rs:65-96` (`assert_close_floored`) + `mlrs_core::{F32_TOL, F64_TOL, Tolerance}`.
**Apply to:** all estimator oracle tests. Strict 1e-5 abs (never loosened) OR'd with 1e-5 rel (numpy-allclose family); per-family looser bound (Phase-3 D-10) only if a genuinely ill-conditioned case forces it. f32 near-zero floor precedent: `F32_SVD_NEAR_ZERO_FLOOR = 1e-2` (svd_test.rs:37).

### Fixture path + loader
**Source:** `crates/mlrs-backend/tests/svd_test.rs:40-47` (`fixture()` resolves `<workspace_root>/tests/fixtures/<name>`) + `mlrs_core::{load_npz, OracleCase}` (`oracle.rs:20,77`; accessors `expect_f64`/`f32`/`shape`).
**Apply to:** all 5 new test files. Naming convention `case_dtype_seed.npz` (e.g. `pca_f32_seed42.npz`).

### PoolStats memory gate (D-03 extension)
**Source:** `crates/mlrs-backend/tests/memory_gate_test.rs` — Phase-3 D-11 section (lines 441-817) is the closest analog (iterative-loop scratch bounded, buffer reuse, `read_backs` terminal-only).
**Apply to:** the new fit→predict/transform gate. Copy the three-gate structure:
- Gate 1 (`memory_gate_jacobi_scratch_bounded:541`): N same-shape calls → `alloc_delta == 0`, `live_bytes`/`peak_bytes` conserve.
- Gate 2 (`memory_gate_eig_reuses_gram_buffer:660`): peak-rise `< 2·n²` proves Gram/factor buffer reuse (the Ridge Cholesky-reuses-Gram check).
- Gate 3 (`memory_gate_svd_no_midsweep_readback:753`): `read_backs == 0` after fit, `== 1` after terminal `to_host_metered`.

### Error enum extension (thiserror)
**Source:** `crates/mlrs-core/src/error.rs:65-134` (`PrimError` — `NotConverged`/`NotSquare` are the templates).
**Apply to:** add `PrimError::NotPositiveDefinite { operand, .. }` (Cholesky negative pivot, RESEARCH Pitfall 4) and estimator-side variants (e.g. `n_components` out of range, RESEARCH Pitfall 6) following the same `#[error("...")]` + named-field style. `thiserror` in libs, `anyhow` at boundaries (D-08, project memory).

### Prim module registration
**Source:** `crates/mlrs-backend/src/prims/mod.rs:12-17` (`pub mod <name>;`).
**Apply to:** add `pub mod cholesky;` to `prims/mod.rs`.

### Fixture generator (gen_oracle.py)
**Source:** `scripts/gen_oracle.py` — existing `gen_saxpy` (line 80) + the SVD/eig shape constants (lines 60-76) are the template; the module-doc (lines 9-12) already anticipates the Phase-4 `import sklearn` extension.
**Apply to:** add `gen_cholesky` (numpy/scipy reference: `scipy.linalg.cholesky` / `solve_triangular`), `gen_linear_regression`, `gen_ridge`, `gen_pca`, `gen_truncated_svd` (sklearn, `algorithm='arpack'`). Regen needs a /tmp venv with numpy+scipy+scikit-learn (PEP 668, project memory). Fixtures are committed blobs — never run in CI.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/mlrs-algos/src/traits.rs` | trait surface (Fit/Predict/Transform) | request-response | First trait abstraction in the workspace; no in-tree estimator-trait analog. Use RESEARCH D-04 shape: `Fit` returns `&mut self`/`self`; mirror sklearn `RegressorMixin`/`TransformerMixin`. LinearRegression/Ridge = `Fit`+`Predict`; PCA = `Fit`+`Transform`[+inverse]; TruncatedSVD = `Fit`+`Transform`. |

The estimator bodies themselves have only ROLE-match analogs (the `prims/*.rs` host-orchestration files), not exact ones — `mlrs-algos` is greenfield. The estimators compose VERIFIED prim signatures (svd/gemm/covariance/reduce/cholesky) with host arithmetic; the "analog" is the orchestration STYLE of `svd.rs`/`covariance.rs`, not a like-for-like estimator.

## Metadata

**Analog search scope:** `crates/mlrs-kernels/src/`, `crates/mlrs-backend/src/prims/`, `crates/mlrs-backend/tests/`, `crates/mlrs-core/src/`, `crates/mlrs-algos/`, `scripts/`
**Files scanned:** ~18 (kernels, prims, tests, core helpers, gen_oracle)
**Pattern extraction date:** 2026-06-12
