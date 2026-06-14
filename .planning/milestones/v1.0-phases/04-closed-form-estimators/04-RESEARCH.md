# Phase 4: Closed-Form Estimators - Research

**Researched:** 2026-06-12
**Domain:** sklearn-compatible closed-form linear models + decomposition estimators (Rust/CubeCL); new dense Cholesky linear-solve primitive
**Confidence:** HIGH (estimator math + reusable assets verified against sklearn source and existing code); MEDIUM (the new Cholesky/triangular-solve kernel — pattern is proven by the existing Jacobi kernels but the kernel itself is unwritten)

## Summary

Phase 4 assembles four sklearn-compatible Rust estimators (`LinearRegression`, `Ridge`, `PCA`, `TruncatedSVD`) in the empty `mlrs-algos` crate on top of the already-validated Phase-2/3 primitives (thin SVD, covariance/Gram, GEMM, column-mean reduction, `align_rows` sign-flip). Three of the four estimators are pure host-side orchestration over the **existing thin SVD** primitive plus arithmetic — no new kernel needed. The single new device kernel is the **Cholesky factorization + triangular solve** that `Ridge` requires (D-02), which has no Phase-2/3 analogue and is the highest implementation risk. That kernel can follow the **exact single-cube, shared-memory, in-kernel pattern already proven by `jacobi_eig.rs`** (small SPD matrix `n = n_features ≤ MAX_DIM`, fits LDS), so the risk is well-bounded by an established blueprint.

The numerical contract is **scikit-learn**, verified directly against sklearn source this session: PCA `svd_solver='full'` = center → `scipy.linalg.svd(full_matrices=False)` → `svd_flip(u_based_decision=False)` → `explained_variance_ = S²/(n−1)`; Ridge cholesky = `linalg.solve(XᵀX + αI, Xᵀy, assume_a="pos")` with intercept via centering; LinearRegression = `scipy.linalg.lstsq` (SVD pseudo-inverse) with the gelsd small-singular-value cutoff; TruncatedSVD `algorithm='arpack'` = uncentered thin SVD, `explained_variance_` = variance of the transformed columns. The `svd_flip(u_based_decision=False)` convention **exactly matches the existing `mlrs-core::sign_flip::align_rows`** — confirmed by reading both implementations — so no new sign-flip logic is needed.

**Primary recommendation:** Sequence the new Cholesky/triangular-solve primitive FIRST (standalone-validated f32+f64, cpu+rocm, against a numpy/scipy reference plus the algebraic invariants `‖L·Lᵀ − A‖` and `‖A·x − b‖`) before Ridge consumes it, mirroring Phase-2/3 primitive-first discipline. Build the three SVD-backed estimators in parallel — they share the thin-SVD primitive and `align_rows` and carry no convergence risk. Hold the global 1e-5 abs+rel tolerance with the Phase-3 D-10 per-family looser-bound escape hatch reserved for genuinely ill-conditioned cases only.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Estimator API (`Fit`/`Predict`/`Transform` traits, structs) | `mlrs-algos` (host orchestration) | — | Estimators compose primitives; no device code of their own (D-04) |
| Centering / mean recovery (intercept, PCA `mean_`) | `mlrs-backend` prims (column-mean reduce) | `mlrs-algos` host arithmetic | Reuses Phase-2 `column_reduce(ScalarOp::Mean)` (D-05) |
| Thin SVD (PCA, LinearRegression, TruncatedSVD) | `mlrs-backend::prims::svd` | `mlrs-kernels::jacobi_svd_sweep` | One Phase-3 primitive, three consumers (D-01/D-02) |
| Gram XᵀX (Ridge normal matrix) | `mlrs-backend::prims::covariance` | — | Reuses Phase-2 covariance/Gram, reuse its out buffer (D-02) |
| **Cholesky factorization + triangular solve (Ridge)** | **`mlrs-backend::prims` (NEW)** | **`mlrs-kernels` (NEW `#[cube]`)** | **No Phase-2/3 analogue — the in-phase sub-deliverable (D-02)** |
| `svd_flip` sign canonicalization | `mlrs-core::sign_flip::align_rows` (host) | `mlrs-algos` calls it | Estimator flips; primitive stays raw (D-01/D-03) — already exists |
| Device-resident fitted state | `mlrs-backend::device_array` + `pool` | `mlrs-algos` holds `DeviceArray` | Fitted attrs stay on-device, lazy host materialize (D-03) |
| Oracle comparison (sklearn fixtures) | `mlrs-core::oracle` + `compare` | `scripts/gen_oracle.py` | Reuses Phase-1 harness; sklearn now in /tmp venv (D-07) |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — PCA `full` = SVD of centered X** (NOT eig-of-covariance). Center X by column
  means → Phase-3 thin SVD on centered matrix → `explained_variance_ = S²/(n−1)`,
  `explained_variance_ratio_ = explained_variance_ / total_variance`, `components_ = Vᵀ`
  (after `svd_flip`), `singular_values_ = S`, `mean_ = column means`,
  `transform(X) = (X − mean)·V`, `inverse_transform(Z) = Z·Vᵀ + mean`. The Phase-3 eig
  primitive is **NOT consumed by PCA in v1** (validated standalone asset, intentional non-use).
  `svd_flip` applied **by the estimator** (D-03 P3), reusing `mlrs-core/src/sign_flip.rs` `align_rows`.
- **D-02 — Ridge = Cholesky normal-equations.** Solve `(XᵀX + αI)·coef = Xᵀy` via Cholesky
  factorization + triangular solve (sklearn dense `solver='auto'`→cholesky). Reuses the Phase-2
  covariance/Gram primitive. **Requires a NEW Cholesky+triangular-solve primitive** in
  `mlrs-backend/src/prims/` + feature-free `#[cube]` kernel in `mlrs-kernels`, validated
  standalone (f32+f64, cpu+rocm) against numpy/sklearn + algebraic invariant
  (`‖L·Lᵀ − A‖`, `‖A·x − b‖`) BEFORE Ridge consumes it. Memory gate + tolerance policy apply.
  **Single highest implementation risk.** LinearRegression (LINEAR-01) is SEPARATE and
  SVD-based (`coef = V·diag(1/σ)·Uᵀ·y` with sklearn small-σ cutoff) — do NOT unify the two solvers.
- **D-03 — Fitted attributes are device-resident (`DeviceArray`).** `coef_`, `intercept_`,
  `components_`, `mean_`, `singular_values_`, `explained_variance_`, etc. stay on-device after
  `fit`. `predict`/`transform`/`inverse_transform` run device-side, no host round-trip; host
  materialize happens lazily at accessor/oracle time. Memory gate (P2 D-10 / P3 D-11) extends here.
- **D-04 — Shared traits `Fit`, `Transform`, `Predict`** (sklearn-mixin-style) in `mlrs-algos`.
  LinearRegression/Ridge: `Fit`+`Predict`; PCA: `Fit`+`Transform`[+inverse]; TruncatedSVD:
  `Fit`+`Transform`. `fit` returns `&mut self`/`self`. Consumed by Phase-6 PyO3.
- **D-05 — Intercept via center-then-solve.** `fit_intercept=true` (default) centers X and y by
  column means, solves for `coef_` on centered data, recovers `intercept_ = ȳ − x̄·coef_`. Ridge
  does NOT penalize the intercept (centering handles this). Reuses Phase-2 column-mean reduction.
- **D-06 — v1 constructor surface minimal, `n_components` is integer.**
  `LinearRegression { fit_intercept }`, `Ridge { alpha, fit_intercept }`, `PCA { n_components }`,
  `TruncatedSVD { n_components }`. `n_components` int only (`k ≤ min(n_samples, n_features)`).
  Float/`'mle'`/`None` semantics deferred. `copy_X`/`tol`/`normalize`/`whiten`/random-state out of scope.
- **D-07 — Oracle source = scikit-learn fixtures** (not bare numpy) for estimator-specific
  attributes. TruncatedSVD uses deterministic `algorithm='arpack'` (NOT default `'randomized'`).
  Committed fixtures via `scripts/gen_oracle.py` (regen needs /tmp venv with numpy+scikit-learn
  per PEP 668). Compare after `align_rows`. Global 1e-5 abs+rel, per-family looser bound (P3 D-10)
  only if a real case forces it.
- **D-08 (carried) —** Estimators generic over `<F: Float + CubeElement + Pod>`; thin SVD returns
  (U[m×k], S[k], Vᵀ[k×n]) k=min(m,n); covariance takes ddof; explicit `(rows, cols)` per call;
  DeviceArray flat 1D; device-resident in/out; optional caller-out + pooled scratch; svd_flip at
  estimator time. Feature-free `#[cube]` kernels in `mlrs-kernels`; launch wrappers in
  `mlrs-backend`; estimators in `mlrs-algos`. `assert_close` 1e-5 abs+rel with near-zero floor;
  f64 capability-gated via `skip_f64_with_log`; `thiserror` in libs / `anyhow` at boundaries;
  deps latest; source/test separation (tests in `crates/*/tests/`, NO in-source `mod tests`).

### Claude's Discretion

- Module/file layout within `mlrs-algos` (e.g. `linear/`, `decomposition/` modules or
  per-estimator files) and the exact trait method signatures (D-04) — honor source/test separation.
- The new Cholesky/solve primitive's internal design (D-02): blocked vs unblocked Cholesky,
  in-place vs out-of-place, triangular-solve kernel structure — subject to the Phase-2/3 memory
  gate, tolerance policy, and no-hardcoded-plane-width rule.
- LinearRegression's small-singular-value cutoff constant (D-02) to match
  sklearn/`scipy.linalg.lstsq` `cond`/`rcond` default — pick the value that holds 1e-5.
- Exact random shapes/seeds for the estimator oracle sweep, and which cases get committed sklearn
  fixtures vs algebraic-invariant-only checks (D-07).
- Naming of new estimator/primitive error variants (extend the `thiserror` enums).

### Deferred Ideas (OUT OF SCOPE)

- `n_components` as float (variance-ratio), `'mle'`, or `None`=all — v1 int only (D-06).
- Additional sklearn constructor knobs — `copy_X`, `tol`, `whiten` (PCA), randomized SVD +
  `random_state`, `positive`/`normalize` (linear models). Out of v1 (D-06).
- Ridge alternative solvers (`svd`, `lsqr`, `sag`, `saga`, `sparse_cg`) — v1 Cholesky only (D-02).
- PCA via eig-of-covariance as a selectable solver — rejected as v1 workhorse (D-01); eig
  primitive exists, could become an alternate path later.
- Reusing the new Cholesky/solve primitive elsewhere (GLM/GP/Mahalanobis) — built generically,
  no v1 consumer beyond Ridge.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LINEAR-01 | Fit `LinearRegression` (OLS, SVD-based to match sklearn default), read `coef_`/`intercept_`, predict within 1e-5 | sklearn = `scipy.linalg.lstsq` (gelsd, SVD pseudo-inverse). Reuse Phase-3 thin SVD: `coef = V·diag(σ⁺)·Uᵀ·y`, `σ⁺ = 1/σ` for `σ > cutoff` else 0. Intercept via D-05 centering. Cutoff = `cond·σ_max` (Pitfall 1, Open Q2). |
| LINEAR-02 | Fit `Ridge` with `alpha`, read `coef_`/`intercept_` matching sklearn | sklearn cholesky = `linalg.solve(XᵀX + αI, Xᵀy, assume_a="pos")`. Reuse Phase-2 Gram; NEW Cholesky+triangular-solve kernel (D-02). Intercept via D-05 centering, α NOT applied to intercept. |
| DECOMP-01 | Fit `PCA` with `n_components`; `components_`/`explained_variance_`/`explained_variance_ratio_`/`singular_values_`/`mean_`/`transform`/`inverse_transform` matching sklearn after sign alignment | sklearn `_fit_full` verified: center → `svd(full_matrices=False)` → `svd_flip(u_based_decision=False)` → `S²/(n−1)`. Reuse thin SVD + `align_rows`. (D-01) |
| DECOMP-02 | Fit `TruncatedSVD` (no centering); `components_`/`explained_variance_`/`singular_values_`/`transform` matching sklearn `arpack` after sign alignment | sklearn `arpack` verified: NO centering; thin SVD of X; `components_=Vᵀ`; `explained_variance_` = variance of `transform(X)` columns; `svd_flip(u_based_decision=False)`. Reuse thin SVD + `align_rows`. (D-07) |
</phase_requirements>

## Backend Gate Reconciliation (FLAGGED for planner)

The ROADMAP §"Phase 4" success criterion 1 says "via cpu and **wgpu**", and PROJECT.md/REQUIREMENTS.md carry the same cpu+wgpu wording. **This is superseded.** Phase-3 D-07 (logged in STATE.md and verified empirically in 03-01) made the project-wide runnable gate **cpu(f64) + rocm(f32)**:

- f64 validates on **cpu** (`f64_supported=true`).
- f32 validates on **rocm** (gfx1100, ROCm 7.1.1; `f32_supported=true`).
- **f64-on-rocm SKIPS-with-log** via the unchanged `skip_f64_with_log` gate — cubecl-cpp 0.10 does NOT register F64 for the HIP backend (EXPECTED, not a defect). [VERIFIED: STATE.md 03-01 + project memory `rocm-is-runnable-gpu-gate`]
- wgpu is **opportunistic only**.

**Planner action:** every Phase-4 f64 oracle test MUST mirror the `gemm_test.rs` / `svd_test.rs` `skip_f64_with_log` pattern (f64 runs on cpu, skips on rocm). Read "cpu+wgpu" in any inherited doc as "cpu+rocm". This is consistent across all four estimators and the new Cholesky primitive. [CITED: 04-CONTEXT.md D-07 scope anchor]

## Standard Stack

No new external crates are introduced this phase. Phase 4 is pure first-party Rust: estimator
structs/traits in `mlrs-algos`, a new hand-written `#[cube]` kernel in `mlrs-kernels`, and a launch
wrapper in `mlrs-backend`. All compute composes existing primitives + one new kernel.

### Core (existing, consumed)
| Crate / module | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (default-features=false) | Generic `#[cube]` kernel for the new Cholesky/solve | The project's only device-kernel substrate (FOUND-02) [VERIFIED: Cargo.toml] |
| `cubek-matmul` / `cubek-std` | 0.2.0 | GEMM substrate behind `prims::gemm` (predict, transform, pseudo-inverse products) | Phase-2 D-01 wrap [VERIFIED: mlrs-backend/Cargo.toml] |
| `mlrs-backend::prims::svd` | in-tree | Thin SVD — PCA/LinearRegression/TruncatedSVD workhorse | Phase-3 validated (PRIM-05) [VERIFIED: prims/svd.rs] |
| `mlrs-backend::prims::covariance` | in-tree | Gram XᵀX for Ridge normal matrix | Phase-2 validated (PRIM-04) [VERIFIED: prims/covariance.rs] |
| `mlrs-backend::prims::gemm` | in-tree | A·V, Uᵀ·y, X·components_ᵀ, predict X·coef | Phase-2 validated (PRIM-01) [VERIFIED: prims/gemm.rs] |
| `mlrs-backend::prims::reduce` | in-tree | column-mean (`ScalarOp::Mean`) centering, column L2-norm | Phase-2 validated (PRIM-02) [VERIFIED: prims/reduce.rs] |
| `mlrs-core::sign_flip` | in-tree | `align_rows` = sklearn `svd_flip(u_based_decision=False)` | Phase-1 (FOUND-08); convention CONFIRMED to match sklearn this session [VERIFIED] |
| `mlrs-core::{oracle,compare,tolerance}` | in-tree | npz fixture load, `assert_close` 1e-5 abs+rel | Phase-1 (FOUND-07/08) [VERIFIED: mlrs-core/src/lib.rs] |
| `thiserror` | workspace | estimator + new-primitive error variants | Project convention (libs use thiserror) [VERIFIED: memory] |

### Supporting (test-side, /tmp venv only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | latest in /tmp venv | fixture generation reference for the new Cholesky primitive | `scripts/gen_oracle.py` regen [CITED: memory oracle-fixture-regen-needs-venv] |
| scikit-learn | >=1.6 | estimator-specific oracle fixtures (D-07) — NEW this phase (Phase 3 used numpy only) | `gen_oracle.py` estimator generators [CITED: 04-CONTEXT.md D-07] |
| scipy | latest in /tmp venv | `scipy.linalg.cholesky` / `solve_triangular` reference for the Cholesky-primitive fixture | new-primitive standalone validation [ASSUMED] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written Cholesky `#[cube]` | A cubek/cubecl linalg solve crate | None exists for cubecl 0.10 (cubecl-linalg abandoned per memory `cubecl-algo-crates-moved-to-cubek`); hand-write following the jacobi_eig blueprint |
| SVD pseudo-inverse for LinearRegression | Cholesky normal equations | LINEAR-01 PINS SVD ("match sklearn's default lstsq"); Cholesky is less faithful to gelsd + worse on rank-deficient X (D-02) |
| eig-of-covariance for PCA | thin SVD of centered X | D-01 chose SVD: more sklearn-faithful, more robust near rank-deficiency |

**Installation:** No `cargo add`. Test-time Python deps only:
```bash
python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn
```

**Version verification:** `cubecl 0.10.0`, `cubek-matmul 0.2.0`, `cubek-std 0.2.0` confirmed in the
checked-in Cargo.toml [VERIFIED: Cargo.toml + crates/mlrs-backend/Cargo.toml]. No new registry packages.

## Package Legitimacy Audit

**Not applicable — Phase 4 installs no new external packages.** All compute crates (`cubecl`,
`cubek-matmul`, `cubek-std`) are already pinned and were legitimacy-audited in Phase 2 (the
cubek-matmul Task-1 checkpoint). The new Cholesky/triangular-solve code is first-party hand-written
Rust + `#[cube]`, not a dependency. The test-side Python tools (numpy/scipy/scikit-learn) install
into a throwaway `/tmp` venv used only to regenerate committed fixture blobs — they are never part
of the shipped artifact and are standard, ubiquitous PyPI packages.

`slopcheck` IS available at `/home/user/.local/bin/slopcheck` should the planner add any package;
none is expected.

## Architecture Patterns

### System Architecture Diagram

```
                       ┌──────────────────────────────────────────────┐
   Arrow / host data → │  mlrs-algos  (estimator host orchestration)  │
                       │   Fit / Predict / Transform traits (D-04)    │
                       └───────────────┬──────────────────────────────┘
                                       │  composes (host calls)
        ┌──────────────────────────────┼───────────────────────────────────────┐
        ▼                              ▼                                         ▼
  ┌───────────────┐         ┌────────────────────┐                  ┌─────────────────────────┐
  │ column-mean   │         │  thin SVD (P3)     │                  │  covariance/Gram (P2)   │
  │ reduce (P2)   │         │  prims::svd        │                  │  prims::covariance      │
  │ centering D-05│         │  U,S,Vᵀ (k=min)    │                  │  XᵀX  (Ridge normal)    │
  └──────┬────────┘         └─────────┬──────────┘                  └──────────┬──────────────┘
         │                            │                                        │  + αI (diag)
         │           ┌────────────────┼────────────────┐                       ▼
         │           ▼                ▼                ▼          ┌────────────────────────────┐
         │   ┌──────────────┐ ┌──────────────┐ ┌─────────────┐   │  NEW Cholesky + tri-solve  │
         │   │ PCA (D-01)   │ │ LinReg(D-02) │ │ TSVD (D-07) │   │  prims::cholesky / solve   │
         │   │ center→SVD   │ │ V·σ⁺·Uᵀ·y    │ │ SVD uncent. │   │  #[cube] kernel (jacobi    │
         │   │ S²/(n-1)     │ │ +cutoff      │ │ var(transf.)│   │  single-cube blueprint)    │
         │   └──────┬───────┘ └──────┬───────┘ └──────┬──────┘   └──────────┬─────────────────┘
         │          │                │                │                     │ coef
         │          ▼                ▼                ▼                     ▼
         │   align_rows (svd_flip u_based_decision=False) ── mlrs-core::sign_flip      Ridge coef
         │          │                │                │                     │
         └──────────┴────────────────┴────────────────┴─────────────────────┘
                                       │  fitted state stays DEVICE-RESIDENT (D-03)
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │  DeviceArray fitted attrs (pool-managed)     │
                       │  predict/transform device-side; lazy host    │
                       │  materialize only at accessor / oracle time  │
                       └───────────────┬──────────────────────────────┘
                                       ▼
                         align_rows + assert_close 1e-5  vs  sklearn npz fixtures (D-07)
```

### Recommended Project Structure (Claude's Discretion D-04 — one suggested layout)
```
crates/mlrs-algos/src/
├── lib.rs                  # re-exports; trait module
├── traits.rs               # Fit / Predict / Transform (D-04)
├── linear/
│   ├── mod.rs
│   ├── linear_regression.rs  # SVD pseudo-inverse (LINEAR-01, D-02)
│   └── ridge.rs              # Cholesky normal eq (LINEAR-02, D-02)
└── decomposition/
    ├── mod.rs
    ├── pca.rs                # center→SVD (DECOMP-01, D-01)
    └── truncated_svd.rs      # uncentered SVD (DECOMP-02, D-07)
crates/mlrs-algos/tests/
├── linear_regression_test.rs
├── ridge_test.rs
├── pca_test.rs
├── truncated_svd_test.rs
└── memory_gate_test.rs       # OR extend mlrs-backend/tests/memory_gate_test.rs (D-03)
crates/mlrs-kernels/src/
└── cholesky.rs               # NEW feature-free #[cube] (D-02)
crates/mlrs-backend/src/prims/
├── cholesky.rs               # NEW launch wrapper + tri-solve host orchestration (D-02)
crates/mlrs-backend/tests/
└── cholesky_test.rs          # NEW standalone primitive validation (f32+f64, cpu+rocm)
```

### Pattern 1: Three-consumer thin-SVD reuse (PCA / LinearRegression / TruncatedSVD)
**What:** All three SVD-backed estimators call the SAME `prims::svd::svd(pool, a, (rows, cols))`,
returning device-resident `(U[m×k], S[k], Vᵀ[k×n])`, k=min(m,n), S descending (D-08).
**When to use:** any estimator needing eigenstructure of X.
**Differences per estimator:**
- **PCA:** feed CENTERED X (subtract column means first, store `mean_`). `components_ = first
  n_components rows of Vᵀ` (after flip); `transform(X) = (X−mean)·V` = `(X−mean)·Vᵀᵀ`;
  `explained_variance_ = S²/(n−1)`; `inverse_transform(Z) = Z·components_ + mean`.
- **LinearRegression:** feed CENTERED X (D-05), solve `coef = V·diag(σ⁺)·Uᵀ·y_centered`, σ⁺ with cutoff.
- **TruncatedSVD:** feed UNCENTERED X. `components_ = first k rows of Vᵀ`; `transform(X) = X·components_ᵀ`;
  `explained_variance_ = var(transform(X) columns)` (NOT `S²/(n−1)`).
```rust
// Source: prims/svd.rs:90 (VERIFIED signature)
let (u, s, vt) = svd::<F>(pool, &x_dev, (n_samples, n_features))?;
// estimator-specific arithmetic + align_rows on the host
```

### Pattern 2: New Cholesky/triangular-solve kernel — clone the jacobi_eig single-cube blueprint
**What:** Factor SPD `A = (XᵀX + αI)` (`n×n`, n = n_features ≤ MAX_DIM=64) as `A = L·Lᵀ`, then
solve `L·z = b`, `Lᵀ·x = z` (`b = Xᵀy`). For multi-target y, b has multiple columns.
**Why this is bounded risk:** `jacobi_eig.rs` already proves a single-cube, all-in-shared-memory,
in-kernel pattern for an `n×n` matrix with n ≤ 64 — a Cholesky tile fits the SAME LDS budget
(64×64 f32 L = 16 KiB, well within gfx1100's 64 KiB; f64 64×64 = 32 KiB still fits). [VERIFIED: jacobi_eig.rs:92-97 LDS budget commentary]
**Blueprint to copy:**
```rust
// Source: jacobi_eig.rs:119-155 (VERIFIED pattern)
#[cube(launch)]
pub fn cholesky_solve<F: Float + CubeElement>(
    a_in: &Array<F>,      // row-major n×n SPD (XᵀX + αI), symmetry TRUSTED
    b_in: &Array<F>,      // n×rhs  (Xᵀy)
    x_out: &mut Array<F>, // n×rhs solution
    info_out: &mut Array<F>, // [0] = non-SPD flag (negative pivot) → host PrimError
    n: u32, rhs: u32,
) {
    let mut l_sh = SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize); // L factor
    // 1. Cholesky-Banachiewicz (row by row): for i in 0..n, for j in 0..=i:
    //      L[i][j] = (A[i][j] - Σ_{k<j} L[i][k]L[j][k]) / L[j][j]   (j<i)
    //      L[i][i] = sqrt(A[i][i] - Σ_{k<i} L[i][k]²)   ← guard arg ≤ 0 → info flag
    //    (one acting unit per row OR unit-0-does-all like the eig "acting unit" idiom)
    // 2. Forward solve  L·z = b   (z = b; for i: z[i] = (b[i]-Σ_{k<i}L[i][k]z[k])/L[i][i])
    // 3. Back solve     Lᵀ·x = z  (for i desc: x[i] = (z[i]-Σ_{k>i}L[k][i]x[k])/L[i][i])
    // sync_cube() between the three phases; `continue` is unsupported in #[cube] → use `if`.
}
```
**Host wrapper (`prims::cholesky.rs`):** validate `a.len()==n*n` and SPD-shape BEFORE launch
(`PrimError::NotSquare`), thread the Gram `out` buffer through (memory gate — reuse, no parallel
n² alloc, exactly like `eig()`), return `PrimError::NotConverged`/a new `NotPositiveDefinite` on a
negative pivot. Keep the solve loop in-kernel (single cube → no host round-trip, D-11 gate 3).

### Pattern 3: Device-resident fitted state with lazy host materialize (D-03)
**What:** `fit` stores `coef_`, `components_`, etc. as `DeviceArray<ActiveRuntime, F>` fields.
`predict`/`transform` compose `gemm`/elementwise on-device. Host `Vec<F>` only on an explicit
accessor or at oracle-compare time (`to_host`/`to_host_metered`).
**Memory gate:** extend the build-failing `memory_gate_test.rs` — a `fit` + repeated same-shape
`transform`/`predict` round must (1) keep read_backs minimal/0 mid-pipeline, (2) drive pool reuse
(bounded allocations across iterations), (3) reuse the Gram/GEMM out buffer for the Cholesky factor.
[VERIFIED: memory_gate_test.rs gate structure]

### Anti-Patterns to Avoid
- **Unifying LinearRegression and Ridge into one solver** — LINEAR-01 pins SVD, LINEAR-02 pins
  Cholesky. They are deliberately different (D-02 / Specifics).
- **Making the SVD/Cholesky primitive apply `svd_flip` or center** — primitives stay RAW; the
  ESTIMATOR centers and flips (D-01/D-03). Mirrors Phase-3 D-03.
- **Penalizing the intercept in Ridge** — center X,y → solve on centered → recover intercept; α
  applies only to `coef_`, never the intercept (D-05; sklearn-exact). [VERIFIED: sklearn _ridge.py]
- **Hardcoding a plane width / 32** in the Cholesky kernel — use the shared-memory tree idiom (like
  jacobi_eig's off-diagonal reduce), not a plane path (carried no-hardcoded-plane-width rule).
- **Host round-trip between Cholesky phases** — keep factor + both triangular solves in ONE launch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Eigenstructure of X | Bespoke SVD/eig | `prims::svd::svd` (Phase 3) | Validated, descending, thin, device-resident (D-08) |
| XᵀX normal matrix | A new Gram kernel | `prims::covariance` (reuse out buffer) | Phase-2 validated; memory-gate reuse already proven (D-02) |
| A·V, Uᵀ·y, X·coef, X·componentsᵀ | A new matmul | `prims::gemm` with transpose flags | Phase-2 cubek-matmul wrap (D-08 "Don't Hand-Roll") |
| Column means (centering) | A new reduction | `column_reduce(.., ScalarOp::Mean, ReducePath::Shared)` | Phase-2 validated; shared path never plane-gated to None |
| Column L2-norm (TSVD/SVD-U) | A new norm | `column_reduce(.., ScalarOp::L2Norm, ..)` | Phase-2 validated |
| svd_flip sign canonicalization | New sign logic | `mlrs-core::sign_flip::align_rows` | CONFIRMED == sklearn `svd_flip(u_based_decision=False)` this session |
| npz fixture load + 1e-5 compare | New harness | `mlrs-core::{oracle, compare}` | Phase-1 FOUND-07/08 |
| **General linear solve** | Generic LU / pivoted solver | **A targeted SPD Cholesky for Ridge ONLY** | SPD structure is guaranteed (`XᵀX + αI`, α>0); no pivoting needed — simpler + faster than general LU (D-02 / sklearn `assume_a="pos"`) |

**Key insight:** Three of four estimators are pure composition of existing validated primitives +
host arithmetic — the ONLY genuinely new device code is the Ridge Cholesky kernel, and even that has
a proven structural blueprint in `jacobi_eig.rs`. Resist building a general-purpose linear-algebra
layer; build exactly the one SPD solve Ridge needs.

## Common Pitfalls

### Pitfall 1: LinearRegression small-singular-value cutoff mismatch (rank-deficient / collinear X)
**What goes wrong:** Computing `coef = V·diag(1/σ)·Uᵀ·y` with `1/σ` for ALL σ (including ~0
singular values from collinear columns) explodes the coefficients and breaks 1e-5 on rank-deficient X.
**Why it happens:** sklearn uses `scipy.linalg.lstsq` (gelsd), which TRUNCATES the pseudo-inverse:
singular values `σ < cond·σ_max` are treated as zero (`σ⁺ = 0`). [VERIFIED: scipy.linalg.lstsq cond semantics]
**How to avoid:** apply the cutoff `σ⁺_i = if σ_i > cond·σ_max { 1/σ_i } else { 0 }`. The scipy default
when `cond=None` (gelsd) is the LAPACK default ≈ `eps · max(m, n)` relative to σ_max
(f32 eps≈1.19e-7, f64 eps≈2.22e-16). [ASSUMED — exact gelsd default not pinned in scipy docs; A1].
Pick the cutoff that holds 1e-5 across the oracle sweep (Claude's Discretion). Reuse the existing
`NEAR_ZERO_FLOOR = 1e-8` precedent as a FALLBACK floor if the relative cutoff under-truncates.
**Warning signs:** coefficients orders of magnitude larger than sklearn's; test passes on
full-rank random X but fails when you add a duplicated/collinear column.

### Pitfall 2: PCA `explained_variance_` uses ddof=1, TruncatedSVD does NOT use S²/(n−1)
**What goes wrong:** Copying PCA's `S²/(n−1)` formula into TruncatedSVD gives wrong `explained_variance_`.
**Why it happens:** sklearn PCA `explained_variance_ = S²/(n_samples−1)` [VERIFIED: _pca.py], but
TruncatedSVD computes `explained_variance_ = var(X_transformed, axis=0)` i.e. the empirical variance
of each transformed column (and `explained_variance_ratio_ = explained_variance_ / total var of X`).
[VERIFIED: _truncated_svd.py].
**How to avoid:** keep the two formulas distinct. PCA: `S²/(n−1)`. TSVD: variance of `transform(X)`
columns; `explained_variance_ratio_` denominator = sum of per-feature variances of the ORIGINAL X.
**Warning signs:** TSVD `explained_variance_` off by a constant factor or by the centering.

### Pitfall 3: svd_flip tie-break / wrong decision basis
**What goes wrong:** sign vectors flip the wrong way vs sklearn, failing `assert_close` even though
the subspace is correct.
**Why it happens:** sklearn PCA AND TruncatedSVD both use `svd_flip(u_based_decision=False)` —
sign taken from the **largest-abs element of each ROW of Vᵀ** (`sign(Vt[i, argmax|Vt[i]|])`), making
that element positive. [VERIFIED: extmath.py]. numpy `argmax` and `mlrs-core::canonical_sign` both
break ties at the LOWEST index — they MATCH. [VERIFIED: sign_flip.rs:21]
**How to avoid:** apply `align_rows` to BOTH the device `components_`/Vᵀ AND the sklearn fixture's
Vᵀ before comparison (the fixture is already flipped by sklearn, but re-aligning both is idempotent
and robust to any residual sign ambiguity). Apply to `transform`/U columns consistently via the
SAME signs. Do NOT use `u_based_decision=True` semantics (that's the default for some other sklearn paths).
**Warning signs:** components match in magnitude but flip sign on specific rows; transform output
sign-flipped relative to fixture.

### Pitfall 4: Cholesky on a non-SPD or near-singular normal matrix (small α, collinear X)
**What goes wrong:** `XᵀX + αI` with tiny α and highly collinear X is near-singular; a naive
Cholesky hits a negative/zero pivot under the square root and produces NaN.
**Why it happens:** f32 cancellation in `A[i][i] − Σ L[i][k]²` can go slightly negative even for a
mathematically SPD matrix.
**How to avoid:** guard the diagonal sqrt argument; if it goes `≤ near-zero floor`, set the
`info_out` non-SPD flag and surface a new `PrimError::NotPositiveDefinite` on the host rather than
emitting NaN (mirrors the SVD `NotConverged` discipline). Choose test α values away from the
degenerate edge; document the floor. sklearn's cholesky is "less stable for singular matrices" —
matching it means the oracle uses well-conditioned X for the strict 1e-5 cases. [VERIFIED: sklearn docs]
**Warning signs:** NaN `coef_`; test passes for α≥1 but NaNs for α=1e-8 on collinear X.

### Pitfall 5: Intercept recovery order and centering both X and y
**What goes wrong:** `intercept_` wrong because X was centered but y wasn't, or coef was solved on
raw X.
**Why it happens:** D-05 / sklearn require centering BOTH X (by column means x̄) AND y (by mean ȳ),
solving for coef on centered data, THEN `intercept_ = ȳ − x̄·coef_`. [VERIFIED: sklearn _preprocess_data].
**How to avoid:** with `fit_intercept=true`: compute x̄ (column means) and ȳ; center; solve; recover.
With `fit_intercept=false`: solve on raw X, `intercept_ = 0`. Same procedure for LinearRegression and Ridge.
**Warning signs:** coef matches but intercept is off by `x̄·coef`.

### Pitfall 6: `n_components` truncation of a thin-SVD that returns k=min(m,n)
**What goes wrong:** `prims::svd` returns ALL k=min(m,n) components; the estimator must TRUNCATE to
`n_components` rows of Vᵀ / S, but truncating in the wrong place (before vs after svd_flip, or
truncating U columns inconsistently) corrupts `explained_variance_ratio_` (whose denominator is the
TOTAL variance over ALL components, not just the kept ones).
**How to avoid:** compute total variance from ALL S (the full `S²/(n−1)` sum for PCA, or full
per-feature variance for TSVD) BEFORE truncating, then keep the top `n_components`. Validate
`n_components ≤ min(n_samples, n_features)` at construction/fit (new error variant). [VERIFIED: D-06]
**Warning signs:** `explained_variance_ratio_` sums to >1 or doesn't match sklearn's denominator.

## Runtime State Inventory

Not applicable — Phase 4 is greenfield estimator code in an empty crate, not a rename/refactor/migration.
The only "state" is committed fixture blobs (regenerated, not migrated) and device buffers (transient).

## Code Examples

### Verified existing primitive signatures (the building blocks)
```rust
// Source: prims/svd.rs:90 [VERIFIED]
pub fn svd<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (rows, cols): (usize, usize),
) -> Result<(DeviceArray<..,F>, DeviceArray<..,F>, DeviceArray<..,F>), PrimError>; // (U[m×k], S[k], Vᵀ[k×n])

// Source: prims/covariance.rs:89 [VERIFIED] — pass out=Some(gram_buf) to reuse (memory gate)
pub fn covariance<F>(pool, a, (n_samples, n_features), ddof: u32,
                     out: Option<DeviceArray<..,F>>) -> Result<DeviceArray<..,F>, PrimError>;
// NOTE: covariance centers + scales by 1/(n-ddof). Ridge wants the RAW Gram XᵀX, NOT the scaled
// covariance — either call gemm(transa=true) directly for XᵀX, or covariance with ddof and
// un-scale. Plan must pick: cleanest is gemm(pool, X,(m,n), X,(m,n), transa=true, transb=false) → XᵀX.

// Source: prims/gemm.rs:54 [VERIFIED] — transpose flags are zero-copy logical (D-06)
pub fn gemm<F>(pool, a, (m,k), b, (k2,n), transa: bool, transb: bool,
               out: Option<DeviceArray<..,F>>) -> Result<DeviceArray<..,F>, PrimError>;

// Source: prims/reduce.rs:224 [VERIFIED] — centering means
pub fn column_reduce<F>(pool, a, rows, cols, op: ScalarOp, path: ReducePath)
    -> Result<Option<DeviceArray<..,F>>, PrimError>; // ScalarOp::Mean for x̄

// Source: mlrs-core/src/sign_flip.rs:60 [VERIFIED] — host, operates on Vec<Vec<f64>> rows
pub fn align_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>>; // == sklearn svd_flip(u_based_decision=False)
```

### sklearn reference arithmetic (the oracle contract — verified against source)
```python
# PCA svd_solver='full'  [VERIFIED: sklearn/decomposition/_pca.py _fit_full]
mean_ = X.mean(axis=0)
Xc = X - mean_
U, S, Vt = scipy.linalg.svd(Xc, full_matrices=False)
U, Vt = svd_flip(U, Vt, u_based_decision=False)
components_ = Vt[:n_components]
explained_variance_ = (S**2) / (n_samples - 1)
explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()  # sum over ALL S
singular_values_ = S[:n_components]
# transform: (X - mean_) @ components_.T ;  inverse: Z @ components_ + mean_

# Ridge solver='cholesky'  [VERIFIED: sklearn/linear_model/_ridge.py _solve_cholesky]
# (after centering X,y when fit_intercept)
A  = X.T @ X
A.flat[::n_features+1] += alpha           # add alpha to diagonal (NOT to intercept)
Xy = X.T @ y
coef = scipy.linalg.solve(A, Xy, assume_a="pos")   # Cholesky-backed SPD solve
# intercept_ = y_mean - X_mean @ coef

# LinearRegression  [VERIFIED: uses scipy.linalg.lstsq / gelsd]
# coef, ... = scipy.linalg.lstsq(Xc, yc)   # SVD pseudo-inverse with cond cutoff
# equiv: coef = V @ diag(sigma_plus) @ U.T @ yc,  sigma_plus = 1/s where s > cond*s_max else 0

# TruncatedSVD algorithm='arpack'  [VERIFIED: sklearn/decomposition/_truncated_svd.py]
# NO centering. U, S, VT = svds(X, k=n_components, v0=...) -> reversed -> svd_flip(u_based_decision=False)
components_ = VT
X_transformed = X @ components_.T          # = U * S
explained_variance_ = X_transformed.var(axis=0)         # variance of transformed columns
explained_variance_ratio_ = explained_variance_ / X.var(axis=0).sum()  # / total feature variance
singular_values_ = S
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PCA via eig-of-covariance (Phase-3 D-01/D-06 anticipated) | PCA via SVD-of-centered-X | Phase-4 D-01 | eig primitive validated but UNUSED in v1; SVD is the workhorse |
| numpy oracle fixtures (Phase 3) | scikit-learn fixtures for estimator attrs (D-07) | Phase-4 | /tmp venv now needs scikit-learn, not just numpy |
| cpu+wgpu gate (ROADMAP/PROJECT wording) | cpu(f64)+rocm(f32) gate (D-07) | Phase-3 03-01 | Every f64 test uses `skip_f64_with_log`; wgpu opportunistic |

**Deprecated/outdated:**
- `cubecl-matmul` / `cubecl-linalg` — abandoned on cubecl 0.9/0.5; do NOT reach for a cubecl linalg
  solve crate for Cholesky. Hand-write the `#[cube]` kernel. [VERIFIED: memory `cubecl-algo-crates-moved-to-cubek`]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy `lstsq` `cond=None` (gelsd) default cutoff ≈ `eps·max(m,n)` relative to σ_max | Pitfall 1 / LINEAR-01 | If the exact constant differs, LinearRegression may fail 1e-5 on near-rank-deficient X. Mitigation: cutoff is Claude's Discretion (D-02) — tune empirically against the oracle sweep; full-rank random X (the common case) is insensitive to the exact value. |
| A2 | scipy/scikit-learn install cleanly into a /tmp venv for fixture regen | Standard Stack | Low — only affects fixture regeneration, not the shipped artifact; fixtures are committed blobs. |
| A3 | A 64×64 f64 Cholesky tile (32 KiB shared) fits gfx1100 LDS alongside the b/x scratch | Pattern 2 | Low — jacobi_eig already runs a 32 KiB (A+V) shared layout on gfx1100; Cholesky needs only L (16/32 KiB) + small b/x. If f64 overflows, keep L in global like jacobi_svd did (precedent exists). |

## Open Questions (RESOLVED)

1. **Raw Gram vs scaled covariance for Ridge.**
   - What we know: `prims::covariance` centers + scales by `1/(n−ddof)`. Ridge needs RAW `XᵀX`.
   - What's unclear: whether to call `gemm(transa=true)` directly for `XᵀX` (cleanest) or reuse
     covariance and un-scale.
   - Recommendation: use `gemm(pool, X,(m,n), X,(m,n), transa=true, transb=false)` for the raw Gram
     (D-02 says "reuses the covariance/Gram primitive" — `covariance` internally IS gemm(transa);
     calling gemm directly avoids the scale/center the normal equations don't want). Planner decides;
     either way thread the out buffer through for the memory gate.

2. **Cholesky kernel acting-unit schedule (Claude's Discretion D-02).**
   - What we know: jacobi_eig serializes pairs with an "acting unit does the whole rotation"; the
     CPU backend serializes a cube's units anyway.
   - What's unclear: whether a per-row-parallel Cholesky (unit i computes L row i, syncing after each
     column) is worth it vs unit-0-does-everything for n ≤ 64.
   - Recommendation: start with the simplest correct version (unit-0 sequential, like the eig "acting
     unit"), since n ≤ 64 makes serialization cheap and correctness-first is the project value;
     optimize only if a memory/perf gate demands it.

3. **Exact LinearRegression cutoff constant (A1).**
   - What we know: scipy default is the gelsd LAPACK default (~eps·max(m,n)).
   - What's unclear: the precise multiplier that holds 1e-5 across the chosen oracle shapes.
   - Recommendation: tune against the committed sklearn `LinearRegression` fixtures; include at least
     one near-collinear case so the cutoff is actually exercised (not just full-rank random).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain + `cargo test` | all build/test | ✓ | workspace | — |
| cpu backend (cubecl) | f64 correctness gate | ✓ | cubecl 0.10 | — |
| rocm/HIP backend (gfx1100) | f32 correctness gate | ✓ | ROCm 7.1.1 | wgpu opportunistic |
| python3 + venv (numpy/scipy/scikit-learn) | fixture regen ONLY | ✓ (via /tmp venv, PEP 668) | latest | committed fixture blobs already in `tests/fixtures/` |
| slopcheck | (not needed — no new pkgs) | ✓ | `/home/user/.local/bin/slopcheck` | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** f64-on-rocm is structurally unavailable (cubecl-cpp 0.10 does
not register F64 for HIP) — fallback is `skip_f64_with_log` (f64 validates on cpu instead). This is
the designed D-07 gate, not a gap.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (no external runner); tests in `crates/*/tests/` (AGENTS.md §2) |
| Config file | none — `cargo test` per crate with backend feature flags |
| Quick run command | `cargo test -p mlrs-algos --features cpu` |
| Full suite command | `cargo test -p mlrs-algos --features cpu && cargo test -p mlrs-algos --features rocm && cargo test -p mlrs-backend --features cpu --test cholesky_test && cargo test -p mlrs-backend --features rocm --test cholesky_test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| (new prim) | Cholesky `‖L·Lᵀ−A‖`, `‖A·x−b‖`, NotPositiveDefinite guard | unit (invariant + oracle) | `cargo test -p mlrs-backend --features cpu --test cholesky_test` | ❌ Wave 0 |
| LINEAR-01 | LinearRegression coef/intercept/predict vs sklearn 1e-5 (incl. collinear cutoff) | oracle | `cargo test -p mlrs-algos --features cpu --test linear_regression_test` | ❌ Wave 0 |
| LINEAR-02 | Ridge coef/intercept vs sklearn 1e-5 (alpha sweep, intercept not penalized) | oracle | `cargo test -p mlrs-algos --features cpu --test ridge_test` | ❌ Wave 0 |
| DECOMP-01 | PCA all attrs + transform/inverse_transform vs sklearn after align_rows | oracle | `cargo test -p mlrs-algos --features cpu --test pca_test` | ❌ Wave 0 |
| DECOMP-02 | TruncatedSVD attrs + transform vs sklearn arpack after align_rows | oracle | `cargo test -p mlrs-algos --features cpu --test truncated_svd_test` | ❌ Wave 0 |
| D-03 | fit→predict/transform memory gate (reuse, read_backs, Gram/factor reuse) | gate | `cargo test --features cpu --test memory_gate_test` | ⚠ extend existing |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate> --features cpu --test <relevant_test>`
- **Per wave merge:** `cargo test -p mlrs-algos --features cpu` (+ `--features rocm` for f32 gate)
- **Phase gate:** full suite green on cpu(f64)+rocm(f32) before `/gsd-verify-work`; f64 tests
  skip-with-log on rocm (mirror `gemm_test.rs`).

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/cholesky_test.rs` — covers the new primitive (invariants + sklearn/scipy fixture)
- [ ] `crates/mlrs-algos/tests/linear_regression_test.rs` — covers LINEAR-01
- [ ] `crates/mlrs-algos/tests/ridge_test.rs` — covers LINEAR-02
- [ ] `crates/mlrs-algos/tests/pca_test.rs` — covers DECOMP-01
- [ ] `crates/mlrs-algos/tests/truncated_svd_test.rs` — covers DECOMP-02
- [ ] Memory gate extension for fit→predict/transform (new file or extend `mlrs-backend/tests/memory_gate_test.rs`)
- [ ] `scripts/gen_oracle.py` new generators: `gen_cholesky`, `gen_linear_regression`, `gen_ridge`,
      `gen_pca`, `gen_truncated_svd` (sklearn arpack) → committed `.npz` fixtures
- [ ] `mlrs-algos/Cargo.toml` — add `mlrs-backend`, `mlrs-core`, `cubecl` dev/deps (currently only `thiserror`)

## Security Domain

`security_enforcement: true`, ASVS level 1. This is a numerical-library phase with no auth, network,
session, or user-facing surface; the relevant ASVS category is **V5 (input validation)** applied to
matrix geometry and hyperparameters before any `unsafe` device launch (the established Phase-2/3 pattern).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate geometry + hyperparams BEFORE any `unsafe` launch: `a.len()==rows*cols`, SPD squareness `n*n` (`PrimError::NotSquare`), `n_components ≤ min(m,n)`, `n ≤ MAX_DIM`, `alpha ≥ 0`, `n_samples > 0` — `Result`-return before launch (mirrors `svd()`/`eig()`/`covariance()` `validate_geometry`). |
| V6 Cryptography | no | — (no crypto; never hand-roll, but N/A here) |

### Known Threat Patterns for Rust/CubeCL numerical kernels
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from bad (rows,cols) vs buffer len | Tampering / DoS | `validate_geometry` returns `PrimError::ShapeMismatch` BEFORE `ArrayArg::from_raw_parts` (existing pattern) |
| `unsafe { ArrayArg::from_raw_parts }` mis-sized | Tampering | Pass the validated element count; kernel bounds-checks `tid < len` (existing center_columns/gemm precedent) |
| Cholesky negative-pivot → NaN propagation | DoS (silent-wrong) | Diagonal sqrt-arg guard → `info_out` flag → `PrimError::NotPositiveDefinite` (no silent NaN) |
| Division by ~0 singular value (LinearRegression) / `n_samples−ddof ≤ 0` | DoS (inf/NaN) | cutoff `σ⁺=0` below threshold; reject `n−ddof ≤ 0` at boundary (covariance precedent WR-01) |
| f64 launch on a backend without F64 (rocm/HIP) | DoS (launch reject) | `skip_f64_with_log` capability gate (D-07) — skip, never crash |

## Sources

### Primary (HIGH confidence)
- `crates/mlrs-backend/src/prims/{svd,covariance,gemm,reduce}.rs` — verified primitive signatures + behavior
- `crates/mlrs-kernels/src/jacobi_eig.rs` — single-cube in-kernel blueprint for the new Cholesky kernel (LDS budget, acting-unit idiom, sync_cube)
- `crates/mlrs-core/src/sign_flip.rs` — `align_rows` == sklearn `svd_flip(u_based_decision=False)` (confirmed by reading both)
- `crates/mlrs-backend/tests/memory_gate_test.rs` — the build-failing gate to extend (D-03)
- `.planning/STATE.md` (03-01 entries) — cpu(f64)+rocm(f32) gate, D-07 supersedes cpu+wgpu
- sklearn source via GitHub: `_pca.py` (_fit_full), `_ridge.py` (_solve_cholesky + _preprocess_data),
  `_truncated_svd.py` (arpack fit), `utils/extmath.py` (svd_flip u_based_decision=False)

### Secondary (MEDIUM confidence)
- scikit-learn docs (LinearRegression, Ridge, ridge_regression) — cond/cholesky solver semantics
- scipy.linalg.lstsq docs — gelsd default driver, cond cutoff semantics

### Tertiary (LOW confidence — flagged in Assumptions Log)
- gelsd exact `cond=None` default constant (A1) — inferred ≈ `eps·max(m,n)`; not pinned in scipy docs, tune empirically

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new crates; all building blocks verified in-tree against the checked-in Cargo.toml
- Estimator math: HIGH — verified against sklearn source (_pca/_ridge/_truncated_svd/extmath) this session
- Cholesky kernel feasibility: MEDIUM — unwritten, but a proven structural blueprint exists (jacobi_eig); LDS budget confirmed
- LinearRegression cutoff constant: MEDIUM-LOW — exact gelsd default not pinned (A1), but it's Claude's Discretion and tunable
- Pitfalls: HIGH — derived from verified sklearn arithmetic differences (ddof, svd_flip basis, intercept centering)

**Research date:** 2026-06-12
**Valid until:** 2026-07-12 (stable — sklearn closed-form semantics and the in-tree primitives are slow-moving; the cubecl 0.10 pin is fixed for the milestone)
