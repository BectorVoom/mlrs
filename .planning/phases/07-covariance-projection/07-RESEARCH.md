# Phase 7: Covariance & Projection - Research

**Researched:** 2026-06-14
**Domain:** Covariance estimation, incremental PCA / streaming SVD merge, random projection, host-side seeded RNG-matrix primitive, the `PartialFit<F>` streaming trait — all assembled on v1's validated covariance / SVD / eig prims (NO new device kernel).
**Confidence:** HIGH for the SVD-merge algorithm and covariance/projection math (verified against scikit-learn 1.7.1 source); HIGH for the reuse map (read directly from the v1 source files); MEDIUM for the exact f32-on-rocm tolerance band numbers (Claude's-discretion bands, pinned by precedent + planner-set trial counts).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Add a new `PartialFit<F>` trait next to `Fit`/`Predict`/`Transform` (same `<F: Float + CubeElement + Pod>` bound, same `pool`/`DeviceArray`/explicit `(rows,cols)` shape convention, device-resident state per D-03). It is the new cross-cutting PY-06 contract (also reused by Phase 10 MBSGD).
- **D-02:** `fit()` is **sklearn-faithful**: it resets fitted state, then iterates `partial_fit` over `gen_batches(n_samples, batch_size)`. This exercises the PRIM-07 multi-batch merge *inside* `fit`, not only via explicit user batching.
- **D-03:** `batch_size=None → 5·n_features` (sklearn IncrementalPCA default); `n_samples_seen_` accumulated across `partial_fit` calls and exposed.
- **D-04:** The `[v2-P1]` incremental-SVD merge algorithm (full Jacobi re-SVD of the stacked `[prev_singular·V; mean-correction; batch]` per batch vs a dedicated rank-update kernel) is **handed to the research spike** before planning. No strong user preference; default leaning is full-Jacobi-per-batch reusing v1 `svd` (zero new kernel), to be confirmed for f32-on-rocm stability by the spike.
- **D-05:** Compute `precision_` as a **symmetric pseudo-inverse via the v1 `eig` prim** (pinvh-equivalent: `V·diag(1/λ_i, with a near-zero floor)·Vᵀ`), matching sklearn's `linalg.pinvh`. Singular-safe — handles the EmpiricalCovariance MLE rank-deficient case (`n_samples ≤ n_features`) without raising. Do **not** use the v1 Cholesky prim for the inverse (SPD-only, fails on singular covariance). Reuse the v1 cutoff/near-zero-floor convention (cf. 04-03 σ⁺ RCOND pattern).
- **D-06:** `whiten` (IncrementalPCA) — whitened transform output (components scaled by `1/sqrt(explained_variance_)`); `inverse_transform` un-whitens.
- **D-07:** `assume_centered` (EmpiricalCovariance + LedoitWolf) — when true, skip mean subtraction and set `location_ = 0`; covariance computed about the origin.
- **D-08:** `store_precision` / `precision_` accessor (default `True`) — required by COV-01's `precision_` criterion regardless; gates whether D-05 runs.
- **D-09:** `batch_size` (IncrementalPCA) — explicit control for the D-02/D-03 `partial_fit` batching loop.
- **D-10:** Use **strict (tight) property-gate thresholds** — JL distortion checked close to the theoretical bound; matrix-distribution moments held to tight tolerances. User's explicit choice over looser flake-resistant bands.
- **D-11:** **Mitigate flakiness of strict bands** with deterministic seeding (fixed SplitMix64 seed → identical matrix across runs/backends per PRIM-06) and **averaging the distortion/moment statistics over many trials**, so tight thresholds stay reproducible across cpu/rocm. Researcher/planner pins the exact threshold numbers and trial count.
- **D-12:** RandomProjection correctness gate is the **structural property set** (JL distortion bound, matrix-distribution stats, seed-reproducibility, `transform == X·componentsᵀ` self-consistency), explicitly **NOT** a 1e-5 value oracle (mlrs SplitMix64 RNG ≠ NumPy MT19937). `johnson_lindenstrauss_min_dim` *is* value-matched to sklearn. SparseRandomProjection `components_` are stored **dense** (no sparse device kernels in v2; acceptable at v2 sizes).

### Claude's Discretion
- Exact f32-on-rocm tolerance bands for LedoitWolf / IncrementalPCA (components band + sign; explained_variance band) — follow the v1 per-family documented-band precedent (Recurring gates in ROADMAP Phase 7).
- Whether `EmpiricalCovariance`/`LedoitWolf` expose `error_norm`/`mahalanobis` helpers — only if cheap and within the COV-01/02 surface; otherwise defer.

### Deferred Ideas (OUT OF SCOPE)
- **Device RNG kernel** — only if the `[v2-P1]` spike shows host-generate-then-upload is a bottleneck (default: not needed in v2).
- **Dedicated incremental rank-update SVD kernel** — only if full-Jacobi-per-batch is unstable on f32/rocm per the spike.
- **Sparse device kernels for SparseRandomProjection** — out of v2 scope; sparse input densified at ingress, `components_` stored dense.
- None outside phase scope surfaced during discussion — stayed within Phase 7.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRIM-06 | Reproducible seeded RNG-matrix primitive (`prims/rng.rs`, host SplitMix64, no `OsRng`): Gaussian + Achlioptas-sparse projection matrices + permutations; distribution + seed-reproducibility validated. | §RNG-matrix primitive — SplitMix64 promotion + Box-Muller + Achlioptas + Fisher-Yates fully specified; PoolStats gate pattern from memory_gate_test.rs. |
| PRIM-07 | Incremental-SVD merge primitive over the v1 Jacobi `svd` (mean-correction row, `svd_flip(u_based_decision=False)`, ddof=1), serving IncrementalPCA. | §Incremental-SVD merge — exact sklearn 1.7.1 stacked-matrix construction + stability analysis + RECOMMENDATION (full-Jacobi-per-batch confirmed). |
| COV-01 | `EmpiricalCovariance`: `covariance_`/`location_`/`precision_` ≤ 1e-5 (MLE / ddof=0). | §EmpiricalCovariance — `np.cov(bias=1)` = covariance prim with `ddof=0`; pinvh via v1 eig (D-05). |
| COV-02 | `LedoitWolf`: shrinkage-regularized `covariance_` + `shrinkage_` (clipped [0,1]) ≤ 1e-5. | §LedoitWolf — exact `ledoit_wolf_shrinkage` β/δ/μ formula from sklearn 1.7.1; host finalize. |
| DECOMP-03 | `IncrementalPCA` via `partial_fit`: `components_`/`explained_variance_`/`explained_variance_ratio_`/`singular_values_`/`mean_`/`var_` + `transform`/`inverse_transform` ≤ 1e-5 after `svd_flip` sign alignment. | §Incremental-SVD merge + §PartialFit trait + §IncrementalPCA attrs. |
| PROJ-01 | `GaussianRandomProjection` (`n_components='auto'`) + `transform` — property-gated (JL bound, dist stats, seed-repro, self-consistency); `johnson_lindenstrauss_min_dim` value-matched. | §RandomProjection — JL formula value-matched; property-gate design (D-10/D-11). |
| PROJ-02 | `SparseRandomProjection` (Achlioptas, configurable `density`) + `transform`, property-gated; sparse input densified at ingress. | §RandomProjection — Achlioptas matrix exact scaling; dense `components_` storage (D-12). |
</phase_requirements>

## Summary

Phase 7 is the lowest-risk v2 opener: every estimator is **host-side orchestration over already-validated v1 prims** — covariance Gram, Jacobi SVD, symmetric eig, GEMM, column reductions — plus **two new host-side prims** (`rng.rs`, `incremental_svd.rs`) and **one new trait** (`PartialFit<F>`). No new device kernel is required, which sidesteps the cpu-MLIR SharedMemory/`F::INFINITY`/atomics landmines entirely (project memory: "both new Phase-7 prims are host-side glue precisely to dodge this").

The decisive `[v2-P1]` question — *full-Jacobi re-SVD per batch vs a dedicated rank-update kernel* — is **settled: full-Jacobi-per-batch, reusing the v1 `svd` primitive, ZERO new kernel.** This is exactly what scikit-learn's `IncrementalPCA.partial_fit` itself does (it calls dense `linalg.svd` on a stacked `(n_components + n_batch + 1) × n_features` matrix every batch). The merged operand is tiny (rows ≤ `n_components + batch_size + 1`, cols = `n_features`, both well under the v1 SVD `MAX_ROWS`/`MAX_COLS` caps at v2 sizes), so the v1 Jacobi cost is negligible and its f32 stability is the SAME stability the v1 PCA path already passes its 1e-5 gate with. f32-on-rocm gets a documented per-family tolerance band (components + sign; explained_variance), not a new algorithm. **A dedicated rank-update kernel is NOT needed and is explicitly deferred.**

The covariance and projection math is fully pinned against scikit-learn 1.7.1 source (verified this session): `empirical_covariance` is `np.cov(X.T, bias=1)` = the v1 covariance prim with `ddof=0`; `precision_` is `scipy.linalg.pinvh` = `V·diag(1/λ floored)·Vᵀ` over the v1 eig prim; `ledoit_wolf_shrinkage` has an exact closed β/δ/μ form; `johnson_lindenstrauss_min_dim = 4·ln(n)/(eps²/2 − eps³/3)`; Gaussian matrix is `N(0, 1/n_components)` and the Achlioptas matrix scales by `sqrt(1/density)/sqrt(n_components)`.

**Primary recommendation:** Build `incremental_svd.rs` as a thin host merge that stacks `[Σ·Vᵀ ; X_batch_centered ; mean_correction]` and re-runs the v1 `svd` per batch; build `rng.rs` by promoting the existing `SplitMix64` from `kmeans.rs` and adding Box-Muller Gaussian + Achlioptas sparse + Fisher-Yates shuffle; implement `EmpiricalCovariance`/`LedoitWolf`/`IncrementalPCA`/`Gaussian`+`SparseRandomProjection` as `mlrs-algos` estimators composing these. No new device kernel, no new dependency.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Seeded RNG-matrix generation (Gaussian/Achlioptas/permute) | Host (CPU, in `prims/rng.rs`) | Device (upload only) | Reproducibility across backends requires a host PRNG (ASVS V6, no `OsRng`); device RNG is backend-divergent (v1 Anti-Pattern). Generate on host → upload as a `DeviceArray`. [VERIFIED: kmeans.rs SplitMix64 precedent] |
| Incremental-SVD merge orchestration | Host (`prims/incremental_svd.rs`) | Device (v1 `svd` kernel does the diagonalization) | The merge is matrix-stacking + sign flip (host glue); the heavy SVD is the existing in-kernel Jacobi loop. [CITED: sklearn _incremental_pca.py] |
| Covariance Gram `AᵀA/(n−ddof)` | Device (v1 `covariance` prim) | Host (scale factor) | Already a validated device prim (GEMM(transa) + scale). [VERIFIED: covariance.rs] |
| `precision_` = pinvh | Device (v1 `eig` prim for the spectral decomp) | Host (1/λ floor + reassembly GEMM) | pinvh = `V·diag(1/λ)·Vᵀ`; eig is device, the floor+reassembly is small host/GEMM work. [VERIFIED: eig.rs + sklearn pinvh] |
| Ledoit-Wolf shrinkage coefficient | Host (closed-form β/δ/μ over the centered batch) | Device (covariance prim for `emp_cov`) | β/δ are scalar reductions over `X²` and `XᵀX`; cheap host finalize (mirrors kmeans inertia host-sum). [CITED: sklearn _shrunk_covariance.py] |
| `PartialFit` streaming state accumulation | Host (estimator struct holds device-resident running components) | Device (per-batch merge) | `n_samples_seen_` / running `mean_`/`var_`/`components_`/`singular_values_` are estimator state per D-01/D-03. |
| RandomProjection `transform` | Device (v1 `gemm`) | Host (densify sparse ingress) | `transform == X·componentsᵀ` is one GEMM (mirrors PCA transform). [VERIFIED: pca.rs transform] |

## Standard Stack

**No new crate dependency** (ROADMAP/REQUIREMENTS hard constraint: "v2 adds zero compute dependencies; no `cubek-random`, no pyo3 bump"). The "stack" is the existing v1 primitive + estimator surface.

### Core (reused v1, no new code)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `prims::covariance::covariance` | `mlrs-backend/src/prims/covariance.rs` | Centered `AᵀA/(n−ddof)` Gram; `ddof=0` for MLE covariance, supports caller `out` reuse | Already 1e-5-validated against `np.cov`; the exact COV-01 math (`np.cov(bias=1)` = `ddof=0`). [VERIFIED: covariance.rs + sklearn] |
| `prims::svd::svd` | `mlrs-backend/src/prims/svd.rs` | Thin SVD `U·diag(S)·Vᵀ`, descending S, tall+wide via Aᵀ-swap | The base PRIM-07 composes; caps `MAX_ROWS`/`MAX_COLS` (check at the merged-matrix shape). [VERIFIED: svd.rs] |
| `prims::eig::eig` | `mlrs-backend/src/prims/eig.rs` | Symmetric eig `V·diag(w)·Vᵀ`, descending w, `out`-reuse | The D-05 pinvh base; trusts symmetry (covariance Gram IS symmetric). [VERIFIED: eig.rs] |
| `prims::gemm::gemm` | `mlrs-backend/src/prims/gemm.rs` | GEMM with transa/transb flags, `out` reuse | `transform`, pinvh reassembly, mean-correction projection. [VERIFIED: gemm.rs] |
| `prims::reduce::column_reduce` | `mlrs-backend/src/prims/reduce.rs` | Column mean / L2-norm, `ReducePath::Shared` (always cpu-safe) | `mean_`, batch means; the Shared path never plane-gates to None. [VERIFIED: covariance.rs usage] |
| `SplitMix64` (host PRNG) | currently private in `mlrs-backend/src/prims/kmeans.rs` | Documented seeded PRNG, no `OsRng` | The exact PRNG to **promote** into `prims/rng.rs` (PRIM-06). [VERIFIED: kmeans.rs L658-710] |
| `mlrs_core::sign_flip::align_rows` | `mlrs-core/src/sign_flip.rs` | `svd_flip(u_based_decision=False)` — make largest-|v| element per Vᵀ row positive | EXACTLY sklearn's `u_based_decision=False` rule (largest-abs element of each `v` row). [VERIFIED: sign_flip.rs + sklearn extmath.svd_flip] |
| `traits::{Fit,Transform}` | `mlrs-algos/src/traits.rs` | Estimator surface to extend with `PartialFit<F>` | D-01 adds `PartialFit<F>` alongside; same bounds/shape convention. [VERIFIED: traits.rs] |
| `any_estimator!` macro | `mlrs-py/src/dispatch.rs` | Unfit/F32/F64 dtype dispatch + `py.detach` GIL release + `guard_f64()` | v2 adds zero binding infra; each estimator gets the enum + hand-written `#[pymethods]`. [VERIFIED: dispatch.rs] |

### Supporting (new code — all host-side glue)
| New file | Purpose | When to Use |
|----------|---------|-------------|
| `prims/rng.rs` | Promote `SplitMix64`; add `gaussian_matrix`, `sparse_achlioptas_matrix`, `permutation` (Fisher-Yates); PoolStats gate | PRIM-06; consumed by Gaussian/SparseRandomProjection and any future shuffle (Phase 10 MBSGD). |
| `prims/incremental_svd.rs` | Host merge: stack `[Σ·Vᵀ ; X_c ; mean_corr]`, run v1 `svd`, apply `align_rows`; PoolStats gate | PRIM-07; consumed by IncrementalPCA `partial_fit`. |
| `mlrs-algos/src/covariance/` (new module group) | `EmpiricalCovariance`, `LedoitWolf` | COV-01/02. Register in `lib.rs`. |
| `mlrs-algos/src/projection/` (new module group) | `GaussianRandomProjection`, `SparseRandomProjection` | PROJ-01/02. Register in `lib.rs`. |
| `mlrs-algos/src/decomposition/incremental_pca.rs` | `IncrementalPCA` (mirrors `pca.rs` skeleton + `PartialFit`) | DECOMP-03. |
| `traits.rs` `PartialFit<F>` | New trait | D-01. |
| `error.rs` new variants | `assume_centered`/`density`/`batch_size`/`eps` guards (struct-variant style) | Hyperparameter validation. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Full-Jacobi re-SVD per batch (PRIM-07) | Dedicated rank-1/rank-k SVD-update kernel (Brand 2006) | A new device kernel: more code, cpu-MLIR risk, and a new validation surface — **rejected** (D-04 default + this research). sklearn itself uses full dense SVD per batch; v2 sizes make it cheap. |
| pinvh via v1 `eig` (D-05) | Cholesky inverse | Cholesky is SPD-only; the MLE covariance is rank-deficient when `n_samples ≤ n_features` → Cholesky fails. eig-pinvh is singular-safe. **Locked by D-05.** |
| Host-generate-then-upload RNG matrix | Device RNG kernel | Device RNG is backend-divergent (breaks seed-reproducibility across cpu/rocm) and adds a kernel. Host generate + upload is the v1 idiom and is reproducible. **Locked by scope.** |
| Box-Muller Gaussian | Ziggurat / inverse-CDF | Box-Muller is exact, branch-light, trivially reproducible from two uniforms; ziggurat is faster but more code and table-dependent. At v2 matrix sizes speed is irrelevant. |

**Installation:** None — no new dependency. Workspace `Cargo.toml` unchanged (REQUIREMENTS hard constraint).

## Package Legitimacy Audit

**Not applicable — this phase installs NO external packages.** REQUIREMENTS explicitly forbids new compute dependencies ("v2 adds zero compute dependencies; no `cubek-random`, no pyo3 bump (stays 0.28)"). All work composes existing in-workspace crates (`mlrs-backend`, `mlrs-algos`, `mlrs-core`, `mlrs-kernels`, `mlrs-py`). The seeded RNG is the hand-rolled `SplitMix64` already in the tree (ASVS V6: never `OsRng`, never the `rand` crate). No registry check required.

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────────────────────┐
  Python (pyarrow)        │  mlrs-py  (any_estimator! Unfit/F32/F64)      │
  fit/partial_fit/        │  py.detach → global_pool().lock()             │
  transform ─────────────▶│  guard_f64() before F64 arm; densify sparse   │
                          └───────────────┬───────────────────────────────┘
                                          │ DeviceArray<F> (row-major flat + (rows,cols))
                          ┌───────────────▼───────────────────────────────┐
                          │  mlrs-algos estimators (host orchestration)    │
                          │                                                 │
   EmpiricalCovariance ──▶│ covariance(ddof=0) ─▶ [optional] pinvh(eig)    │
   LedoitWolf ───────────▶│ covariance(ddof=0) + host β/δ/μ shrink scalar  │
   IncrementalPCA ───────▶│ PartialFit loop over gen_batches:              │
       │                  │   incremental_svd.merge(prev, X_batch) per batch│
       │                  │     stack [Σ·Vᵀ ; X_c ; mean_corr] ─▶ v1 svd    │
       │                  │     align_rows (svd_flip u_based=False)         │
   Gaussian/SparseRP ────▶│ rng.gaussian|sparse_achlioptas ─▶ transform=GEMM│
                          └───────────────┬───────────────────────────────┘
                                          │ prim calls (validate → launch → device-resident)
                          ┌───────────────▼───────────────────────────────┐
                          │  mlrs-backend prims (device)                    │
                          │  covariance · svd(Jacobi) · eig(Jacobi) · gemm  │
                          │  · reduce  +  NEW: rng.rs (host) ·              │
                          │  incremental_svd.rs (host glue over svd)        │
                          └─────────────────────────────────────────────────┘
   gate = cpu(f64) + rocm(f32);  f64-on-rocm → skip_f64_with_log
```

### Recommended Project Structure
```
crates/mlrs-backend/src/prims/
├── rng.rs              # PRIM-06: SplitMix64 (promoted) + gaussian/sparse/permute + PoolStats gate
├── incremental_svd.rs  # PRIM-07: host merge over v1 svd + PoolStats gate
└── kmeans.rs           # SplitMix64 now `use`s prims::rng (no duplicate PRNG)

crates/mlrs-algos/src/
├── traits.rs                     # + PartialFit<F>
├── covariance/                   # NEW module group
│   ├── mod.rs
│   ├── empirical_covariance.rs   # COV-01
│   └── ledoit_wolf.rs            # COV-02
├── projection/                   # NEW module group
│   ├── mod.rs
│   ├── gaussian.rs               # PROJ-01
│   └── sparse.rs                 # PROJ-02
└── decomposition/
    └── incremental_pca.rs        # DECOMP-03 (mirrors pca.rs)

crates/mlrs-backend/tests/        # rng_test.rs, incremental_svd_test.rs, memory gates
crates/mlrs-algos/tests/          # empirical_covariance_test.rs, ledoit_wolf_test.rs,
                                  # incremental_pca_test.rs, random_projection_test.rs
scripts/gen_oracle.py             # + gen_empirical_covariance/ledoit_wolf/incremental_pca/jl_min_dim
```

### Pattern 1: Incremental-SVD merge (PRIM-07) — THE decisive pattern
**What:** Update a running thin decomposition `(components_ [k×n_features], singular_values_ [k], mean_ [n_features], var_ [n_features], n_samples_seen_)` with a new batch `X_batch [b×n_features]`, by re-running a **dense SVD on a small stacked matrix**.

**Exact sklearn 1.7.1 math** (verified from `sklearn/decomposition/_incremental_pca.py`):

```text
# 1. Update running per-feature mean/var/count (Chan-Golub-LeVeque, extmath._incremental_mean_and_var):
col_mean, col_var, n_total = _incremental_mean_and_var(
    X_batch, last_mean=mean_, last_variance=var_,
    last_sample_count=repeat(n_samples_seen_, n_features))

# 2. Center the batch by ITS OWN running col_mean:
X = X_batch - col_mean            # b × n_features

# 3a. FIRST batch (no prior components): X stays as the centered batch only.
# 3b. SUBSEQUENT batch: build the stacked matrix
mean_correction = sqrt( (n_samples_seen_ * b) / n_total ) * (mean_ - col_batch_mean)
X = vstack([
    singular_values_.reshape(-1,1) * components_,   # k × n_features  (= Σ·Vᵀ, the prior decomposition re-expanded)
    X,                                              # b × n_features  (centered batch)
    mean_correction,                                # 1 × n_features  (correction row)
])
# → stacked is (k + b + 1) × n_features   (FIRST batch: just b × n_features)

# 4. Dense thin SVD of the small stacked matrix:
U, S, Vt = svd(X, full_matrices=False)              # ← the v1 prims::svd::svd call
U, Vt = svd_flip(U, Vt, u_based_decision=False)     # ← mlrs_core::sign_flip::align_rows on Vt rows

# 5. Store the running state (keep top n_components):
explained_variance       = S**2 / (n_total - 1)               # ddof=1
explained_variance_ratio = S**2 / sum(col_var * n_total)
components_      = Vt[:n_components]
singular_values_ = S[:n_components]
mean_ = col_mean ; var_ = col_var ; n_samples_seen_ = n_total
```
[CITED: github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/decomposition/_incremental_pca.py] [CITED: .../sklearn/utils/extmath.py for _incremental_mean_and_var + svd_flip]

**Key mapping to v1 code:**
- `col_batch_mean` = `column_reduce(X_batch, Mean)` (the batch's own mean, before the running-mean update). `mean_` is the *prior running* mean.
- `singular_values_.reshape(-1,1) * components_` = per-row scale of `components_` (k rows) by `singular_values_` — a host elementwise multiply (k ≤ n_components is tiny).
- The stacked matrix is built on the host (the rows are heterogeneous), uploaded once, then fed to `prims::svd::svd`. This mirrors `pca.rs` which already centers on the host and uploads.
- `svd_flip(u_based_decision=False)` = `align_rows` on the `Vt` rows — **already implemented and used by `pca.rs`** (it picks the largest-|element| per row of Vᵀ and makes it positive; this is EXACTLY sklearn's `u_based_decision=False` branch which uses `argmax(|v|)` per row). [VERIFIED: sign_flip.rs vs sklearn svd_flip]

**When to use:** Every `partial_fit` call. `fit()` (D-02) resets state then loops `partial_fit` over `gen_batches(n_samples, batch_size)` with `batch_size = 5·n_features` when `None` (D-03).

### Pattern 2: pinvh `precision_` via v1 eig (D-05)
**What:** `precision_ = pinvh(covariance_)`. scipy's `pinvh` eigendecomposes the symmetric matrix, inverts eigenvalues above a cutoff, and reassembles.

```text
w, V = eig(covariance_)                       # v1 eig: descending w, V columns
cutoff = rcond * max(|w_i|)                    # scipy.linalg.pinvh default rcond
inv_w_i = (|w_i| > cutoff) ? 1/w_i : 0         # floored inverse (handles rank-deficient)
precision_ = V · diag(inv_w_i) · Vᵀ            # two GEMMs (or one scaled-V GEMM)
```
scipy `pinvh` default cutoff: `rcond = max(M,N) * eps` of the dtype (the same RCOND family the v1 04-03 LinearRegression σ⁺ pseudo-inverse already uses — reuse that constant/convention). For the rank-deficient MLE case (`n_samples ≤ n_features`) the near-zero eigenvalues are floored to a zero inverse, so `precision_` is the Moore-Penrose pseudo-inverse, never `inf`/`NaN`. [CITED: scipy.linalg.pinvh] [VERIFIED: eig.rs supplies w descending + V columns]

**Note on V layout:** `eig.rs` returns `V` **column-major** (`v[c*n + r] = V[r,c]`). pinvh reassembly must respect this — either build `V·diag` on the host (n is small, the Gram is n_features×n_features ≤ a few hundred) or use GEMM with the right transpose flag. Host reassembly is simplest and matches the small-n finalize idiom in `eig.rs`/`pca.rs`.

### Pattern 3: Ledoit-Wolf shrinkage (COV-02)
**Exact sklearn 1.7.1 `ledoit_wolf_shrinkage` math** (from `sklearn/covariance/_shrunk_covariance.py`):
```text
X = X_batch - mean (unless assume_centered)    # n × p
emp_cov = empirical_covariance(X, assume_centered)   # = covariance prim ddof=0
X2 = X**2
emp_cov_trace = sum(X2, axis=0) / n_samples          # length p
mu = sum(emp_cov_trace) / n_features                 # scalar

# β, δ accumulators (blocked dot products; at v2 sizes a single host pass is fine):
beta_  = sum( X2.T @ X2 )                             # = sum over (i,j) of (Σ_t X2[t,i]·X2[t,j])
delta_ = sum( (X.T @ X)**2 )                          # = Frobenius² of the unnormalized Gram

beta  = (1/(n_features*n_samples)) * (beta_/n_samples - delta_)
delta = (delta_ - 2*mu*emp_cov_trace.sum() + n_features*mu**2) / n_features
beta  = min(beta, delta)
shrinkage_ = 0 if beta == 0 else beta/delta          # ∈ [0,1] by construction (clip not strictly needed but apply)

# shrunk covariance:
covariance_ = (1 - shrinkage_) * emp_cov
covariance_[diag] += shrinkage_ * mu                 # + shrinkage·μ·I
```
[CITED: github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/covariance/_shrunk_covariance.py]
`emp_cov` reuses the v1 covariance prim (`ddof=0`). `beta_` and `delta_` are scalar reductions over `X²` and the Gram — compute `emp_cov` and the Gram on-device, read back the small p×p matrices, finalize β/δ/μ on the host in f64 (mirrors the kmeans inertia host-sum idiom). `shrinkage_` is mathematically in `[0,1]`; still apply the `min(...,1)`/`max(...,0)` clip per COV-02 wording.

### Pattern 4: RandomProjection matrices (PROJ-01/02)
**Gaussian** (verified): `components_[i,j] ~ N(0, 1/n_components)`, i.e. `splitmix_gaussian() / sqrt(n_components)`. Shape `n_components × n_features`.

**Sparse Achlioptas** (verified): density `s_inv = 1/density` (default `density = 1/sqrt(n_features)`). Each entry is:
- `0` with probability `1 − density`
- `+v` with probability `density/2`, `−v` with probability `density/2`, where `v = sqrt(1/density) / sqrt(n_components) = sqrt(s_inv/n_components)`.

[CITED: github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/random_projection.py]

`n_components='auto'` → `johnson_lindenstrauss_min_dim(n_samples, eps)`. `transform == X · components_ᵀ` (one GEMM, exactly like PCA transform; D-12 self-consistency gate). `components_` stored **dense** even for the sparse case (D-12). Sparse input densified at the Python ingress (D-12 / PROJ-02).

### Pattern 5: johnson_lindenstrauss_min_dim (value-matched)
```text
denominator = eps²/2 − eps³/3
n_components = floor( 4 · ln(n_samples) / denominator )
```
[CITED: sklearn/random_projection.py] This is the ONE projection quantity that is **value-matched to sklearn at 1e-5** (it's deterministic, no RNG). Returns an integer; emit it from the oracle as a value fixture.

### Anti-Patterns to Avoid
- **Building a new SVD-update device kernel.** Rejected (D-04 + this research): sklearn itself re-SVDs a dense stacked matrix; v2 sizes make the v1 Jacobi cost trivial. A new kernel adds cpu-MLIR risk and a validation surface for zero benefit.
- **Device-side RNG.** Backend-divergent → breaks seed-reproducibility across cpu/rocm (the PRIM-06 hard gate). Generate on host, upload once. [VERIFIED: kmeans.rs Anti-Pattern note]
- **Cholesky for `precision_`.** SPD-only; fails on the rank-deficient MLE covariance. Use eig-pinvh (D-05).
- **`np.cov` default ddof=1 for EmpiricalCovariance.** `empirical_covariance` uses `bias=1` (= `ddof=0`, divide by `n`). Using `ddof=1` silently fails the 1e-5 gate. Call `covariance(ddof=0)`.
- **A SharedMemory / `F::INFINITY` / mutable-bool kernel.** Both new prims are host-side glue precisely to avoid the cpu-MLIR launch panic (project memory). If the spike had forced a kernel it would have to be SharedMemory-free, F/u32-accumulator-only — but it does NOT force one.
- **Biased modulo in the RNG integer draw.** Use the existing `SplitMix64::next_below` rejection-sampling method (already in kmeans.rs) for Fisher-Yates, not `next_u64() % n`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense SVD of the stacked merge matrix | A new rank-update kernel | `prims::svd::svd` | Validated 1e-5; sklearn does the same; v2 sizes are tiny. |
| Symmetric inverse for `precision_` | A matrix-inversion kernel | `prims::eig::eig` + floored 1/λ + GEMM | pinvh is the exact sklearn contract; eig is validated and singular-safe. |
| Centered Gram `AᵀA/n` | Hand GEMM + scale | `prims::covariance::covariance(ddof=0)` | Already buffer-reuse-optimized + 1e-5-validated. |
| Sign canonicalization | New svd_flip | `mlrs_core::sign_flip::align_rows` | Already == sklearn `u_based_decision=False`. |
| Seeded PRNG | `rand` crate / `OsRng` | promote `SplitMix64` from kmeans.rs | ASVS V6; reproducible; zero new dep. |
| Unbiased integer draw / shuffle | `next_u64() % n` | `SplitMix64::next_below` (rejection) + Fisher-Yates | `next_below` already exists and is unbiased. |
| GIL release + dtype dispatch | New PyO3 boilerplate | `any_estimator!` macro | v2 adds zero binding infra. |

**Key insight:** Phase 7 is an *assembly* phase. The entire `[v2-P1]` risk collapses once you observe that scikit-learn's own `IncrementalPCA` is a dense-re-SVD-per-batch algorithm, not a streaming rank-update — so reusing the v1 SVD is not a compromise, it is *exactly the reference algorithm*.

## Runtime State Inventory

> Greenfield additions to the codebase (new prims/estimators), but one **refactor** touches existing code: promoting `SplitMix64` out of `kmeans.rs`.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores; oracle fixtures are committed `.npz` blobs regenerated offline. | None. |
| Live service config | None. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | None — the RNG seed is a caller-supplied documented `u64`, never an env var or `OsRng`. | None. |
| Build artifacts / source moves | `SplitMix64` struct (`kmeans.rs` L658-710) moves to `prims/rng.rs`. `kmeans.rs` must `use crate::prims::rng::SplitMix64` (or keep a re-export) so `kmeanspp_sample` still compiles. The `prims/mod.rs` index must register `pub mod rng;` and `pub mod incremental_svd;`. | Code edit: move struct, add `use`, update `mod.rs`. Verify `cargo build --features cpu` and the existing `kmeanspp_test.rs`/`lloyd_test.rs` still pass (the PRNG stream must be byte-identical after the move — it's a pure relocation). |

**Canonical question — after the source move, what still references the old location?** Only `kmeans.rs::kmeanspp_sample` (uses `SplitMix64::new`, `next_below`, `next_f64`). A re-export or a `use` fixes it; no behavior change. Run `kmeanspp_test.rs` to confirm the relocated PRNG produces the identical sequence.

## Common Pitfalls

### Pitfall 1: ddof confusion (covariance vs PCA variance)
**What goes wrong:** EmpiricalCovariance uses `ddof=0` (`np.cov(bias=1)`, divide by `n`); IncrementalPCA's `explained_variance_` uses `ddof=1` (`S²/(n_total−1)`). Mixing them fails the 1e-5 gate silently.
**Why:** Two different sklearn conventions in one phase.
**How to avoid:** EmpiricalCovariance/LedoitWolf → `covariance(ddof=0)`. IncrementalPCA `explained_variance_ = S²/(n_total−1)`. Pin both with separate oracle fixtures.
**Warning signs:** covariance off by a factor `n/(n−1)`.

### Pitfall 2: `mean_` vs `col_batch_mean` in the merge
**What goes wrong:** Using the wrong mean in `mean_correction`. The term is `sqrt((n_seen·b)/n_total)·(mean_ − col_batch_mean)` where `mean_` is the **prior running** mean and `col_batch_mean` is **this batch's own** mean. Centering `X` uses `col_mean` (the **updated running** mean), not `col_batch_mean`.
**Why:** Three different means (prior running, batch, updated running) appear in one step.
**How to avoid:** Compute `col_batch_mean = column_reduce(X_batch, Mean)` first; then update the running stats to get `col_mean`/`col_var`/`n_total`; center `X = X_batch − col_mean`; build `mean_correction` from the *prior* `mean_` and `col_batch_mean`. Follow the quoted sklearn order exactly.
**Warning signs:** First batch matches PCA, later batches drift.

### Pitfall 3: first-batch vs subsequent-batch stacking
**What goes wrong:** Building the 3-row-block stack on the first `partial_fit` (when there are no prior components). sklearn only stacks `Σ·Vᵀ` and `mean_correction` from the **second** batch onward.
**How to avoid:** Branch: first call → SVD of the centered batch alone (like PCA fit); subsequent → the full stack.
**Warning signs:** A shape mismatch or NaN on the very first batch.

### Pitfall 4: f32-on-rocm stability of the stacked re-SVD
**What goes wrong:** Fear that re-SVDing every batch accumulates f32 error past 1e-5 on rocm.
**Reality (analysis):** The stacked matrix is small (rows = `k + b + 1` ≤ a few hundred at v2 sizes; cols = `n_features`), so the v1 Jacobi's f32 behavior is the SAME well-characterized behavior the v1 PCA path already passes at 1e-5. The `Σ·Vᵀ` re-expansion *preserves* the energy of the running decomposition exactly (it's the rank-k reconstruction), so error does not compound beyond per-batch SVD rounding. **Conclusion: full-Jacobi-per-batch IS stable enough.** f64-on-cpu stays strict 1e-5; f32-on-rocm gets a documented band (see Validation Architecture).
**How to avoid:** Accumulate host-side combine math in f64 (the `host_to_f64`/`f64_to_host` idiom already used everywhere). Apply the documented f32 band only on rocm, gated by family.
**Warning signs:** Components drift only on rocm f32 and only after many batches — if seen, it's a sign-alignment bug (Pitfall 5), not an SVD-stability bug.

### Pitfall 5: sign alignment across batches AND vs oracle
**What goes wrong:** `components_` differ by a sign vs the sklearn oracle, OR the running `components_` flip sign between batches and destabilize the merge.
**Why:** Singular vectors are sign-ambiguous; sklearn applies `svd_flip(u_based_decision=False)` after every batch's SVD, and so must mlrs — using the SAME rule.
**How to avoid:** Call `align_rows` on `Vt` after every batch's SVD (it IS `u_based_decision=False`). Compare to the oracle only after `align_rows` (DECOMP-03 explicitly says "after svd_flip sign alignment").
**Warning signs:** A whole component row negated.

### Pitfall 6: explained_variance_ratio_ denominator
**What goes wrong:** Using the truncated sum or `S²`-sum as the ratio denominator. sklearn uses `S²/sum(col_var · n_total)` — the denominator is the **total feature variance × n_total**, not the sum of the kept `S²`.
**How to avoid:** `explained_variance_ratio_ = S²/(sum(col_var)·n_total)` (note `col_var` is per-feature ddof-aware variance from `_incremental_mean_and_var`). This differs from full-PCA's "sum over all explained_variance_" denominator (Pitfall 6 of v1 PCA) — IncrementalPCA's is the running total feature variance.
**Warning signs:** ratios don't sum sensibly / off by a constant factor.

### Pitfall 7: RNG stream identity after the source move
**What goes wrong:** Promoting `SplitMix64` subtly changes the stream (e.g. a different seed-mix), breaking `kmeanspp_test.rs` reproducibility.
**How to avoid:** Move the struct **verbatim**; do not "improve" it. Run `kmeanspp_test.rs` post-move.
**Warning signs:** KMeans++ init indices change.

### Pitfall 8: property-gate flakiness (D-10 strict bands)
**What goes wrong:** A strict JL-distortion / moment band flakes because a single random matrix happens to be near the band edge.
**How to avoid (D-11):** Fix the SplitMix64 seed (identical matrix across runs/backends) AND average the distortion/moment statistic over many trials (planner pins the trial count, e.g. 30–50 matrices / many sample pairs) so the *averaged* statistic concentrates tightly. The deterministic seed means the test is bit-reproducible per backend; averaging makes the strict threshold robust.
**Warning signs:** A test that's green most runs but occasionally red — means the band is on a single-draw statistic, not an averaged one.

## Code Examples

### Stacked-matrix merge (PRIM-07), host side
```rust
// Source pattern: pca.rs (host center + upload + svd) generalized to the stack.
// prev: (components_ k×p, singular_values_ k, mean_ p, var_ p, n_seen)
// batch: X_batch b×p  →  returns updated state.
//
// 1. batch own mean (device reduce), running stats update (host, Chan-Golub-LeVeque):
let col_batch_mean = column_reduce(pool, &x_batch, b, p, ScalarOp::Mean, ReducePath::Shared)?;
let (col_mean, col_var, n_total) = incremental_mean_var(/* host */ &prev_mean, &prev_var, n_seen, &x_batch_host, b, p);
// 2. center batch by the UPDATED running mean:
//    X_c[r,c] = X_batch[r,c] - col_mean[c]
// 3. build the stacked host matrix (subsequent batch):
//    rows 0..k     : singular_values_[i] * components_[i, :]
//    rows k..k+b   : X_c
//    row  k+b      : sqrt((n_seen*b)/n_total) * (prev_mean - col_batch_mean)
// 4. upload, SVD via the v1 prim, sign-flip:
let stacked_dev = DeviceArray::from_host(pool, &stacked);   // (k+b+1) × p
let (_u, s, vt) = svd::<F>(pool, &stacked_dev, (k + b + 1, p))?;   // MAX_ROWS/MAX_COLS checked
let vt_rows: Vec<Vec<f64>> = /* rows of vt */;
let vt_flipped = align_rows(&vt_rows);     // == svd_flip(u_based_decision=False)
// 5. explained_variance = s²/(n_total-1); ratio = s²/(sum(col_var)*n_total);
//    keep top n_components; store device-resident running state.
```

### EmpiricalCovariance fit (COV-01)
```rust
// location_ = mean (or 0 if assume_centered); covariance_ = covariance(ddof=0).
let location = if assume_centered { vec![F::zero(); p] }
               else { column_reduce(pool, &x, n, p, ScalarOp::Mean, Shared)?.to_host(pool) };
// center on host iff !assume_centered, then:
let cov = covariance::<F>(pool, &x_centered, (n, p), /*ddof=*/0, None)?;   // == np.cov(bias=1)
// precision_ (D-05/D-08), pinvh via eig:
let (w, v) = eig::<F>(pool, &cov, p, None)?;     // descending w, V columns
// cutoff = rcond*max|w|; inv = (|w|>cutoff)?1/w:0; precision = V·diag(inv)·Vᵀ
```

### johnson_lindenstrauss_min_dim (value-matched)
```rust
fn jl_min_dim(n_samples: f64, eps: f64) -> usize {
    let denom = eps*eps/2.0 - eps*eps*eps/3.0;
    (4.0 * n_samples.ln() / denom).floor() as usize
}   // == sklearn; emit as a value oracle fixture.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Streaming rank-update SVD (Brand/Bunch-Nielsen-Sorensen) | Dense re-SVD of a small stacked matrix per batch | sklearn IncrementalPCA design | No bespoke kernel; reuse the dense SVD. Confirmed adequate at v2 sizes. |
| `multi_class='multinomial'` arg | (n/a here) sklearn ≥1.5 removed it | — | Not relevant to Phase 7 estimators. |

**Deprecated/outdated:** Nothing in this phase's surface is deprecated in sklearn 1.7.1. `IncrementalPCA`, `EmpiricalCovariance`, `LedoitWolf`, `GaussianRandomProjection`, `SparseRandomProjection`, `johnson_lindenstrauss_min_dim` are all current.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy `pinvh` default cutoff family (`rcond = max(M,N)·eps`) is the right floor to match sklearn `precision_` at 1e-5; reuse the v1 04-03 σ⁺ RCOND constant. | Pattern 2 | If scipy's pinvh uses a different default threshold the rank-deficient `precision_` could differ at the floored eigenvalues. Mitigation: oracle fixture pins it; planner adds a near-singular covariance case. |
| A2 | The v1 SVD `MAX_ROWS`/`MAX_COLS` caps comfortably exceed the merged-matrix shape `(k+b+1) × n_features` at all v2 test sizes. | Pattern 1 | If a fixture uses a large `batch_size`/`n_features`, the stacked rows could exceed `MAX_ROWS` → `ShapeMismatch`. Mitigation: planner must read the actual `MAX_ROWS`/`MAX_COLS` constants and size fixtures under them; or cap `batch_size`. |
| A3 | f32-on-rocm full-Jacobi-per-batch holds a documented band ≈ the v1 PCA f32 band (no extra compounding). | Pitfall 4 | If error compounds across many batches the band must widen. Mitigation: the standalone 2+-batch PRIM-07 test (ROADMAP success criterion 2) measures it before estimators consume it; planner sets the band from that measurement. |
| A4 | Exact f32 tolerance band numbers (components/explained_variance) and the property-gate trial count are Claude's-discretion, to be pinned by the planner from the standalone prim test, following the v1 per-family band precedent. | Validation Architecture | Too-tight → flaky; too-loose → misses regressions. Mitigation: measure on the standalone prim, then set with margin (v1 precedent). |
| A5 | `_incremental_mean_and_var` returns `col_var` as the per-feature variance used directly in the ratio denominator `sum(col_var·n_total)`; mlrs reimplements the Chan-Golub-LeVeque update host-side. | Pattern 1 / Pitfall 6 | A wrong variance update fails `var_` and the ratio at 1e-5. Mitigation: quoted formula from extmath.py; oracle pins `var_` and `explained_variance_ratio_`. |

## Open Questions

1. **Exact scipy `pinvh` cutoff constant (A1).**
   - What we know: pinvh eigendecomposes + inverts eigenvalues above `rcond·max|λ|`; default `rcond` is dtype-eps-scaled.
   - What's unclear: whether to reuse the v1 04-03 σ⁺ RCOND constant verbatim or read scipy's exact default.
   - Recommendation: reuse the v1 σ⁺ RCOND convention (D-05 says to), and pin with a near-singular covariance oracle fixture; if it misses 1e-5, read scipy's `pinvh` default and match it.

2. **Whether to expose `error_norm`/`mahalanobis` (Claude's discretion).**
   - What we know: cheap to add; within the sklearn covariance surface.
   - Recommendation: defer unless trivially free — COV-01/02 only require `covariance_`/`location_`/`precision_`/`shrinkage_`. Keep the phase lean.

3. **`MAX_ROWS`/`MAX_COLS` headroom for the stacked matrix (A2).**
   - Recommendation: planner reads the constants from `mlrs-kernels` and sizes the IncrementalPCA fixtures (and the standalone PRIM-07 fixture) so `k + batch_size + 1 ≤ MAX_ROWS` and `n_features ≤ MAX_COLS`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust workspace (`mlrs-backend`/`-algos`/`-py`) | all | ✓ | in-tree | — |
| CubeCL cpu backend (f64 gate) | f64 oracle cases | ✓ | cubecl ^0.10 (cpu MLIR) | — |
| ROCm backend (f32 gate, gfx1100) | f32 oracle cases | ✓ | ROCm 7.1.1 (f32 only; f64 UNSUPPORTED) | `skip_f64_with_log` on rocm |
| Python venv (numpy+scipy+sklearn) | regenerating `.npz` oracle fixtures only | ✗ (PEP 668) | needs `/tmp` venv | committed `.npz` blobs already in tree; only needed when ADDING the new fixtures |
| New compute crate | — | n/a (forbidden) | — | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** sklearn/scipy/numpy are needed *only* to regenerate the new oracle fixtures (`gen_empirical_covariance`, `gen_ledoit_wolf`, `gen_incremental_pca`, `gen_jl_min_dim`). Per project memory ("oracle fixture regen needs venv"), create `/tmp/oracle-venv` with `numpy scipy scikit-learn`, run `scripts/gen_oracle.py`, commit the blobs. CI never runs the script.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test), tests in `crates/*/tests/` (AGENTS.md §2 — never in-source `mod tests`) |
| Config file | none — cargo test; feature-gated by `--features cpu` / `--features rocm` |
| Quick run command | `cargo test --features cpu -p mlrs-algos <test_name>` (targeted) |
| Full suite command | `cargo test --features cpu` then `cargo test --features rocm` (the two correctness gates) |

> Project memory: the `mlrs-backend` cpu suite is ~6 min (reduce_test 248s, svd_test 99s). Run **targeted** post-merge gates; background the full run. The new SVD-heavy `incremental_svd_test.rs` will be slow — keep its fixtures tiny.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-06 | Gaussian matrix distribution stats (mean≈0, var≈1/n_components) + seed-reproducibility (same seed → identical matrix) + Achlioptas density/value stats + Fisher-Yates permutation is a bijection | unit (distribution + repro) | `cargo test --features cpu -p mlrs-backend rng_` | ❌ Wave 0 |
| PRIM-06 | PoolStats memory gate for `rng.rs` (host-generate + single upload; bounded allocations) | unit (pool) | `cargo test --features cpu -p mlrs-backend rng_memory_gate` | ❌ Wave 0 |
| PRIM-07 | 2+-batch merge vs a host reference decomposition; ddof=1; svd_flip applied; f64 1e-5 / f32 band | unit (oracle + multi-batch) | `cargo test --features cpu -p mlrs-backend incremental_svd_` | ❌ Wave 0 |
| PRIM-07 | PoolStats memory gate for `incremental_svd.rs` | unit (pool) | `cargo test --features cpu -p mlrs-backend incremental_svd_memory_gate` | ❌ Wave 0 |
| COV-01 | `covariance_`/`location_`/`precision_` vs sklearn EmpiricalCovariance, 2 sizes incl. rank-deficient (n≤p) for precision_ | oracle (1e-5) | `cargo test --features cpu -p mlrs-algos empirical_covariance_` | ❌ Wave 0 |
| COV-02 | `shrinkage_` (∈[0,1]) + `covariance_` vs sklearn LedoitWolf, across two `n` | oracle (1e-5) | `cargo test --features cpu -p mlrs-algos ledoit_wolf_` | ❌ Wave 0 |
| DECOMP-03 | all attrs + `transform`/`inverse_transform` vs sklearn IncrementalPCA, via `partial_fit` over batches AND via `fit()`; whiten on/off | oracle (1e-5, post align_rows) | `cargo test --features cpu -p mlrs-algos incremental_pca_` | ❌ Wave 0 |
| PROJ-01/02 | property gate: JL distortion bound (averaged, strict), matrix moment stats, seed-repro across backends, `transform==X·componentsᵀ`; `johnson_lindenstrauss_min_dim` value-matched | property + 1 value-oracle | `cargo test --features cpu -p mlrs-algos random_projection_` | ❌ Wave 0 |
| (recurring) | every f64 oracle case gated by `skip_f64_with_log` | gate | (in each test) | pattern exists (gemm_test.rs) |

### Oracle harness extension (gen_oracle.py)
Add these generators + wire into `main()` (each emits f32 + f64 blobs unless property-gated):
- `gen_empirical_covariance(seed,dtype,n,p)` — `EmpiricalCovariance(assume_centered=…).fit(X)`; store `X`, `covariance_`, `location_`, `precision_`. Add a **rank-deficient** case (`n ≤ p`) so the pinvh floor is exercised. **VALUE-matched 1e-5.**
- `gen_ledoit_wolf(seed,dtype,n,p)` — `LedoitWolf().fit(X)`; store `X`, `covariance_`, `shrinkage_` (a length-1 array). Two `n` per ROADMAP criterion 3. **VALUE-matched 1e-5.**
- `gen_incremental_pca(seed,dtype,shape,n_components,batch_size,whiten)` — `IncrementalPCA(n_components,whiten,batch_size).fit(X)`; store `X`, `n_components`, `batch_size`, `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `var_`, `n_samples_seen_`, `transform(X)`, `inverse_transform(transform(X))`. Force C-contiguous on `components_` (the Fortran-order pitfall fixed in `gen_pca`). **VALUE-matched 1e-5 after align_rows.**
- `gen_jl_min_dim()` — emit `johnson_lindenstrauss_min_dim(n_samples, eps)` for a small grid of `(n_samples, eps)` as a value array. **VALUE-matched 1e-5.**
- RandomProjection: **NO value oracle for the matrix/transform** (D-12). The property test generates the matrix from a fixed SplitMix64 seed in Rust and checks structural properties; only `johnson_lindenstrauss_min_dim` has an oracle blob.

### Sampling Rate
- **Per task commit:** the targeted test for the file touched (`cargo test --features cpu -p <crate> <name>`).
- **Per wave merge:** the phase's new tests on cpu (`-p mlrs-algos` + the two new `-p mlrs-backend` prim tests); background the full cpu suite (~6 min).
- **Phase gate:** full cpu suite green, then `cargo test --features rocm` for the f32 bands (f64 skips-with-log on rocm), before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/rng_test.rs` — PRIM-06 distribution + seed-repro + Achlioptas + permutation + memory gate
- [ ] `crates/mlrs-backend/tests/incremental_svd_test.rs` — PRIM-07 2+-batch merge + memory gate
- [ ] `crates/mlrs-algos/tests/empirical_covariance_test.rs` — COV-01 (incl. rank-deficient precision_)
- [ ] `crates/mlrs-algos/tests/ledoit_wolf_test.rs` — COV-02 (two n)
- [ ] `crates/mlrs-algos/tests/incremental_pca_test.rs` — DECOMP-03 (partial_fit + fit + whiten)
- [ ] `crates/mlrs-algos/tests/random_projection_test.rs` — PROJ-01/02 property gate + jl_min_dim value
- [ ] `scripts/gen_oracle.py` — 4 new generators + `main()` wiring; regen in `/tmp` venv, commit blobs
- [ ] Framework install: none (cargo built-in)

## Security Domain

`security_enforcement: true`, ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface. |
| V3 Session Management | no | None. |
| V4 Access Control | no | Library code. |
| V5 Input Validation | **yes** | Every new prim/estimator validates geometry + hyperparameters BEFORE any unsafe device launch (the established `validate_geometry` / `AlgoError` pattern). New guards: `density ∈ (0,1]`, `eps ∈ (0,1)` for JL, `batch_size ≥ 1`, `n_components` range, `assume_centered` bool. Reject as typed `PrimError::ShapeMismatch`/`AlgoError`, never an OOB device read. |
| V6 Cryptography | **yes (negative requirement)** | RNG is the documented seeded `SplitMix64` — **NEVER `OsRng`, never the `rand` crate** (PRIM-06 + ASVS V6). SplitMix64 is NOT a CSPRNG and is not used for any security purpose; the seed is a caller-supplied documented `u64`. This is the existing kmeans.rs convention, preserved verbatim on the move. |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted hyperparameter → OOB device launch (e.g. `n_components`, `batch_size`, `density`) | Tampering / DoS | Validate-before-launch; typed `AlgoError`/`PrimError` (existing pattern, mirrored in every v1 prim). |
| `usize → u32` launch-dim truncation | Tampering | `guard_u32` before any kernel launch (existing kmeans.rs helper). |
| Stacked-matrix shape exceeding SVD caps | DoS (launch failure) | Validate `(k+b+1) ≤ MAX_ROWS`, `n_features ≤ MAX_COLS` before the `svd` call; size fixtures accordingly. |
| Non-reproducible RNG (OsRng) breaking the seed-repro gate | Repudiation (non-determinism) | Host SplitMix64 only; the PRIM-06 seed-reproducibility test is the enforcing control. |

## Sources

### Primary (HIGH confidence)
- v1 source files read directly this session: `traits.rs`, `covariance.rs`, `svd.rs`, `eig.rs`, `kmeans.rs` (SplitMix64), `pca.rs`, `error.rs`, `dispatch.rs`, `sign_flip.rs`, `gen_oracle.py`, `memory_gate_test.rs`, `pool.rs`, `capability.rs` — the reuse map, conventions, and the `svd_flip`/`align_rows` equivalence are read, not assumed.
- scikit-learn 1.7.1 source (fetched this session):
  - `sklearn/decomposition/_incremental_pca.py` — the exact stacked-matrix merge, mean-correction, svd_flip, explained_variance math.
  - `sklearn/covariance/_shrunk_covariance.py` — exact `ledoit_wolf_shrinkage` β/δ/μ + `shrunk_covariance`.
  - `sklearn/covariance/_empirical_covariance.py` — `np.cov(bias=1)` (ddof=0), `location_`, `precision_ = pinvh`.
  - `sklearn/random_projection.py` — `johnson_lindenstrauss_min_dim`, Gaussian `N(0,1/n_components)`, Achlioptas `sqrt(1/density)/sqrt(n_components)`, `n_components='auto'`.
  - `sklearn/utils/extmath.py` — `_incremental_mean_and_var` (Chan-Golub-LeVeque), `svd_flip(u_based_decision)` exact logic.
- Context7 resolution: `/scikit-learn/scikit-learn` v1.7.1 (docs index; source pinned via raw GitHub for exact formulas).

### Secondary (MEDIUM confidence)
- scipy.linalg.pinvh default cutoff convention (cross-referenced; pinned by oracle fixture — see A1).

### Tertiary (LOW confidence)
- Exact f32-on-rocm band numbers — to be measured on the standalone PRIM-07 prim test and set by the planner (A4); not yet numerically pinned.

## Metadata

**Confidence breakdown:**
- Incremental-SVD merge algorithm (the `[v2-P1]` decision): **HIGH** — it is sklearn's own algorithm, source-verified, and reuses a validated v1 prim.
- Covariance / LedoitWolf / RandomProjection math: **HIGH** — exact formulas read from sklearn 1.7.1 source this session.
- Reuse map / conventions: **HIGH** — read from v1 source.
- f32-on-rocm tolerance bands: **MEDIUM/LOW** — Claude's-discretion, pinned by the standalone prim measurement (precedent-based).
- pinvh cutoff constant: **MEDIUM** — reuse v1 RCOND, pin by fixture.

**Research date:** 2026-06-14
**Valid until:** 2026-07-14 (stable; sklearn 1.7.x algorithm math is settled. Re-verify only if sklearn major bumps or the v1 prim caps change.)

## RESEARCH COMPLETE
