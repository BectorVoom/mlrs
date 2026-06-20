# Phase 8: Kernel Family - Research

**Researched:** 2026-06-21
**Domain:** Kernel methods (KernelRidge dual solve, KernelDensity log-density) + one new CubeCL elementwise-map device prim over v1 distance/Gram
**Confidence:** HIGH (sklearn formulas read from local source v1.9.0; all prim signatures read from repo)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Kernel selection is a typed `Kernel<F>` enum with per-variant params — `Linear`, `Rbf { gamma }`, `Poly { gamma, degree, coef0 }`, `Sigmoid { gamma, coef0 }`. One value carries everything; the prim matches on it.
- **D-02:** Always compute the full general `K(X, Y)` (`rows_x × rows_y`). Training `K(X, X)` passes `Y = X`. No symmetry/upper-triangle special-case — one branch-free path; ~2× redundant compute on the symmetric case accepted at v2 sizes.
- **D-03:** The prim is self-contained — internally dispatches the base op (v1 `distance` squared-euclidean for RBF; v1 `gemm` `XYᵀ` Gram for linear/poly/sigmoid) then applies the elementwise map kernel. Callers just call `kernel_matrix(X, Y, Kernel::…)`. Map kernel feature-free, SharedMemory-free, no atomics, F/u32 accumulators only.
- **D-04:** Support multi-target `y` (`n_samples × n_targets`). `dual_coef_ = (K + αI)⁻¹ Y` with `Y` as `n_targets` RHS columns — v1 `cholesky_solve` already takes multiple `rhs`. `dual_coef_` is `n_samples × n_targets`.
- **D-05:** Mirror sklearn's `gamma=None` exactly: `gamma=None → 1/n_features` (computed at fit from `n_features`) for rbf/poly/sigmoid; explicit `gamma` as-is. Oracle pins BOTH the None-default and explicit-gamma paths to ≤ 1e-5.
- **D-06:** Scalar `alpha` + the 4 computed kernels only. NO `kernel='precomputed'`, NO per-target `alpha` array. KernelRidge has no intercept / no centering — do not add one.
- **D-07:** Ship all 6 sklearn KD kernels — gaussian, tophat, epanechnikov, exponential, linear, cosine. Compact-support kernels (tophat/epanechnikov/linear/cosine) yield exactly-zero density outside `bandwidth`; handled in the linear (non-log) domain, never with `F::INFINITY` in a kernel.
- **D-08:** KD is a distinct kernel family — functions of raw euclidean distance with dimension-dependent normalization, NOT the prim's dot-product kernels. KD composes directly over the v1 `distance` prim + density-kernel map + normalization + log-sum-exp. KD does NOT route through `kernel_matrix.rs`. The `kernel_matrix` prim serves KernelRidge + spectral (Phase 9) only.
- **D-09:** Support numeric `bandwidth` (float > 0) AND `'scott'` / `'silverman'` auto-bandwidth string rules (host-side closed-form). Pin exact formulas from sklearn source.
- **D-10:** Oracle = sklearn `KernelDensity` forced exact (`rtol=0.0, atol=0.0` → tree falls back to exact summation). mlrs computes brute-force exact pairwise log-density and matches within the documented tolerance.
- **D-11:** Device-side log-sum-exp, device-resident, via the v1 `reduce` prim. Operate in the linear kernel domain — kernel values non-negative with exact `0` for out-of-support points, zeros summed directly, never become `F::INFINITY`. Optional reduce-max rescale (divide by per-query max kernel value before summing, add `log(max)` back) gives max-shift without touching ±∞; single `log` at the end.
- **D-12:** Add `ScoreSamples<F>` next to `Fit`/`Predict`/`Transform`/`PartialFit` in `traits.rs`, same `<F: Float + CubeElement + Pod>` bound, same `pool`/`DeviceArray`/explicit-`(rows,cols)` device-resident convention. Returns length-`n` log-densities; `KernelDensity` implements it.

### Claude's Discretion
- Exact f32-on-rocm tolerance bands for KernelRidge predictions and KernelDensity log-density — follow v1 per-family documented-band precedent; f64 stays strict (≤ 1e-5 / documented KD tolerance), gated by `skip_f64_with_log`.
- Whether the D-11 reduce-max rescale is actually needed (vs plain linear reduce-sum) — decide from numerical testing during planning/execution.
- The precise `'scott'`/`'silverman'` formulas — pin from sklearn source. **(PINNED below — see KernelDensity section.)**
- Whether the elementwise map is a single kernel parameterized by a kernel-type uniform vs one kernel per variant — planner's call (keep it SharedMemory/atomics-free either way).

### Deferred Ideas (OUT OF SCOPE)
- `kernel='precomputed'` + per-target `alpha` array (KernelRidge).
- Tree-based KD acceleration (BallTree/KDTree) — brute-force exact only.
- Kernel SVC/SVR (SMO) — v3 backlog.
- Graph-Laplacian / spectral estimators (Phase 9, hard-depends on this prim).
- A bespoke fused kernel-matrix-then-reduce device kernel — only if the compose path proves a memory/perf problem (default: no).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRIM-08 | Kernel-matrix prim (linear/RBF/poly/sigmoid) composes over pairwise-distance/GEMM prims, validated vs host reference within tolerance f32/f64 — serving KernelRidge, KernelDensity, spectral affinity | `kernel_matrix.rs` design: `distance`/`gemm` base op dispatch + elementwise map kernel (Architecture Patterns). Note D-08 refines the requirement wording: KD does NOT consume this prim — the shared base under both is the v1 `distance` prim. |
| KERNEL-01 | `KernelRidge` dual-coefficient solve of `(K + αI)` via v1 Cholesky; kernels linear/rbf/polynomial/sigmoid with `gamma`/`degree`/`coef0`; `predict` ≤ 1e-5 | Exact sklearn dual math + kernel formulas + gamma/degree/coef0 defaults pinned from source (KernelRidge section). `cholesky_solve(n, rhs)` multi-RHS confirmed for D-04. |
| KERNEL-02 | `KernelDensity` (kernels + `bandwidth`); `score_samples` log-density via numerically-stable log-sum-exp ≤ documented tolerance | All 6 per-kernel log formulas + per-kernel log-normalization constants + scott/silverman bandwidths + the `logsumexp − log(N)` assembly pinned from sklearn source (KernelDensity section). |
| PY-06 (incremental share) | `score_samples` exposed; both estimators `#[pyclass]`-backed via `any_estimator!` | `any_estimator!` macro + dtype-suffixed accessors + `py.detach`/`guard_f64` pattern (Code Examples). `score_samples` is the one new exposed method. |
</phase_requirements>

## Summary

Phase 8 adds two estimators and one device primitive. The design is settled (ROADMAP marks no research flag); this research pins the **concrete numeric formulas and constants** the oracle tests must match, plus the exact reuse signatures of the v1 prims the planner writes call sites against.

The new device work is a **single SharedMemory-free elementwise map kernel** (`kernel_matrix`) that post-processes a base matrix produced by the validated v1 `distance` (RBF) or `gemm` (linear/poly/sigmoid) prim. This is structurally identical to the existing `dist_combine_clamp` / `center_columns` / `scale` kernels in `elementwise.rs` — one unit per output element, bounds-checked, scalar params passed by value. The only new ingredient is transcendental intrinsics (`F::exp`, `F::tanh`, `F::powf`/integer power, plus `F::log`/`F::cos` for KD), which exist as static associated functions on cubecl 0.10's `Float` trait and are NOT a cpu-MLIR landmine (the landmine is specifically SharedMemory + mutable `bool` + `F::INFINITY` + descending-shift-loops, none of which this map needs).

`KernelRidge` mirrors `ridge.rs` closely but with two deletions and one addition: **no centering / no intercept** (sklearn `KernelRidge` fits raw data), the normal matrix is `K` (not the Gram `XᵀX`), and `α` is added to the `K` diagonal exactly as sklearn's `_solve_cholesky_kernel` does (`K.flat[::n+1] += alpha`), then solved with `assume_a="pos"` (Cholesky) — a direct fit for v1 `cholesky_solve` with `rhs = n_targets` (D-04 is near-free). `KernelDensity` composes over the v1 `distance` prim and a device log-sum-exp over the v1 `reduce` prim; all six kernel formulas and their dimension-dependent log-normalization constants were read verbatim from sklearn 1.9.0 `_binary_tree.pxi.tp`.

**Primary recommendation:** Land + standalone-validate `prims/kernel_matrix.rs` (4 kernels, f32+f64, host reference, PoolStats gate) FIRST; then build `KernelRidge` on it (mirror `ridge.rs` minus centering); build `KernelDensity` directly on the v1 `distance` prim + a new device log-sum-exp helper (NOT on `kernel_matrix`); add `ScoreSamples<F>`; wrap both via `any_estimator!`. Pin every formula and constant below into the plans verbatim.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kernel matrix `K(X,Y)` (linear/rbf/poly/sigmoid) | `mlrs-backend` prim (`kernel_matrix.rs`) | `mlrs-kernels` (the map kernel) | Composes v1 distance/gemm + one new elementwise map; reusable seam for KernelRidge + Phase 9 + kernel-SVM |
| `K + αI` dual solve | `mlrs-algos` estimator (`KernelRidge`) | `mlrs-backend` `cholesky_solve` | Estimator orchestrates kernel build → diagonal-α → Cholesky multi-RHS solve |
| KD per-kernel log-density + normalization | `mlrs-algos` estimator (`KernelDensity`) | `mlrs-backend` `distance` + `reduce` + new map/logsumexp kernels | KD is a distinct family over raw distance (D-08); does not touch `kernel_matrix.rs` |
| Device log-sum-exp (linear domain) | `mlrs-backend` (new helper) over v1 `reduce` | `mlrs-kernels` (exp/scale/log map kernels) | D-11: reduce-max + rescale + reduce-sum + final log, never ±∞ |
| `ScoreSamples<F>` trait surface | `mlrs-algos` (`traits.rs`) | — | Cross-cutting estimator contract, mirrors `PartialFit<F>` |
| Python `fit`/`predict`/`score_samples` dispatch | `mlrs-py` (`estimators/kernel.rs`) | `any_estimator!` macro | Unfit/F32/F64 enum + GIL release + f64 guard |

## Standard Stack

**No new compute dependency** (workspace `Cargo.toml` line 16 pins `cubecl = "0.10.0"`; v2 mandate adds zero deps, pyo3 stays 0.28). Everything below already exists in the repo.

### Core (existing prims this phase composes)
| Component | Where | Purpose | Exact signature (read from source) |
|-----------|-------|---------|------------------------------------|
| `distance` | `mlrs-backend/src/prims/distance.rs` | squared-euclidean (RBF base + ALL 6 KD kernels' base); optional sqrt boundary | `distance::<F>(pool, x, (rows_x,cols), y, (rows_y,cols_y), sqrt: bool, out: Option<DeviceArray>) -> Result<DeviceArray, PrimError>` |
| `gemm` | `prims/gemm.rs` | `XYᵀ` Gram (linear/poly/sigmoid base) | `gemm::<F>(pool, a, (m,k), b, (k2,n), transa: bool, transb: bool, out: Option<DeviceArray>) -> Result<DeviceArray, PrimError>` |
| `cholesky_solve` | `prims/cholesky.rs` | SPD `(K+αI)` dual solve, MULTI-RHS | `cholesky_solve::<F>(pool, a, b, n: usize, rhs: usize, out: Option<DeviceArray>) -> Result<DeviceArray, PrimError>` — `b` is `n×rhs`, returns `n×rhs`. `n ≤ MAX_DIM` required. |
| `reduce::max` / `reduce::sum` | `prims/reduce.rs` | full-array max + sum for D-11 log-sum-exp | `max::<F>(pool, input, ReducePath::Shared)`; `row_reduce::<F>(pool, input, rows, cols, ScalarOp::Max\|Sum, ReducePath::Shared)` for per-query rows |
| elementwise map kernels | `mlrs-kernels/src/elementwise.rs` | `#[cube(launch)]` pattern to mirror | `scale`, `sqrt_elem`, `center_columns`, `dist_combine_clamp` — copy this exact shape |

### Supporting (new files to create)
| File | Purpose | Mirror |
|------|---------|--------|
| `mlrs-backend/src/prims/kernel_matrix.rs` | host orchestration: base-op dispatch + map launch + PoolStats | `covariance.rs` (GEMM → in-place map idiom) |
| `mlrs-kernels/src/<kernel_map>.rs` (or extend `elementwise.rs`) | the `#[cube(launch)]` map kernel(s) | `dist_combine_clamp` / `scale` |
| `mlrs-algos/src/kernel_ridge/…` (new module group) | `KernelRidge` estimator | `linear/ridge.rs` MINUS centering |
| `mlrs-algos/src/<density>/…` (planner's home, e.g. `neighbors` or new `density`) | `KernelDensity` estimator | composes `distance` + new logsumexp |
| `mlrs-py/src/estimators/kernel.rs` | both `#[pyclass]` wrappers | `estimators/covariance.rs` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Compose `kernel_matrix` = distance/gemm + map (D-03) | A single fused kernel-matrix kernel | Fused avoids one buffer pass but reinvents validated GEMM/distance and risks cpu-MLIR; deferred per CONTEXT (default no) |
| KD over v1 `distance` (D-08) | KD over `kernel_matrix(Rbf)` | Only gaussian is rbf-like; the other 5 KD kernels are raw-distance compact-support — one consistent path is simpler and correct |

**Installation:** None. No new crate. Register `pub mod kernel_matrix;` in `prims/mod.rs`; register new estimator modules in `mlrs-algos/src/lib.rs`; register the new `#[pyclass]`es in the pymodule.

## Package Legitimacy Audit

This phase installs **no external packages** (Rust workspace deps unchanged; pyo3 0.28 unchanged; no new Python runtime deps — sklearn/scipy/numpy are already the build-time-only oracle generator deps in a `/tmp` venv). Package Legitimacy Gate not applicable.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
KernelRidge.fit(X, y):                       KernelDensity.score_samples(Q):
  X (n×d), y (n×t)                              Q (m×d) queries, X_fit (n×d) training, bandwidth h
        │                                              │
        ▼                                              ▼
  kernel_matrix(X, X, Kernel)  ── D-03 ──▶       distance(Q, X_fit, sqrt=as-needed)   ── D-08, v1 distance ──▶
   ├ Rbf  → distance(X,X,sqrt=false)=sqdist        D = pairwise dist  (m×n)   [sqdist for gaussian; sqrt for compact kernels]
   │        → map: exp(-gamma·sqdist)                    │
   └ lin/poly/sig → gemm(X,Xᵀ)=G (XYᵀ)            ┌──────┴───────────────────────────┐
            → map: G | (γG+c0)^deg | tanh(γG+c0)  │ per-element KD map (linear domain)│
        │  K (n×n)                                │  k_ij = density_kernel(D_ij, h)   │  ── exact 0 outside support, never ∞ (D-11)
        ▼                                         └──────┬───────────────────────────┘
  K.diagonal += alpha   (sklearn _solve_cholesky_kernel) │  kernel matrix k (m×n), non-negative
        ▼                                                ▼
  cholesky_solve(K, y, n, rhs=t)   ── D-04 ──▶     per-query (row) log-sum-exp  ── D-11, v1 reduce ──▶
        │  dual_coef_ (n×t)                          row_max = reduce_max(k_row); s = Σ (k_row / row_max)
        ▼                                            lse_row = log(s) + log(row_max)
  predict(X_test):                                       │
   K_test = kernel_matrix(X_test, X_fit, Kernel)         ▼
   y_pred = K_test · dual_coef_   (gemm)            log_density = lse_row + log_norm(h,d,kernel) − log(N)
                                                         │  length m
                                                         ▼  (NB: log_norm is dimension-dependent, per-kernel)
```

### Recommended Project Structure
```
crates/mlrs-backend/src/prims/
├── kernel_matrix.rs    # NEW: K(X,Y) host orchestration (distance/gemm dispatch + map launch + PoolStats)
└── mod.rs              # add: pub mod kernel_matrix;
crates/mlrs-kernels/src/
├── elementwise.rs      # extend (or new file) with the kernel-map + exp/log/scale map kernels
crates/mlrs-algos/src/
├── traits.rs           # add ScoreSamples<F>
├── kernel_ridge/       # NEW estimator module group (planner picks layout)
├── <density>/          # NEW KernelDensity home (planner's call: neighbors/ or new density/)
└── lib.rs              # register new modules + re-export ScoreSamples
crates/mlrs-py/src/
├── estimators/kernel.rs  # NEW: PyKernelRidge + PyKernelDensity
└── ...                   # register pyclasses in the pymodule
```

### Pattern 1: Elementwise-map prim over a GEMM/distance base (the `kernel_matrix` shape)
**What:** Run the validated base prim into a pool buffer, then launch a SharedMemory-free per-element map kernel that overwrites/derives the kernel value.
**When to use:** `kernel_matrix.rs` and the KD density map.
**Example (host orchestration — mirror `covariance.rs` GEMM→in-place-scale):**
```rust
// Source: pattern read from crates/mlrs-backend/src/prims/covariance.rs (lines 161-204)
// 1. base op (Rbf branch): squared-euclidean distance, sqrt=false → sqdist (rows_x × rows_y)
let base = distance::<F>(pool, x, (rows_x, cols), y, (rows_y, cols), false, out)?;
// 2. per-element map IN PLACE over the base buffer (input handle == output handle)
let n = rows_x * rows_y;
let (count, dim) = launch_dims_1d(n);           // 256-wide ceiling-div, like covariance
let in_arg  = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
rbf_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, gamma /* scalar F by value */);
Ok(base)
```

### Pattern 2: The map kernel itself (`#[cube(launch)]`, SharedMemory-free)
**What:** One unit per element, bounds-checked, transcendental via static associated fn.
**Example (mirror `scale` in elementwise.rs; RBF shown):**
```rust
// Source: crates/mlrs-kernels/src/elementwise.rs scale() shape; F::exp is the static assoc fn
// per cubecl_manual/.../mismatched types.md: "F::exp(x), NOT x.exp()"
#[cube(launch)]
pub fn rbf_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, gamma: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        // input is squared-euclidean distance; RBF = exp(-gamma * sqdist)
        output[tid] = F::exp(-gamma * input[tid]);
    }
}
// poly:    out = F::powf(gamma * g + coef0, degree)   // or integer-power loop if degree is u32
// sigmoid: out = F::tanh(gamma * g + coef0)
// linear:  out = g                                    // identity (no map; can skip the launch)
```
**Poly degree note:** sklearn allows non-integer `degree` (default 3) via `K **= degree`. `F::powf` matches it. If the planner restricts to integer degree, an integer-power multiply-loop also works (still SharedMemory-free) but `powf` is the sklearn-faithful default — use `powf`.

### Pattern 3: KernelRidge dual solve (mirror `ridge.rs` MINUS centering)
**What:** Build `K(X,X)`, add `α` to its diagonal, Cholesky multi-RHS solve.
```rust
// Source: sklearn _solve_cholesky_kernel (linalg.solve(K, y, assume_a="pos")) + repo ridge.rs
let k = kernel_matrix::<F>(pool, x, (n,d), x, (n,d), kernel)?;   // n×n, Y=X (D-02)
// α on the K DIAGONAL only — sklearn: K.flat[::n_samples+1] += alpha[0]  (host pass, like ridge.rs)
let mut k_host = k.to_host(pool);
for i in 0..n { let v = host_to_f64(k_host[i*n + i]) + alpha64; k_host[i*n+i] = f64_to_host::<F>(v); }
k.release_into(pool);
let k_reg = DeviceArray::from_host(pool, &k_host);
// multi-RHS: y is n×t (D-04); cholesky_solve handles rhs columns natively
let k_out = DeviceArray::from_raw(k_reg.handle().clone(), n*n);
let dual = cholesky_solve::<F>(pool, &k_reg, &y, n, n_targets, Some(k_out))?;  // dual_coef_ (n×t)
// NO intercept, NO centering (D-06). predict: K_test (m×n) · dual (n×t) via gemm.
```

### Anti-Patterns to Avoid
- **Centering X/y in KernelRidge** — sklearn `KernelRidge` fits RAW data (verified: `fit` calls `_get_kernel(X)` then `_solve_cholesky_kernel(K, y, alpha)` with no centering; predict is `K(X_test, X_fit) · dual_coef_`). Do NOT copy ridge.rs's center-then-recover-intercept block (D-06).
- **`F::INFINITY` for out-of-support KD kernels** — compact-support kernels produce exact `0` in the linear domain; `log` is applied ONCE at the very end after summing (D-11 / cpu-MLIR landmine).
- **SharedMemory in the map kernel** — the `n×n` operands stay in global memory (gfx1100 LDS ≤ 65536 B); the map is purely elementwise (no reduction in the map; reductions go through the v1 `reduce` prim).
- **Routing KD through `kernel_matrix(Rbf)`** — D-08: KD is raw-distance + dimension-dependent normalization; only gaussian is rbf-like and even it keeps the KD path for consistency.
- **`gamma` from training n_features in `predict`** — sklearn resolves `gamma=None → 1.0/X.shape[1]` inside each `pairwise_kernels` call, where `X` is the FIRST argument. In `predict`, `_get_kernel(X, self.X_fit_)` passes the test X first — but `n_features` is identical for train and test, so `1/n_features` is unambiguous. Compute `gamma` once at `fit` from `n_features` (D-05) and reuse it in `predict`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pairwise squared distance | A new distance loop | v1 `distance` prim (`sqrt=false`) | Validated GEMM-expansion + non-negative clamp; base for RBF and all 6 KD kernels |
| `XYᵀ` Gram | A new matmul | v1 `gemm` (`transb=true`) | cubek-matmul wrap, f64-accurate accumulator |
| SPD `(K+αI)` multi-target solve | A new Cholesky | v1 `cholesky_solve(n, rhs)` | Multi-RHS already supported (D-04 near-free); non-SPD pivot → typed `NotPositiveDefinite`, never NaN |
| max / sum reductions for log-sum-exp | A bespoke reduce kernel | v1 `reduce::max` / `reduce::row_reduce(Sum)` | Multi-pass pairwise-stable; shared path is cpu-portable |
| Dtype dispatch + GIL + f64 guard | Hand-written enum per estimator | `any_estimator!` macro | v2 adds zero binding infra; encodes `py.detach` + `guard_f64()` contracts |
| Oracle RNG | Rust-side random data | `numpy.random.default_rng(seed)` in `gen_oracle.py` | Byte-reproducible committed `.npz` blobs; CI never runs Python |

**Key insight:** Every heavy operation already exists as a validated v1 prim. The ONLY genuinely new device code is one elementwise map kernel (and a small KD log-sum-exp helper) — both trivially SharedMemory-free. The risk is numeric fidelity (matching sklearn's exact constants), not new algorithm design — which is why this research pins every formula below.

## KernelRidge — exact sklearn math (VERIFIED: local sklearn 1.9.0 source)

### Dual solve
`sklearn.kernel_ridge.KernelRidge.fit` → `K = pairwise_kernels(X, metric=kernel, gamma=…, degree=…, coef0=…)` then
`dual_coef_ = _solve_cholesky_kernel(K, y, alpha)`. [VERIFIED: sklearn/kernel_ridge.py read locally]

`_solve_cholesky_kernel(K, y, alpha)` (single scalar alpha path): [VERIFIED: sklearn/linear_model/_ridge.py read locally]
```
K.flat[:: n_samples + 1] += alpha[0]          # α added to K's DIAGONAL in place
dual_coef = linalg.solve(K, y, assume_a="pos") # Cholesky (SPD) solve, y may be n×t
K.flat[:: n_samples + 1] -= alpha[0]           # restored (irrelevant to result)
```
- `assume_a="pos"` ⇒ Cholesky ⇒ **exactly v1 `cholesky_solve`**. Multi-target `y` (n×t) solved in ONE call ⇒ `rhs = n_targets` (D-04).
- **No centering, no intercept, no sample scaling.** `α` is NOT divided by `n_samples` (unlike some Ridge solvers).
- `predict(X)`: `K_test = pairwise_kernels(X, X_fit_, …)` (m×n), `y_pred = K_test @ dual_coef_` (m×t). [VERIFIED]

### Hyperparameter defaults (VERIFIED from `KernelRidge.__init__` and the kernel functions)
| Param | Default | Notes |
|-------|---------|-------|
| `alpha` | `1.0` | scalar; `Interval(Real, 0, None)` ⇒ α ≥ 0 |
| `kernel` | `"linear"` | enum maps to the 4 supported |
| `gamma` | `None` → `1.0 / X.shape[1]` = **1/n_features** | resolved per-call inside each kernel fn (D-05) |
| `degree` | `3` | poly only; sklearn allows real degree via `K **= degree` |
| `coef0` | `1` | **poly AND sigmoid default to coef0=1, NOT 0** — easy to get wrong |

### Kernel formulas (VERIFIED from `sklearn/metrics/pairwise.py`)
With `G = X @ Yᵀ` (the Gram / dot-product matrix) and `gamma` resolved as above:
| Kernel | Formula | Source lines |
|--------|---------|--------------|
| linear | `K = G` | `linear_kernel`: `X @ Y.T` |
| polynomial | `K = (gamma * G + coef0) ** degree` | `K *= gamma; K += coef0; K **= degree` |
| sigmoid | `K = tanh(gamma * G + coef0)` | `K *= gamma; K += coef0; np.tanh(K, K)` |
| rbf | `K = exp(-gamma * ‖x−y‖²)` | `K *= -gamma; np.exp(K, K)` (over squared-euclidean) |

**RBF uses squared distance** (`euclidean_distances(X, Y, squared=True)`) — so v1 `distance(sqrt=false)` is the exact base. Linear/poly/sigmoid use `G = X@Yᵀ` — so v1 `gemm(transb=true)` is the exact base.

### Oracle cases the planner must pin (≤ 1e-5 f64, documented band f32)
- One per kernel (linear, rbf, poly, sigmoid), single-target.
- One **2-target** rbf (or any) case to exercise the multi-RHS path (D-04 / D-10).
- One **explicit-gamma** AND one **gamma=None** case (D-05 pins both paths).
- Use `degree=3, coef0=1` defaults in at least one poly/sigmoid case so the non-zero coef0 default is exercised.

## KernelDensity — exact sklearn math (VERIFIED: local sklearn 1.9.0 `_binary_tree.pxi.tp` + `_kde.py`)

### Density assembly (`score_samples`)
For query point with raw distances `dist_i` to the N training points: [VERIFIED: `_kde.py` score_samples read locally]
```
log_density(query) = logsumexp_i [ log_norm(h, d, kernel) + log_kernel(dist_i, h) ]  −  log(N)
```
`log_norm` is per-query CONSTANT (does not depend on i), so it factors out of the sum:
```
log_density(query) = log_norm(h, d, kernel) + logsumexp_i[ log_kernel(dist_i, h) ] − log(N)
```
This is the D-11 linear-domain form: `Σ_i exp(log_kernel_i)` is exactly `Σ_i kernel_value_i` (non-negative, exact 0 out of support), then `+log_norm − log(N)` applied once. `N = n_training_samples` (no sample weights in scope). [VERIFIED]

### Per-kernel unnormalized log-kernel `log_kernel(dist, h)` (VERIFIED, `_binary_tree.pxi.tp` lines 376-415)
> `dist` is RAW euclidean distance (not squared); `h` = bandwidth. Kernels normalized so `K(0,h)=1`.

| Kernel | log_kernel(dist, h) | linear-domain kernel value |
|--------|---------------------|----------------------------|
| gaussian | `−0.5 * dist² / h²` | `exp(−0.5·dist²/h²)` — uses **squared** dist ⇒ v1 `distance(sqrt=false)` directly |
| tophat | `0` if `dist < h` else `−∞` | `1` inside `h`, **exact 0** outside |
| epanechnikov | `log(1 − dist²/h²)` if `dist < h` else `−∞` | `1 − dist²/h²` inside, **0** outside |
| exponential | `−dist / h` | `exp(−dist/h)` — uses **raw** dist ⇒ needs sqrt boundary |
| linear | `log(1 − dist/h)` if `dist < h` else `−∞` | `1 − dist/h` inside, **0** outside |
| cosine | `log(cos(0.5·π·dist/h))` if `dist < h` else `−∞` | `cos(0.5·π·dist/h)` inside, **0** outside |

**Implementation note (D-11):** compute kernel VALUES (right column), never the log form, so out-of-support yields exact `0` and never `−∞`/`F::INFINITY`. gaussian/epanechnikov need only `dist²` (use `distance(sqrt=false)`); exponential/linear/cosine/tophat compare against `h` and so need raw `dist` (use `distance(sqrt=true)` or sqrt the squared base in the map). The compact-support guard is a STATEMENT-form `if d < h { val } else { 0 }` (mirror the `dist_combine_clamp` clamp statement form per `Cubecl_conditionals.md`).

### Per-kernel log-normalization constant `log_norm(h, d, kernel)` (VERIFIED, lines 438-476)
Helpers (d = n_features):
```
logVn(n) = 0.5*n*log(π) − lgamma(0.5*n + 1)        # log volume of unit n-ball
logSn(n) = log(2π) + logVn(n − 1)                   # log surface area
_log_kernel_norm(h,d,kernel) = −factor − d*log(h)   # NOTE the leading minus and −d·log(h)
```
| Kernel | `factor` | Full `log_norm = −factor − d·log(h)` |
|--------|----------|--------------------------------------|
| gaussian | `0.5 * d * log(2π)` | `−0.5·d·log(2π) − d·log(h)` |
| tophat | `logVn(d)` | `−logVn(d) − d·log(h)` |
| epanechnikov | `logVn(d) + log(2/(d+2))` | `−logVn(d) − log(2/(d+2)) − d·log(h)` |
| exponential | `logSn(d−1) + lgamma(d)` | `−logSn(d−1) − lgamma(d) − d·log(h)` |
| linear | `logVn(d) − log(d+1)` | `−logVn(d) + log(d+1) − d·log(h)` |
| cosine | `log(Σ-series) + logSn(d−1)` (series below) | `−log(Σ) − logSn(d−1) − d·log(h)` |

Cosine series (chain-rule integration, lines 466-473):
```
factor = 0; tmp = 2/π
for k in range(1, d+1, 2):           # k = 1, 3, 5, …, ≤ d
    factor += tmp
    tmp *= −(d−k)*(d−k−1)*(2/π)²
factor = log(factor) + logSn(d−1)
```
**This `log_norm` is a host-side scalar** (depends only on `h`, `d`, kernel) — compute it on the host in `f64` with `libm`/std `lgamma`/`ln`/etc., then add it to the device-computed `logsumexp`. Do NOT attempt `lgamma` on device. [ASSUMED: that host-side f64 lgamma matches the Cython `lgamma` to ≤ documented tolerance — verify in the oracle test; both call the same C `lgamma`.]

### Bandwidth resolution (VERIFIED, `_kde.py` lines 223-231)
```
scott:     bandwidth_ = n_samples ** (−1 / (n_features + 4))
silverman: bandwidth_ = (n_samples * (n_features + 2) / 4) ** (−1 / (n_features + 4))
numeric:   bandwidth_ = bandwidth        # float > 0 used as-is
```
Both are pure host-side closed forms over `n_samples`/`n_features` (D-09). **sklearn's scott/silverman differ from scipy's** (scipy multiplies by per-feature std; sklearn does NOT) — these formulas are the sklearn ones, the correct oracle.

### KernelDensity defaults (VERIFIED, `_kde.py __init__`)
| Param | Default | In scope |
|-------|---------|----------|
| `bandwidth` | `1.0` | yes (float or "scott"/"silverman" — D-09) |
| `kernel` | `"gaussian"` | yes (all 6 — D-07) |
| `metric` | `"euclidean"` | yes (only euclidean; normalization correct only for euclidean) |
| `atol` | `0` | oracle forces exact (D-10) |
| `rtol` | `0` | oracle forces exact (D-10) — tree falls back to exact summation |

### Oracle (D-10) the planner must pin
- `KernelDensity(kernel=…, bandwidth=…, atol=0, rtol=0).fit(X).score_samples(Q)` per kernel.
- Small `n` so brute-force matches the exact-forced tree deterministically.
- At least one `bandwidth='scott'` and one `'silverman'` case (D-09).
- Documented tolerance (not strict 1e-5) for KD log-density per KERNEL-02 wording — large dynamic range. f64 strict to a documented KD tol; f32-on-rocm a wider documented band (Claude's discretion).

## Runtime State Inventory

> Greenfield additive phase (new prim + new estimators + new trait). No rename/refactor. Omitting the full table; nothing stored/registered/migrated. New files only; no existing string/key/collection is renamed.

## Common Pitfalls

### Pitfall 1: Copying ridge.rs's centering into KernelRidge
**What goes wrong:** `dual_coef_` and predictions diverge from sklearn.
**Why:** sklearn `KernelRidge` fits raw data — no `fit_intercept`, no centering, no `intercept_`. `ridge.rs` centers and recovers an intercept.
**How to avoid:** Delete the centering/intercept block; the normal matrix is `K` (not `XᵀX`), `α` goes on the `K` diagonal, solve, done (D-06).
**Warning sign:** Any `x_mean`/`y_mean`/`intercept_` appearing in KernelRidge.

### Pitfall 2: Wrong `coef0` default (0 vs 1)
**What goes wrong:** poly/sigmoid oracle off by a constant inside the nonlinearity.
**Why:** sklearn defaults `coef0=1` for both poly and sigmoid (NOT 0).
**How to avoid:** Default `coef0 = 1`; pin a default-coef0 oracle case. [VERIFIED]

### Pitfall 3: `−∞` / `F::INFINITY` in compact-support KD kernels
**What goes wrong:** cpu-MLIR launch panic ([[cubecl-cpu-no-shared-memory]]) and/or NaN in the sum.
**Why:** out-of-support log-kernel is `−∞`; the cpu-MLIR backend dies on `F::INFINITY` and `−∞` poisons the sum.
**How to avoid:** D-11 — compute kernel VALUES (exact 0 out of support) via a STATEMENT-form `if d < h` guard; sum in the linear domain; apply `log` ONCE at the end.
**Warning sign:** any `F::INFINITY` / `NEG_INFINITY` / `log` inside the per-element map.

### Pitfall 4: Using raw distance where sklearn uses squared (or vice versa)
**What goes wrong:** gaussian/epanechnikov wrong if you sqrt; exponential/linear/cosine/tophat wrong if you don't.
**Why:** gaussian = `−0.5·dist²/h²` and epanechnikov = `1−dist²/h²` use **squared** distance; the others compare raw `dist < h`.
**How to avoid:** gaussian/epanechnikov over `distance(sqrt=false)`; the 4 raw-distance kernels over `distance(sqrt=true)`. RBF (KernelRidge) uses squared.

### Pitfall 5: `gamma` resolution timing/source
**What goes wrong:** explicit-gamma vs gamma=None paths disagree with sklearn.
**Why:** `gamma=None → 1.0/X.shape[1]` is resolved per-call; `X.shape[1] = n_features`.
**How to avoid:** Resolve `gamma` once at `fit` from `n_features`, store it, reuse in `predict`. Pin both paths (D-05). [VERIFIED]

### Pitfall 6: `cholesky_solve` `n ≤ MAX_DIM` cap on the `n×n` K matrix
**What goes wrong:** K is `n_samples × n_samples`; for KernelRidge `n` = training samples (not features). If `n_samples > MAX_DIM` the single-cube Cholesky kernel rejects it (`PrimError::NotSquare`).
**Why:** the Cholesky kernel stages L in shared memory capped at `MAX_DIM` (read from `cholesky.rs`).
**How to avoid:** Keep KernelRidge oracle `n_samples ≤ MAX_DIM`; document the cap (consistent with v2 problem sizes). Confirm `MAX_DIM`'s value in `mlrs-kernels` when sizing fixtures.

### Pitfall 7: Transcendental intrinsic call form
**What goes wrong:** `x.exp()` / `x.tanh()` fail to compile in `#[cube]`.
**Why:** cubecl exposes these as static associated fns: `F::exp(x)`, `F::tanh(x)`, `F::powf(x, y)`, `F::cos(x)`, `F::log(x)` — not instance methods (per `cubecl_manual/.../mismatched types.md`).
**How to avoid:** Always `F::op(args)`. Ignore rust-analyzer macro-expansion noise inside `#[cube]` ([[rust-analyzer-no-feature-false-positives]]); verify with `cargo build --features cpu`.

## Code Examples

### `any_estimator!` wrapper with a new method (score_samples) — mirror covariance.rs
```rust
// Source: crates/mlrs-py/src/estimators/covariance.rs (read locally) + dispatch.rs macro
crate::any_estimator! {
    any:   AnyKernelDensity,
    algo:  mlrs_algos::density::kernel_density::KernelDensity,   // planner's module path
    unfit: { kernel: u8, bandwidth: f64 /* or an enum/string-resolved */ },
}

#[pymethods]
impl PyKernelDensity {
    fn score_samples(&self, py: Python<'_>, q: &Bound<'_, PyAny>, rows: usize, cols: usize)
        -> PyResult<...>
    {
        let qa = capsule_to_array(q)?;
        let dt = float_dtype(&qa)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => { let qd = validated_f32(as_f32(&qa)?, &mut pool)?;
                    /* self.inner F32 .score_samples(&mut pool, &qd, (rows,cols)) */ }
                FloatDtype::F64 => { crate::capability::guard_f64()?;   // D-04 BEFORE upload
                    let qd = validated_f64(as_f64(&qa)?, &mut pool)?; /* … */ }
            }
        })
        // dtype-suffixed accessors: log_density_f32 / log_density_f64 (mirror covariance_f32/_f64)
    }
}
```

### ScoreSamples trait (D-12) — mirror PartialFit shape in traits.rs
```rust
// Source: crates/mlrs-algos/src/traits.rs PartialFit<F> shape (read locally)
/// Compute per-sample log-density (length-n), NOT Predict semantics (D-12).
pub trait ScoreSamples<F>
where F: Float + CubeElement + Pod {
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;  // length n_samples log-densities
}
```
Add `pub mod`-level re-export in `lib.rs`: `pub use traits::{…, ScoreSamples};`

### New AlgoError variants (extend error.rs in the existing struct-variant style)
```rust
// bandwidth > 0 guard, degree >= 1 guard, kernel-name validation (CONTEXT canonical_refs)
InvalidBandwidth { estimator: &'static str, bandwidth: f64 },   // must be > 0
InvalidDegree    { estimator: &'static str, degree: f64 },       // must be >= 1 (sklearn Interval(Real,1,None))
InvalidKernel    { estimator: &'static str, kernel: String },    // unknown kernel name
// alpha >= 0 already covered by existing InvalidAlpha (reuse it).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gamma='auto'` (1/n_features) string | `gamma=None` → 1/n_features (numeric) | sklearn ≥ 0.22 | KernelRidge uses `None`; match the numeric default (D-05) |
| `bandwidth` scott/silverman = scipy semantics | sklearn 1.x own closed forms (no std factor) | sklearn 1.0+ | Pin the sklearn formulas above, NOT scipy's |

**Deprecated/outdated:** none affecting this phase. sklearn 1.9.0 confirmed as the local oracle version.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Host-side f64 `lgamma`/`log` for KD `log_norm` matches the Cython `lgamma` (same C libm) within the documented KD tolerance | KernelDensity log_norm | KD oracle off by a tiny constant; mitigated by computing log_norm in f64 host-side and a documented (not 1e-5) KD tol |
| A2 | `MAX_DIM` (Cholesky single-cube cap) ≥ the planned KernelRidge oracle `n_samples` | Pitfall 6 | KernelRidge fixture rejected; planner must read `MAX_DIM` from `mlrs-kernels` and size fixtures under it |
| A3 | Poly `degree` kept as real (`F::powf`) rather than integer-only | Pattern 2 / KernelRidge defaults | If planner restricts to integer degree, non-integer-degree sklearn cases would diverge — keep `powf` for full parity |

**Everything else in this research was read directly from the local sklearn 1.9.0 source or the repo and is VERIFIED.**

## Open Questions

1. **Is the D-11 reduce-max rescale needed, or does a plain linear reduce-sum suffice?**
   - What we know: CONTEXT D-11 marks this Claude's discretion; decide from numerical testing.
   - What's unclear: whether f32-on-rocm KD log-density over a large dynamic range overflows/underflows without the max-shift.
   - Recommendation: implement plain reduce-sum first; add reduce-max rescale only if the f32 oracle band fails. Both are SharedMemory-free.

2. **KernelDensity module home (`neighbors/` vs new `density/`).**
   - What we know: CONTEXT leaves it to the planner; sklearn places `KernelDensity` under `neighbors`.
   - Recommendation: a small new `density/` module is cleaner (KD is not a neighbor estimator in mlrs's trait sense); either is fine — file-disjoint, register in `lib.rs`.

3. **Kernel-type representation across the Python boundary.**
   - What we know: `any_estimator!` `unfit` fields are plain Rust types; the `Kernel<F>` enum (D-01) carries params.
   - Recommendation: store sklearn kernel NAME (string/u8 tag) + raw gamma/degree/coef0/bandwidth in `Unfit`, construct the typed `Kernel<F>` at `fit` once `n_features` (for gamma=None) is known.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cubecl | new map kernel | ✓ | 0.10.0 (workspace) | — |
| cubecl Float intrinsics (`exp`/`tanh`/`powf`/`log`/`cos`) | RBF/sigmoid/poly + KD maps | ✓ | static assoc fns on `Float` (cubecl 0.10) | — |
| cubecl-cpu backend (correctness gate) | f64 oracle gate | ✓ | 0.10 | — (cpu is the f64 gate) |
| ROCm (gfx1100) | f32 gate | ✓ | 7.1.1 (f32 only; f64 skips-with-log) | cpu f64 covers f64 |
| numpy + scipy + scikit-learn | `gen_oracle.py` fixture regen ONLY | ✓ (sklearn 1.9.0 local) | sklearn 1.9.0 | /tmp venv (PEP 668) — committed `.npz`, CI never regenerates |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none (cpu(f64)+rocm(f32) gate fully available).

## Validation Architecture

> nyquist_validation = true in config — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (backend prim tests in `crates/mlrs-backend/tests/`; algo tests in `crates/mlrs-algos/tests/`; py tests via existing harness) |
| Config file | none (cargo); oracle fixtures `tests/fixtures/*.npz` via `scripts/gen_oracle.py` |
| Quick run command | `cargo test --features cpu -p mlrs-backend kernel_matrix` (targeted; suite is slow — [[backend-test-suite-slow]]) |
| Full suite command | `cargo test --features cpu` then opportunistic `cargo test --features rocm` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-08 | `kernel_matrix` 4 kernels vs host reference, f32+f64 | unit (prim) | `cargo test --features cpu -p mlrs-backend kernel_matrix` | ❌ Wave 0 |
| PRIM-08 | PoolStats memory gate (live_bytes conserves, peak plateaus) | unit (prim) | `cargo test --features cpu -p mlrs-backend kernel_matrix_memory_gate` | ❌ Wave 0 |
| KERNEL-01 | KernelRidge predict ≤ 1e-5 (4 kernels, multi-target, gamma None+explicit) | integration | `cargo test --features cpu -p mlrs-algos kernel_ridge` | ❌ Wave 0 |
| KERNEL-02 | KernelDensity score_samples ≤ documented tol (6 kernels, scott/silverman) | integration | `cargo test --features cpu -p mlrs-algos kernel_density` | ❌ Wave 0 |
| KERNEL-02 | ScoreSamples<F> trait + length-n log-density shape | unit | `cargo test -p mlrs-algos score_samples` | ❌ Wave 0 |
| PY-06 (share) | PyKernelRidge / PyKernelDensity fit/predict/score_samples, f32/f64 dispatch | py/smoke | existing py test harness | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted `cargo test --features cpu <new_test>` (avoid the full slow suite).
- **Per wave merge:** `cargo test --features cpu -p mlrs-backend` (prim) then `-p mlrs-algos` (estimators).
- **Phase gate:** full `cargo test --features cpu` green + opportunistic `--features rocm` (f32) before `/gsd-verify-work`; every f64 case behind `skip_f64_with_log`.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/kernel_matrix_test.rs` — PRIM-08 values + PoolStats gate (mirror `incremental_svd_test.rs` gate shape)
- [ ] `crates/mlrs-algos/tests/kernel_ridge_test.rs` — KERNEL-01 (mirror `ridge_test.rs`)
- [ ] `crates/mlrs-algos/tests/kernel_density_test.rs` — KERNEL-02 (new)
- [ ] `scripts/gen_oracle.py` extensions: `gen_kernel_matrix`, `gen_kernel_ridge`, `gen_kernel_density` (numpy/sklearn, committed `.npz`)
- [ ] `#[ignore]` scaffold tests in a Wave-0 plan (mirror Phase-7 07-01 scaffold: trait + AlgoError guards + module index + ignored tests + oracle generators)

## Security Domain

> security_enforcement = true (config). Applicable controls below.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | library, no auth surface |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate-before-launch: hyperparameter guards (`bandwidth>0`, `degree≥1`, `alpha≥0`, kernel-name) + geometry checks BEFORE any `unsafe` device launch (mirror ridge.rs/covariance.rs) — untrusted host→estimator boundary (T-04-01-01) |
| V6 Cryptography | no (but RNG hygiene) | Oracle RNG is `numpy.random.default_rng(seed)` host-side (no `OsRng`); this phase adds no RNG (no `OsRng` per the project's ASVS-V6 convention) |

### Known Threat Patterns
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from bad `(rows,cols)`/`rhs` | Tampering | Geometry validated before `ArrayArg::from_raw_parts` (existing prim convention); `cholesky_solve` rejects `n>MAX_DIM` |
| NaN-poisoned result (non-SPD `K+αI`, `F::INFINITY`) | Tampering/DoS | `cholesky_solve` → typed `NotPositiveDefinite`, never NaN; D-11 linear-domain log-sum-exp never emits ±∞ |
| Untrusted hyperparameter (negative bandwidth, degree<1) | Tampering | Typed `AlgoError` at `fit` before any launch |

## Sources

### Primary (HIGH confidence — read locally this session)
- `/home/user/.local/lib/python3.12/site-packages/sklearn/neighbors/_binary_tree.pxi.tp` (sklearn 1.9.0) — `compute_log_kernel`, `log_*_kernel`, `logVn`/`logSn`, `_log_kernel_norm` for all 6 KD kernels (lines 340-476)
- `.../sklearn/neighbors/_kde.py` — score_samples assembly (`logsumexp − log(N)`), scott/silverman bandwidth (lines 223-288), defaults
- `.../sklearn/kernel_ridge.py` — fit/predict/`_get_kernel`, defaults (alpha=1, kernel=linear, gamma=None, degree=3, coef0=1)
- `.../sklearn/linear_model/_ridge.py::_solve_cholesky_kernel` — diagonal-α + `assume_a="pos"` multi-target solve
- `.../sklearn/metrics/pairwise.py` — linear/polynomial/sigmoid/rbf kernel bodies + `gamma=None → 1/X.shape[1]`
- Repo prim signatures: `prims/distance.rs`, `gemm.rs`, `cholesky.rs`, `reduce.rs`, `covariance.rs`; `mlrs-kernels/src/elementwise.rs`; `mlrs-algos/src/{traits.rs,linear/ridge.rs,error.rs}`; `mlrs-py/src/{dispatch.rs,estimators/covariance.rs}`; `scripts/gen_oracle.py`; `incremental_svd_test.rs` (PoolStats gate)
- `cubecl_manual/manual/cubecl/cubecl_error_solution_guide/mismatched types.md` — `F::exp(x)` static-assoc-fn form

### Secondary (MEDIUM)
- scikit-learn pairwise metrics docs (kernel formula confirmation)

### Tertiary (LOW)
- none load-bearing.

## Metadata

**Confidence breakdown:**
- KernelRidge math (formulas, defaults, dual solve): HIGH — read from sklearn 1.9.0 source.
- KernelDensity math (6 kernels, normalization, bandwidth): HIGH — read from sklearn 1.9.0 `_binary_tree.pxi.tp` + `_kde.py`.
- Prim reuse signatures: HIGH — read from repo source.
- CubeCL map-kernel pattern: HIGH — mirrors existing validated `elementwise.rs` kernels; transcendental form confirmed in cubecl manual.
- KD log_norm host-f64 vs Cython lgamma parity: MEDIUM — same C libm, but pinned as A1 to verify in oracle.

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (stable — sklearn 1.9.0 pinned local oracle; cubecl 0.10 workspace-pinned)
