# Phase 10: SGD / Linear-SVM - Research

**Researched:** 2026-06-21
**Domain:** Minibatch-SGD device solver (the one genuinely-new device solver of v2) + four supervised linear estimators (MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR) with a builder-pattern Rust API and pinned-deterministic sklearn oracle.
**Confidence:** HIGH for codebase patterns & GATHER-kernel structure (read from source); HIGH for the builder/PyO3 seam (read from source); MEDIUM for the exact sklearn SGD math (sklearn docs + Cython source via WebFetch, not run-in-session); HIGH for the cpu-MLIR GATHER constraints (cross-confirmed against 5 prior shipped prims).

This is the **highest-risk phase in the project** (`[v2-P4]` research spike). The two named risks — the two-pass GATHER kernel under cpu-MLIR and the pinned-deterministic sklearn oracle — are de-risked below into concrete, validated structures the planner can specify directly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** The **builder pattern is the canonical constructor** for all four Phase-10 estimators (e.g. `MBSGDClassifier::builder().loss(Loss::Hinge).alpha(1e-4).build()?`). This **replaces the `new(positional)` + `with_opts()` convention** used in Phases 4–9 for these estimators.
- **D-02:** The builder is the **going-forward project standard**, not a one-off. Retrofitting the existing low-arity estimators is **deferred work, OUT OF SCOPE for Phase 10**. Phase 10 only *introduces* the standard via the four new estimators.
- **D-03:** The builder **seeds sklearn-exact defaults**. `Estimator::builder().build()` (no setters) MUST reproduce scikit-learn's *default* estimator — `MBSGDClassifier`: `loss='hinge'`, `penalty='l2'`, `alpha=1e-4`, `learning_rate='optimal'`, `max_iter=1000`, `tol=1e-3`; `MBSGDRegressor`: `loss='squared_error'`, `learning_rate='invscaling'`; `LinearSVC`: `loss='squared_hinge'`, `dual='auto'`; `LinearSVR`: `loss='squared_epsilon_insensitive'`. (The pinned-deterministic oracle overrides several of these at test time — `shuffle=False`, fixed `eta0`/schedule, fixed `max_iter`, `tol=0` — but the *default* builder must still equal sklearn's default.)
- **D-04:** Categorical hyperparameters are **Rust enums**, following the existing `KernelKind` precedent (NOT `String`):
  - `Loss::{Hinge, Log, SquaredHinge, SquaredLoss, EpsilonInsensitive, SquaredEpsilonInsensitive}`
  - `Penalty::{L1, L2, ElasticNet}`
  - `LearningRate::{Optimal, InvScaling, Constant, Adaptive}` (incl. Bottou `t0` for `optimal`)
  - Each estimator's builder accepts only the loss variants valid for it.
- **D-05:** Each enum implements **`TryFrom<&str>` using sklearn's spelling**, and this is the **single source of truth** for the string↔enum mapping, living in `mlrs-algos` (NOT duplicated in the Py wrapper). The PyO3 layer accepts the sklearn string, converts via `TryFrom`, and raises a Python **`ValueError`** on an unknown/invalid value.
- **D-06:** **Per-estimator builders**, each exposing **only the knobs valid for that estimator**. Each `build()` **lowers into one shared internal `SgdConfig`/`SgdParams` struct** that the SGD prim consumes — a single prim contract.
- **D-07:** **Solver choice is implicit** (internal, not a user-facing knob): `MBSGDClassifier`/`MBSGDRegressor` always use the new SGD prim; `LinearSVC`/`LinearSVR` resolve `dual='auto'` internally and may reuse the v1 coordinate-descent (CD) solver for the converged optimum.
- **D-08:** **Split validation.** `build() -> Result<Estimator, BuildError>` validates **data-independent** hyperparameters at the earliest point (`alpha >= 0`, `l1_ratio ∈ [0,1]`, `eta0 > 0`, `epsilon >= 0`, valid enum/loss combos). **Data-dependent** checks stay at **`fit() -> AlgoError`**.
- **D-09:** PyO3 surfaces **`build()` errors (and enum `TryFrom` failures) as Python `ValueError` at estimator construction time**. Map `BuildError -> ValueError` in the Py wrapper alongside the existing `algo_err_to_py`.

### Claude's Discretion
- Exact builder method names, the `BuildError` variant set, and the precise `SgdConfig` field layout are left to the planner/researcher, provided they honor D-01…D-09.
- Whether `LinearSVC`/`LinearSVR` builders physically reuse the existing CD estimator type or wrap it is an implementation detail (D-07 only fixes that the choice is internal).

### Deferred Ideas (OUT OF SCOPE)
- **Retrofit existing estimators to the builder pattern** (Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression, KMeans, PCA, spectral family, etc.). Capture for a future cleanup phase.
- **Explicit user-facing solver selection** (force SGD vs CD on LinearSVC/SVR).
- **Typestate compile-time validation** of invalid knob combinations.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **PRIM-10** | A minibatch SGD solver primitive (hinge/log/squared/squared-hinge/epsilon-insensitive losses; learning-rate schedules; GATHER two-pass margin+gradient update, cpu-MLIR-safe) validated standalone on a convex objective before any estimator consumes it. | §"The Two-Pass GATHER Kernel", §"SGD Math", §Pattern 1, §Pitfall 1. Concrete kernel structure given (one-thread-per-coordinate, F/u32 accumulators, ascending scan, no SharedMemory/atomics). Standalone convex test specified in §Validation Architecture. |
| **SGDSVM-01** | `MBSGDClassifier` (hinge/log/squared-hinge; schedules incl. `optimal`) with `predict`/`predict_proba`, pinned deterministic oracle. | §"The Pinned-Deterministic Oracle", §"SGD Math" (per-loss dloss), §"Optimal schedule + Bottou t0". Exact-label hard gate + log loss → proba in §Validation Architecture. |
| **SGDSVM-02** | `MBSGDRegressor` (squared-loss/epsilon-insensitive; `invscaling` default) with `predict`, pinned oracle. | §"SGD Math" (squared_error grad=p-y; epsilon_insensitive subgradient), §"invscaling schedule". |
| **SGDSVM-03** | `LinearSVC` (`loss='squared_hinge'` default, `penalty`, `dual='auto'`, `intercept_scaling`) with `predict`. | §"LinearSVC/LinearSVR + dual='auto' + CD reuse" — recommends CD-reuse path; intercept_scaling synthetic-feature mechanism documented. |
| **SGDSVM-04** | `LinearSVR` (`loss='squared_epsilon_insensitive'` default, `epsilon`) with `predict`. | §"LinearSVC/LinearSVR + dual='auto' + CD reuse". |
</phase_requirements>

## Summary

Phase 10 has two genuinely hard pieces and three pieces that are assembly over validated v1/v2 primitives.

The **hard piece #1** is `prims/sgd.rs`, the minibatch-SGD device solver. The ROADMAP's "two-pass GATHER kernel" maps cleanly onto the 5-prior-prim cpu-MLIR-safe idiom already shipped (dbscan `eps_core_count`, kmeans `centroid_sumcount`, coordinate `col_dot`/`enet_gap`, laplacian `laplacian_map`): **a per-minibatch two-pass structure where pass 1 computes the per-sample margin/residual `p_i = x_i·w + b` for every sample in the batch (one thread per sample — a GATHER over the `d` features), and pass 2 updates each weight coordinate `w_j` by accumulating the loss-gradient contribution over the batch's samples (one thread per coordinate `j` — a GATHER over the `B` batch rows).** Both passes are single-owner GATHERs: pass 1 owns `p[i]`, pass 2 owns `w[j]`. No SharedMemory, no cross-unit atomics, no `F::INFINITY`, no mutable `bool`, no descending shift — exactly the constructs the cubecl-cpu MLIR lowering rejects at launch (verified against the four prior prims that each launched on cpu first-try by construction).

The **hard piece #2** is the **pinned-deterministic sklearn oracle**. SGD is path-dependent: the result depends bit-for-bit on the sample visitation order, the learning-rate schedule at each step, and the `optimal`-schedule `t0` initialization. To match sklearn within tolerance, the oracle must pin `shuffle=False` (so the visitation order is the natural row order, reproducible without matching sklearn's MT19937 permutation), `max_iter` fixed with `tol=0` (so neither side early-stops at a different iterate), `eta0`/schedule fixed, and — critically — the Rust solver must replicate sklearn's exact per-sample update sequence (lazy L2 `wscale` shrink, L1 cumulative-penalty shrink, intercept `update*intercept_decay`) and the `optimal` schedule's Bottou `t0` heuristic. The exact `dloss` subgradient per loss and the `t0` formula are documented below.

The **three assembly pieces**: MBSGDClassifier/Regressor wrap the SGD prim; **LinearSVC/LinearSVR should reuse the v1 coordinate-descent solver, NOT SGD** — sklearn's LinearSVC/SVR are liblinear coordinate-descent (a converged optimum), not SGD, so reusing `coordinate_descent.rs` reproduces the *converged* solution that the SGD path only approaches. The builder API + PyO3 seam is a well-understood extension of the shipped `any_estimator!` macro; the macro emits only the enum, and the `#[pymethods]` are hand-written, so a builder-fronted `Unfit{}` block needs no macro change — only the hand-written `fit` body calls `Estimator::builder()...build()?` instead of `Estimator::new(...)`.

**Primary recommendation:** Build the SGD prim FIRST as a standalone two-pass GATHER kernel validated on a convex objective with a pinned host reference, then wire MBSGD* on it. Route LinearSVC/SVR through the existing CD solver (not SGD). Put `Loss`/`Penalty`/`LearningRate` enums + `TryFrom<&str>` + `SgdConfig` + the four `*Builder` types + `BuildError` in `mlrs-algos`; add `BuildError -> ValueError` to `mlrs-py/src/errors.rs`; the `any_estimator!` macro is reused verbatim.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SGD device solve (per-sample margin + per-coordinate weight update) | `mlrs-kernels` (#[cube] GATHER kernels) + `mlrs-backend` (prims/sgd.rs host orchestration) | — | The n/d-heavy compute is device work; the epoch loop / schedule / convergence is host orchestration (mirrors `prims/coordinate_descent.rs` and `prims/lbfgs.rs`). [VERIFIED: codebase grep] |
| Loss/Penalty/LR-schedule typing + string mapping | `mlrs-algos` (enums + `TryFrom<&str>`, D-05 single source of truth) | — | D-04/D-05 lock this. Mirrors `KernelKind` in `kernel_ridge.rs`. [CITED: 10-CONTEXT.md D-04/D-05] |
| Builder construction + data-independent validation | `mlrs-algos` (`*Builder`, `build() -> Result<_, BuildError>`) | — | D-01/D-08. Estimators stay generic `<F: Float + CubeElement + Pod>`. [CITED: 10-CONTEXT.md D-01/D-08] |
| Shared SGD parameter lowering | `mlrs-algos` (`SgdConfig`/`SgdParams`) | — | D-06: four builders → one prim contract. [CITED: 10-CONTEXT.md D-06] |
| Data-dependent validation + fit orchestration | `mlrs-algos` (estimator `fit`) | `mlrs-backend` (prim launch) | D-08 fit-time geometry/label checks; mirrors logistic.rs label inference. [VERIFIED: codebase] |
| LinearSVC/SVR converged solve | `mlrs-algos` (reuse `linear/coordinate_descent.rs`) | `mlrs-backend` (`prims/coordinate_descent.rs`) | D-07: dual='auto' internal; sklearn uses liblinear CD, so reuse the shipped CD solver, not SGD. [VERIFIED: docs + codebase] |
| Python `#[pyclass]` surface, dtype dispatch, GIL release | `mlrs-py` (`estimators/linear.rs` via `any_estimator!`) | — | PY-06 incremental wrap; reuse shipped binding infra. [VERIFIED: codebase] |
| `BuildError`/`TryFrom` → Python `ValueError` | `mlrs-py` (`errors.rs`) | — | D-09. [CITED: 10-CONTEXT.md D-09] |

## Standard Stack

This phase adds **NO new compute dependency** — the v2 contract is "workspace `Cargo.toml` unchanged; pyo3 stays 0.28" [CITED: ROADMAP.md §v2.0]. Everything is built from already-shipped crates.

### Core
| Library / Module | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` (+ `cubecl::prelude`) | `^0.10` (pinned, workspace) | `#[cube(launch)]` GATHER kernels for the SGD passes | The project's sole device-kernel framework; every prior prim uses it. [VERIFIED: codebase] |
| `mlrs-backend::prims::gemm` | workspace | `p = X·w` margin for the predict path and batch-margin if vectorized | v1 GEMM (wraps `cubek-matmul`); reused by every linear estimator. [VERIFIED: codebase] |
| `mlrs-backend::prims::reduce` | workspace | row/column reductions if batch statistics are needed | v1 reduce prim. [VERIFIED: codebase] |
| `mlrs-backend::prims::coordinate_descent::cd_solve` | workspace | LinearSVC/LinearSVR converged optimum (D-07) | Shipped Phase-5 CD solver; sklearn LinearSVC/SVR are liblinear CD. [VERIFIED: codebase] |
| `mlrs-backend::prims::rng::SplitMix64` | workspace | host shuffle/permutation if `shuffle=True` is ever supported | Promoted Phase-7 host PRNG (never `OsRng`, ASVS-V6). The pinned oracle uses `shuffle=False` so the permutation is identity — but the host SGD epoch loop reuses SplitMix64 for any non-pinned use. [VERIFIED: codebase + MEMORY] |
| `mlrs-algos::traits::{Fit, PartialFit, Predict, PredictLabels, PredictProba}` | workspace | the post-`build()` estimator surface | Unchanged trait contract; the `PartialFit` y-slot was reserved for MBSGD in Phase-7. [VERIFIED: traits.rs] |
| `pyo3` | `0.28` (ABI-pinned) | `#[pyclass]` wrappers | Single linked ABI; arrow-59 pins it. [VERIFIED: codebase/MEMORY] |

### Supporting
| Module | Purpose | When to Use |
|---------|---------|-------------|
| `bytemuck` (`Pod`, `from_bytes`) | host↔`F` reinterpretation for f64 host accumulation of schedule/loss scalars | The `host_to_f64`/`f64_to_host` helper pattern in logistic.rs/coordinate_descent.rs — copy it. [VERIFIED: codebase] |
| `mlrs_core::PrimError` / `mlrs_algos::error::AlgoError` | typed errors; `AlgoError` wraps `PrimError` via `#[from]` | Fit-time data-dependent errors (D-08). [VERIFIED: error.rs] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CD-reuse for LinearSVC/SVR | SGD prim for all four | SGD only *approaches* the optimum; LinearSVC/SVR oracle is sklearn liblinear (converged). CD reproduces the converged optimum exactly; SGD would force a much looser tolerance and a pinned-iteration oracle even for the SVMs. **Recommend CD-reuse (D-07 permits "may reuse").** [VERIFIED: docs+codebase] |
| Two-pass GATHER (pass-per-sample then pass-per-coordinate) | single fused scatter-add kernel | A scatter-add over weight coordinates needs cross-unit atomics, which cubecl-cpu does not lower (5 prior prims avoided exactly this). The two-pass GATHER is mandatory for the cpu launch gate. [VERIFIED: codebase MEMORY] |
| Builder in `mlrs-algos` | builder generated by extending `any_estimator!` | The macro emits only the dtype enum; the builder is a Rust-API concern that belongs in the algos crate (D-01/D-05 single source of truth). Extending the macro would couple the Python ABI to the Rust builder. **Recommend builder in mlrs-algos, macro unchanged.** [VERIFIED: dispatch.rs] |

**Installation:** None. No `cargo add`, no `Cargo.toml` edit. (If the plan finds it genuinely needs a new crate, that is a deviation from the v2 "Cargo unchanged" contract and must be escalated as a `checkpoint:human-verify`.)

**Version verification:** All modules are first-party workspace crates already in-tree; no registry lookup applies.

## Package Legitimacy Audit

> This phase installs **no external packages** (v2 "workspace `Cargo.toml` unchanged" contract). No registry/legitimacy check applies. The only third-party crates touched (`cubecl ^0.10`, `cubek-matmul 0.2.0`, `pyo3 0.28`, `bytemuck`, `arrow 59`) are already pinned and shipped from Phases 1–9.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none — no new packages) | — | N/A |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                        ┌─────────────────── mlrs-py (#[pyclass]) ───────────────────┐
  Python: MBSGDClassifier(loss="hinge", alpha=1e-4, ...)                              │
        │ sklearn-named strings + scalars                                            │
        ▼                                                                            │
   any_estimator! Unfit{ loss:String, penalty:String, alpha:f64, ... }              │
        │  fit(X,y): TryFrom<&str> on each enum string ──► ValueError on bad value   │
        │            Estimator::builder().loss(L).alpha(a)....build()? ─► ValueError │
        └──────────────────────────────┬─────────────────────────────────────────────┘
                                        ▼  (BuildError → ValueError, D-09)
        ┌──────────────────────── mlrs-algos ─────────────────────────────────────┐
        │ Loss/Penalty/LearningRate enums (TryFrom<&str>, D-05 single source)      │
        │ MBSGDClassifierBuilder / ...Regressor / LinearSVC / LinearSVR builders   │
        │   build() ─► validate data-INDEPENDENT params ─► Estimator{ SgdConfig }  │
        │                                                                          │
        │ MBSGD*.fit(): infer classes/geometry (data-DEPENDENT, AlgoError) ───┐    │
        │ LinearSVC/SVR.fit(): resolve dual='auto', intercept_scaling ──┐     │    │
        └───────────────────────────────────────────────────────────────┼─────┼────┘
                                                                         │     │
                   ┌─────────────────────────────────────────────────────┘     │
                   ▼  (converged optimum)                                       ▼ (SGD epochs)
   prims/coordinate_descent::cd_solve         ┌──── prims/sgd.rs (host epoch loop) ────┐
   (LinearSVC/SVR, reuse Phase-5)             │  for epoch in 0..max_iter:             │
                                              │    for batch in minibatches(no shuffle)│
                                              │      PASS 1 (one thread / sample):     │
                                              │        p[i] = Σ_j X[i,j]·w[j] + b   ◄── GATHER over d
                                              │      host: eta = schedule(t)           │
                                              │      PASS 2 (one thread / coord j):     │
                                              │        w[j] -= eta·Σ_i dloss(p_i,y_i)·X[i,j]  ◄── GATHER over B
                                              │        + penalty shrink (L2 lazy / L1) │
                                              │      b -= eta·Σ_i dloss·intercept_decay │
                                              └────────────────────────────────────────┘
                                                  device buffers via BufferPool (PoolStats gate)
```

The diagram traces the primary use case (fit an MBSGDClassifier) input→output by arrows. File-to-implementation mapping is in the Component Responsibilities table below.

### Component Responsibilities

| Capability | File (new or edited) | Notes |
|------------|----------------------|-------|
| SGD GATHER kernels | `crates/mlrs-kernels/src/sgd.rs` (NEW) | Two `#[cube(launch)]` fns: `sgd_margin` (pass 1, per-sample) + `sgd_weight_update` (pass 2, per-coordinate). Feature-free, no SharedMemory/atomics. |
| SGD host orchestration | `crates/mlrs-backend/src/prims/sgd.rs` (NEW) | Epoch loop, minibatch slicing, schedule, penalty, convergence cap, geometry validation before launch. |
| Loss/Penalty/LR enums + TryFrom + SgdConfig | `crates/mlrs-algos/src/linear/sgd_config.rs` (NEW) or in each estimator module | D-04/D-05/D-06 single source of truth. |
| Builders + BuildError | per-estimator module + a shared `BuildError` | D-01/D-08. `BuildError` likely in `mlrs-algos::error`. |
| MBSGDClassifier | `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` (NEW) | Fit + PredictLabels + PredictProba (+ optional PartialFit). |
| MBSGDRegressor | `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` (NEW) | Fit + Predict (+ optional PartialFit). |
| LinearSVC | `crates/mlrs-algos/src/linear/linear_svc.rs` (NEW) | Fit + PredictLabels; internally CD via `coordinate_descent.rs` reuse. |
| LinearSVR | `crates/mlrs-algos/src/linear/linear_svr.rs` (NEW) | Fit + Predict; internally CD. |
| PyO3 wrappers | `crates/mlrs-py/src/estimators/linear.rs` (EDIT) | Four `any_estimator!` blocks + hand-written `#[pymethods]`. |
| BuildError → ValueError | `crates/mlrs-py/src/errors.rs` (EDIT) | `build_err_to_py` alongside `algo_err_to_py`. |
| Module index | `mlrs-algos/src/linear/mod.rs`, `lib.rs`, `mlrs-kernels/src/lib.rs`, `prims/mod.rs` | Wave-0 scaffold owns these (file-disjoint pattern from 07-01/08-01/09-01). |

### Recommended Project Structure

Follow the **Wave-0 scaffold** pattern that every prior v2 phase used (07-01, 08-01, 09-01): one wave front-loads ALL shared-file edits (`error.rs` new `BuildError`, `lib.rs`/`mod.rs` registrations, kernel stubs, the `SgdConfig` + enum definitions, the four estimator struct homes, the PyO3 `any_estimator!` stubs, and the `#[ignore]` Nyquist test scaffolds + oracle generators) so later waves are file-disjoint and parallel-safe.

```
crates/
├── mlrs-kernels/src/sgd.rs              # NEW: sgd_margin + sgd_weight_update #[cube] GATHER kernels
├── mlrs-backend/src/prims/sgd.rs        # NEW: host epoch loop, schedule, penalty, validate-before-launch
├── mlrs-algos/src/linear/
│   ├── sgd_config.rs                    # NEW: Loss/Penalty/LearningRate enums + TryFrom<&str> + SgdConfig + BuildError variants
│   ├── mbsgd_classifier.rs              # NEW: builder + Fit + PredictLabels + PredictProba
│   ├── mbsgd_regressor.rs               # NEW: builder + Fit + Predict
│   ├── linear_svc.rs                    # NEW: builder + Fit + PredictLabels (CD-reuse)
│   └── linear_svr.rs                    # NEW: builder + Fit + Predict (CD-reuse)
└── mlrs-py/src/estimators/linear.rs     # EDIT: four #[pyclass] wrappers
```

### Pattern 1: Two-Pass GATHER SGD kernel (cpu-MLIR-safe)

**What:** The SGD weight update is a two-pass per-minibatch structure where each pass is a single-owner GATHER (no scatter, no atomics).

**When to use:** PRIM-10, every minibatch of every epoch.

**Why two passes:** A naive SGD does `for each sample: p = x·w; w -= eta·dloss·x` — but the per-sample weight write `w -= ...` is a scatter that, if parallelized over coordinates, is fine, but if you try to parallelize the whole sample loop you get a write-write race on `w`. The minibatch formulation breaks this: **within a minibatch, freeze `w`, compute all margins (pass 1), then apply the summed gradient (pass 2).** This is exactly sklearn's *minibatch* semantics (cuML's `MBSGD*` averages the gradient over the minibatch) and it makes both passes embarrassingly parallel GATHERs.

```rust
// Source: PATTERN derived from shipped crates/mlrs-kernels/src/{kmeans.rs,dbscan.rs,coordinate.rs}
// [VERIFIED: codebase — these four kernels launch on cpu(f64) first-try by this exact profile]

/// PASS 1 — per-sample margin GATHER. One UNIT per sample i in the minibatch:
/// p[i] = Σ_j X[i,j]·w[j] + b. A GATHER over the d features (no atomic, no
/// SharedMemory, ascending scan, F/u32 accumulators only).
#[cube(launch)]
pub fn sgd_margin<F: Float + CubeElement>(
    x: &Array<F>,      // B×d minibatch (row-major)
    w: &Array<F>,      // length-d weights (frozen this minibatch)
    bias: F,           // scalar intercept by value (cubecl 0.10 — no ScalarArg)
    p: &mut Array<F>,  // length-B output margins (single-owner per i)
    b: u32,            // batch size
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < b as usize {
        let mut acc = F::new(0.0);
        let mut j = 0u32;
        let base = (i as u32) * d;
        while j < d {
            acc += x[(base + j) as usize] * w[j as usize];
            j += 1u32;
        }
        p[i] = acc + bias;
    }
}

/// PASS 2 — per-coordinate weight-update GATHER. One UNIT per weight coordinate j:
/// grad_j = Σ_i g[i]·X[i,j]  where g[i] = dloss(p[i], y[i]) is precomputed host-side
/// (or in a tiny pass-1.5 kernel). w[j] is owned solely by unit j → no race.
/// Penalty (L2 lazy scale / L1 shrink) applied AFTER the gradient step, per j.
#[cube(launch)]
pub fn sgd_weight_update<F: Float + CubeElement>(
    x: &Array<F>,      // B×d minibatch
    g: &Array<F>,      // length-B per-sample loss gradients dloss(p_i,y_i)
    w: &mut Array<F>,  // length-d weights (in/out, single-owner per j)
    eta: F,            // learning rate this step (host-computed from schedule)
    inv_b: F,          // 1/B minibatch averaging factor
    d: u32,
    b: u32,
) {
    let j = ABSOLUTE_POS;
    if j < d as usize {
        // GATHER the j-th feature's gradient contribution over the batch.
        let mut grad = F::new(0.0);
        let mut i = 0u32;
        while i < b {
            grad += g[i as usize] * x[(i * d + (j as u32)) as usize];
            i += 1u32;
        }
        // averaged minibatch gradient step (no atomic; w[j] is unit-private).
        w[j] = w[j] - eta * grad * inv_b;
        // L2/L1 penalty shrink is a per-j elementwise op done here or in a
        // separate tiny map kernel (see §SGD Math for the exact shrink).
    }
}
```

**Critical cpu-MLIR rules (verified against the 05-02 failure and 4 surviving prims):**
- NO `SharedMemory` (cpu MLIR `run pass` panics at launch).
- NO cross-unit atomics (cpu does not lower them).
- NO `F::INFINITY` constant (raises `From<NativeExpand<F>>` compile error).
- NO mutable `bool` flag (E0283 `NativeExpand` inference failure); use explicitly-typed `let mut x: u32 = 0u32;`.
- NO descending-shift `while q > pos` loop.
- Scalar args passed BY VALUE (`bias: F`, `b: u32`), not `ScalarArg` (cubecl 0.10).
- Bounds-check every thread (`if i < b`) since cubes over-provision.
- **The MEMORY caveat:** these constraints are from cubecl-cpp/cubecl-cpu 0.10 as observed in Phase 5/9. The plan MUST re-verify by actually launching under `--features cpu` (the cpu-launch gate is a success criterion, not just a compile). Do not trust the recalled list blindly — write the kernel SharedMemory-free *by construction* and let the cpu-launch test confirm.

### Pattern 2: Host epoch loop with schedule + convergence cap (mirrors lbfgs.rs / cd_solve)

**What:** `prims/sgd.rs` host orchestration owns the epoch loop, minibatch slicing, schedule evaluation, penalty bookkeeping (the L1 cumulative `u` and the lazy L2 `wscale`), and the `tol`/`max_iter` convergence decision.

**Why host-side:** Identical to `cd_solve` (the convergence test + the cyclic update bookkeeping are host scalars; only the n/d-heavy dot/axpy is device) and `lbfgs_minimize` (the history/line-search is host; only the objective is device). The schedule arithmetic and `t0` init are host f64. [VERIFIED: prims/coordinate_descent.rs, prims/lbfgs.rs]

### Pattern 3: Builder lowering into shared SgdConfig (D-06)

```rust
// Source: PATTERN derived from KernelRidge::new(...) field-store + 10-CONTEXT D-06
pub struct SgdConfig {
    pub loss: Loss,
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,          // only used when penalty == ElasticNet
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub learning_rate: LearningRate,
    pub eta0: f64,
    pub power_t: f64,
    pub epsilon: f64,           // epsilon-insensitive tube width (regressor)
    pub batch_size: usize,      // minibatch size (cuML MBSGD knob)
    pub shuffle: bool,          // pinned oracle sets false
    pub seed: u64,              // SplitMix64 seed when shuffle
}

pub struct MBSGDClassifierBuilder { /* only classifier-valid knobs */ }
impl MBSGDClassifierBuilder {
    pub fn loss(mut self, l: Loss) -> Self { /* reject invalid variant at build() */ }
    pub fn alpha(mut self, a: f64) -> Self { ... }
    // ...
    pub fn build(self) -> Result<MBSGDClassifier<F>, BuildError> {
        // D-08: validate data-INDEPENDENT params here.
        if !(self.alpha >= 0.0) { return Err(BuildError::InvalidAlpha { .. }); }
        if self.penalty == Penalty::ElasticNet && !(0.0..=1.0).contains(&self.l1_ratio) { ... }
        if self.eta0 <= 0.0 && self.learning_rate != LearningRate::Optimal { ... }
        // valid loss-for-estimator combo (classifier ⊄ EpsilonInsensitive, etc.)
        Ok(MBSGDClassifier { config: SgdConfig { .. }, /* device state None until fit */ })
    }
}
```

`builder().build()` with no setters MUST equal sklearn's default estimator (D-03) — encode the sklearn defaults as the builder's field initializers.

### Anti-Patterns to Avoid
- **Scatter-add gradient over weight coordinates.** Needs atomics; cpu-MLIR rejects. Use the two-pass GATHER. [VERIFIED: codebase]
- **Routing LinearSVC/SVR through the SGD prim.** sklearn's are converged liblinear-CD; SGD only approaches the optimum and would force a loose oracle. Reuse `cd_solve`. [VERIFIED: docs]
- **Duplicating the string↔enum mapping in mlrs-py** (like `kernel.rs`'s local `parse_kernel_kind`). D-05 mandates `TryFrom<&str>` in mlrs-algos as the single source. The Py wrapper calls `Loss::try_from(s).map_err(...)`. [CITED: D-05]
- **Validating everything at fit (the Phase 4–9 convention).** D-08 splits: data-independent → `build()`. Don't re-validate `alpha>=0` at fit; do validate geometry/labels at fit.
- **Centering X for the SVMs and forgetting `intercept_scaling`.** LinearSVC/SVR don't center; the intercept is the synthetic-feature weight × `intercept_scaling` (see §SGD Math). [VERIFIED: docs]
- **Trusting the recalled cpu-MLIR constraint list without a cpu-launch test.** The launch gate is mandatory. [CITED: 10-CONTEXT memory-recall caveat]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batched matrix-vector margin | a bespoke GEMV kernel | `prims::gemm` (or the per-sample GATHER in Pattern 1) | GEMM is validated f32/f64 on cpu+rocm; the GATHER is for the fused per-sample dloss path. [VERIFIED] |
| Host PRNG for shuffle | `rand` crate / `OsRng` | `prims::rng::SplitMix64` | ASVS-V6 forbids OsRng; SplitMix64 is the seed-reproducible promoted PRNG. (Pinned oracle uses shuffle=False, so identity order.) [VERIFIED: MEMORY] |
| LinearSVC/SVR converged solve | a new SVM solver | `prims::coordinate_descent::cd_solve` | Phase-5 CD solver already matches sklearn liblinear-class objectives within 1e-5. [VERIFIED] |
| dtype dispatch / GIL / ingress | new PyO3 plumbing | `any_estimator!` + `ingress`/`egress`/`capability`/`errors` | v2 adds ZERO binding infra (07-07/08-05 precedent). [VERIFIED] |
| f64 host accumulation of schedule scalars | ad-hoc casts | the `host_to_f64`/`f64_to_host`/`narrow_to_f` helpers | Copy verbatim from logistic.rs/coordinate_descent.rs. [VERIFIED] |
| Convergence/iteration-cap bookkeeping | new loop logic | mirror `cd_solve`'s `max_iter`/`tol_scaled` + `NotConverged` | Same shape; reuse the `NotConverged` AlgoError variant. [VERIFIED] |

**Key insight:** Phase 10's only irreducibly-new device code is the two `sgd_*` GATHER kernels and the `prims/sgd.rs` epoch loop. Everything else is assembly over shipped, sklearn-validated primitives. The risk is concentrated in (a) the cpu-launch of the new kernels and (b) bit-for-tolerance replication of sklearn's per-sample update sequence — both addressed below.

## SGD Math (the exact sklearn formulas the planner must specify)

> Source: sklearn user guide `modules/sgd.html` + `_sgd_fast.pyx.tp` (per-loss `dloss`) via WebFetch [CITED: scikit-learn.org/stable/modules/sgd.html] [CITED: github.com/scikit-learn/scikit-learn `_sgd_fast.pyx.tp`]. **These are MEDIUM confidence** (fetched, not run-in-session). The plan's Wave-0 oracle generator (run in the /tmp venv against live sklearn) is the authoritative pin — see §Validation Architecture.

**Objective:** `E(w,b) = (1/n)·Σ_i L(y_i, f(x_i)) + α·R(w)` where `f(x)=wᵀx+b`.

**Per-sample update sequence (the exact order, load-bearing for parity):**
```
p      = wᵀx_i + b                          # margin (PASS 1)
g      = dloss(p, y_i)                        # clipped to ±1e12
update = -eta * g                            # (× class_weight × sample_weight, both 1 here)
w      = w * max(0, 1 - (1-l1_ratio)*eta*alpha)   # L2 LAZY shrink (wscale trick)
w      = w + update * x_i                     # gradient step
if fit_intercept: b += update * intercept_decay   # intercept_decay = 1.0 dense, 0.01 sparse
# L1 cumulative penalty: u += l1_ratio*eta*alpha; then l1penalty soft-shrinks each w_j toward 0 by u
```
For the **minibatch** formulation (cuML MBSGD / Pattern 1), the gradient `Σ_i g_i·x_i` is averaged over the batch (`× 1/B`) and `w` is frozen during the batch's margin pass.

**Per-loss `dloss(p, y)` (the subgradient — EXACT):**

| Loss enum | sklearn name | `dloss(p, y)` | Notes |
|-----------|-------------|---------------|-------|
| `Hinge` | `hinge` | `z = p·y; if z <= 1: -y else 0` | threshold 1.0; SVM loss |
| `SquaredHinge` | `squared_hinge` | `z = 1 - p·y; if z > 0: -2·y·z else 0` | LinearSVC default loss |
| `Log` | `log_loss` | `-y / (1 + exp(y·p))` | loss `log(1+exp(-y·p))`; → predict_proba sigmoid |
| `SquaredLoss` | `squared_error` | `p - y` | loss `½(p-y)²`; MBSGDRegressor default |
| `EpsilonInsensitive` | `epsilon_insensitive` | `if y-p > ε: -1; elif p-y > ε: 1; else 0` | SVR loss |
| `SquaredEpsilonInsensitive` | `squared_epsilon_insensitive` | `z = y-p; if z > ε: -2(z-ε); elif z < -ε: 2(-z-ε); else 0` | LinearSVR default loss |
| (`ModifiedHuber`) | `modified_huber` | `z=p·y; if z>=1: 0; elif z>=-1: -2y(1-z); else -4y` | NOT in the locked `Loss` enum (D-04); out of scope unless added |

**Penalty `R(w)` and its application:**
- `L2`: `R = ½‖w‖²`; applied as the lazy `wscale` shrink `w *= max(0, 1-(1-l1_ratio)·eta·alpha)` (for pure L2, `l1_ratio=0`).
- `L1`: `R = ‖w‖₁`; applied as the cumulative-penalty soft-shrink (`u += l1_ratio·eta·alpha`, then `l1penalty`).
- `ElasticNet`: `R = (ρ/2)‖w‖² + (1-ρ)‖w‖₁` with `ρ = 1 - l1_ratio` — combine both shrinks.

**Learning-rate schedules:**
| Schedule | `eta(t)` | Notes |
|----------|----------|-------|
| `Optimal` | `1 / (alpha·(t0 + t - 1))` | default classifier; `t` counts samples 0..n·n_iter |
| `InvScaling` | `eta0 / t^power_t` | default regressor; `power_t=0.25` (reg) |
| `Constant` | `eta0` | fixed |
| `Adaptive` | start `eta0`; divide by 5 on no-improvement; stop when `< 1e-6` | |

**The `optimal` Bottou `t0` (the most error-prone detail — `_init_t`):**
```
typw         = sqrt(1.0 / sqrt(alpha))
initial_eta0 = typw / max(1.0, dloss(-typw, 1.0))
t0           = 1.0 / (initial_eta0 * alpha)      # == optimal_init
# then eta(t) = 1 / (alpha · (t0 + t - 1))
```
[CITED: training knowledge of sklearn `BaseSGD._init_t` + confirmed `optimal_init = 1/(initial_eta0*alpha)` from `_sgd_fast`; **flag [ASSUMED] until the Wave-0 oracle reproduces it** — A1 below]. The `dloss(-typw, 1.0)` uses the *estimator's* loss; for hinge `dloss(-typw,1)= -1` so `max(1, 1)=1` and `initial_eta0=typw`.

**Default parameters (D-03 builder seeds — VERIFIED against sklearn docs):**
| Estimator | loss | penalty | alpha | l1_ratio | max_iter | tol | learning_rate | eta0 | power_t | epsilon |
|-----------|------|---------|-------|----------|----------|-----|---------------|------|---------|---------|
| `MBSGDClassifier` (≈`SGDClassifier`) | hinge | l2 | 1e-4 | 0.15 | 1000 | 1e-3 | optimal | 0.01 | 0.5 | 0.1 |
| `MBSGDRegressor` (≈`SGDRegressor`) | squared_error | l2 | 1e-4 | 0.15 | 1000 | 1e-3 | invscaling | 0.01 | 0.25 | 0.1 |
| `LinearSVC` | squared_hinge | l2 | (C=1.0) | — | 1000 | 1e-4 | (CD, no schedule) | — | — | — |
| `LinearSVR` | squared_epsilon_insensitive | l2 | (C=1.0) | — | 1000 | 1e-4 | (CD) | — | — | 0.0 |
[CITED: scikit-learn.org generated docs for each estimator]

## LinearSVC / LinearSVR + dual='auto' + CD reuse (D-07)

**Recommendation: route LinearSVC/SVR through the v1 coordinate-descent solver, NOT the SGD prim.**

sklearn's `LinearSVC`/`LinearSVR` use **liblinear coordinate descent** to a *converged* optimum — they are not SGD estimators [CITED: scikit-learn.org LinearSVC]. The Phase-5 `cd_solve` already matches sklearn's penalized-CD objectives within 1e-5 (Lasso/ElasticNet exact-sparsity gate). Reusing it reproduces the converged solution the oracle compares against, whereas the SGD prim only *approaches* the optimum and would demand a much looser, pinned-iteration oracle for the SVMs.

**`dual='auto'` resolution (internal, D-07):** `if n_samples < n_features AND optimizer supports (loss, multi_class, penalty): dual=True else dual=False` [CITED: scikit-learn.org]. For the Phase-10 fixtures (small `n`, `n_samples >= n_features`), this resolves to `dual=False` (primal). The estimator resolves this at `fit` and never exposes it as a builder knob.

**`intercept_scaling` (the SVM-specific intercept handling):** When `fit_intercept=True`, sklearn appends a synthetic feature of constant value `intercept_scaling` to each `x`: `x → [x_1..x_n, intercept_scaling]`. The fitted intercept is then `intercept_ = intercept_scaling · synthetic_feature_weight`. Higher `intercept_scaling` reduces the L2 penalty's effect on the intercept [CITED: scikit-learn.org]. The CD-reuse path must replicate this: append the synthetic column, solve, then recover `intercept_ = intercept_scaling · w_last`.

**Loss mapping to the CD objective:** LinearSVC default `squared_hinge` + `penalty=l2` is the standard L2-regularized squared-hinge SVM; LinearSVR default `squared_epsilon_insensitive`. The plan must confirm `cd_solve`'s objective can express the squared-hinge / squared-epsilon-insensitive losses, OR (more likely) implement the SVM-specific dual/primal CD update on top of the shipped CD bookkeeping. **OPEN QUESTION Q1** flags this — `cd_solve` is currently the *Lasso/ElasticNet* soft-threshold CD, which is a *different* per-coordinate update than the SVM hinge-CD. The planner should either (a) extend `cd_solve` with the SVM loss, or (b) add a small SVM-CD path. This is the one genuinely-open design choice for the SVMs.

**`C` ↔ `alpha`:** sklearn LinearSVC uses `C` (inverse reg). The internal penalty maps like LogisticRegression's `l2_reg = 1/(C·n)` precedent (Pitfall 3 in logistic.rs) [VERIFIED: logistic.rs].

## The Builder API + PyO3 Seam (D-01…D-09)

**The `any_estimator!` macro needs NO extension.** [VERIFIED: dispatch.rs] The macro emits ONLY the dtype-dispatch enum (`Unfit{...} | F32(E<f32>) | F64(E<f64>)`); the `#[pymethods]` are hand-written per estimator. So a builder-fronted estimator integrates by:
1. The `Unfit{}` block stores the **sklearn-named strings + scalars** verbatim (e.g. `unfit: { loss: String, penalty: String, alpha: f64, l1_ratio: f64, learning_rate: String, eta0: f64, ... }`), exactly as `kernel.rs` stores `kernel: String`. [VERIFIED: kernel.rs unfit stores `kernel: String`]
2. The hand-written `fit` body: (a) `Loss::try_from(loss_str).map_err(build_err_to_py)?` for each enum string (D-05/D-09 → ValueError), (b) `Estimator::<F>::builder().loss(l).alpha(a)...build().map_err(build_err_to_py)?` (D-09 → ValueError), (c) `est.fit(...).map_err(algo_err_to_py)?` (fit-time AlgoError → ValueError). This replaces the current `Estimator::<F>::new(...)` / `with_opts(...)` call (logistic.rs line 694/702 pattern) with the builder chain.

**`build_err_to_py` (new in errors.rs):** add alongside `algo_err_to_py`:
```rust
// D-09: BuildError (data-independent param / invalid enum string) → ValueError,
// sklearn-faithful (raised at construction time in sklearn; here at the fit()
// boundary because the Unfit arm stores raw strings until the first fit).
pub fn build_err_to_py(err: BuildError) -> PyErr { PyValueError::new_err(err.to_string()) }
```
The enum `TryFrom<&str>` failure type should also map to `ValueError` — either make `BuildError` carry an `UnknownLoss/UnknownPenalty/UnknownLearningRate` variant (so a single `build_err_to_py` covers both), or give the `TryFrom` error its own tiny mapper. Recommend folding both into `BuildError` for one mapping site (mirrors the single-site `algo_err_to_py` rationale in errors.rs).

**`TryFrom<&str>` enums in mlrs-algos (D-05 single source):** mirror `KernelKind` (kernel_ridge.rs) shape; add `impl TryFrom<&str> for Loss { type Error = ...; fn try_from(s)->{ match s {"hinge"=>Ok(Hinge), "log"|"log_loss"=>Ok(Log), ...; other=>Err(...) } } }`. This **replaces** the local `parse_kernel_kind`-style matcher that currently lives in the Py wrapper (kernel.rs lines 50-77) — for Phase 10 the matcher lives in algos, and the Py wrapper just calls `try_from`.

**Where the builder fits the trait surface (unchanged):** after `build()`, the estimator implements the existing `Fit` (+ `PredictLabels`/`PredictProba` for the classifier, `Predict` for the regressors). `fit` still returns `&mut self`. MBSGD* may also implement `PartialFit` (the y-slot was reserved in Phase-7). [VERIFIED: traits.rs PartialFit doc]

## Runtime State Inventory

> Phase 10 is a greenfield additive phase (four new estimators + one new prim), NOT a rename/refactor/migration. This section is included only to record that no runtime-state migration is needed.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys/collections renamed | none |
| Live service config | None | none |
| OS-registered state | None | none |
| Secrets/env vars | None | none |
| Build artifacts | New `mlrs-kernels`/`mlrs-backend`/`mlrs-algos` modules compile fresh; the four per-backend wheels rebuild via maturin (PY-06, Phase-11 final sign-off). No stale egg-info equivalent. | rebuild wheels at PY-06 |

**Nothing found in any migration category — verified: this is additive new-estimator + new-prim work, not a rename.**

## Common Pitfalls

### Pitfall 1: SGD kernel compiles but panics at cpu launch
**What goes wrong:** A SharedMemory/atomic/`F::INFINITY`/mutable-bool/descending-shift kernel passes `cargo build` but the cubecl-cpu MLIR `run pass` panics at launch (the 05-02 failure mode).
**Why it happens:** cubecl-cpu 0.10 lowering doesn't support those constructs; compile and launch are different gates.
**How to avoid:** Write both `sgd_*` kernels SharedMemory-free / atomic-free by construction (Pattern 1). The two-pass GATHER makes every write single-owner.
**Warning signs:** `failed to run pass` at launch; `From<NativeExpand<F>>` compile error (= `F::INFINITY`); E0283 (= un-annotated `let mut flag`).
**Gate:** The cpu-launch test (`--features cpu`) is a SUCCESS CRITERION, not just compile. Add a literal grep gate (`grep -c SharedMemory == 0`, `grep -c INFINITY == 0`) on the new kernel source per the 08-02/09-02 precedent.

### Pitfall 2: SGD oracle non-determinism (the central oracle risk)
**What goes wrong:** Rust SGD and sklearn SGD diverge because the sample visitation order, schedule, or stopping iterate differ.
**Why it happens:** SGD is path-dependent. sklearn shuffles each epoch with its own MT19937 (Rust can't cheaply reproduce that permutation), and `tol`-based early-stop halts the two solvers at different iterates.
**How to avoid:** Pin `shuffle=False` (natural row order — reproducible without matching MT19937), `tol=0` + fixed `max_iter` (both run the SAME number of full epochs to the SAME iterate), fixed `eta0`/schedule. Replicate the EXACT per-sample update sequence (lazy L2 wscale, L1 cumulative shrink, intercept_decay) and the `optimal` `t0` init.
**Warning signs:** weights match for 1 epoch then drift; classifier labels match but `coef_` doesn't.

### Pitfall 3: `optimal` schedule `t0` mismatch
**What goes wrong:** `eta` is off from step 1 because `t0` (the Bottou heuristic) isn't computed identically.
**Why it happens:** `t0 = 1/(initial_eta0·alpha)` with `initial_eta0 = typw/max(1, dloss(-typw,1))`, `typw=sqrt(1/sqrt(alpha))` — easy to get the `dloss(-typw,1)` term or the `t-1` offset (`eta=1/(alpha·(t0+t-1))`) wrong.
**How to avoid:** Pin the exact formula (§SGD Math); the Wave-0 oracle (live sklearn) is the authoritative witness. Consider a `constant` or fixed-`eta0` schedule in the FIRST oracle case to isolate the loss/penalty math from the schedule math, then add an `optimal`-schedule case.
**Warning signs:** A `constant`-schedule case matches but an `optimal`-schedule case doesn't → the bug is in `t0`/`eta(t)`, not the gradient.

### Pitfall 4: Classifier label encoding ±1 vs {0,1}
**What goes wrong:** Hinge/log losses are defined for `y ∈ {-1, +1}`, but training labels are `{0, 1}` (or arbitrary integers).
**Why it happens:** sklearn internally maps the binary label to ±1 for the margin loss. The Rust classifier must do the same remap (and map back for predict, like logistic.rs's `classes_` round-trip).
**How to avoid:** Replicate the logistic.rs `classes_` distinct-sorted-labels pattern; map to ±1 for the loss, store `classes_`, map argmax/sign back to the original id. Exact predicted labels are the HARD gate (recurring gate), so this must be exact.

### Pitfall 5: LinearSVC/SVR centering vs intercept_scaling
**What goes wrong:** Applying the Lasso-style center-then-solve intercept recovery to the SVMs gives a different intercept than sklearn's synthetic-feature `intercept_scaling`.
**Why it happens:** SVMs don't center; the intercept is `intercept_scaling · synthetic_weight`.
**How to avoid:** For LinearSVC/SVR, append the synthetic feature column (value `intercept_scaling`), solve, recover `intercept_ = intercept_scaling · w_last`. Do NOT reuse the cd_fit center-then-solve intercept path.

### Pitfall 6: f64-on-rocm skip + f32 band for weights
**What goes wrong:** A strict 1e-5 assert on f32 SGD weights fails on the rocm gate, or an f64 oracle case runs on rocm where f64 is unregistered.
**Why it happens:** Project gate is cpu(f64) + rocm(f32); f64-on-rocm skips-with-log; f32 SGD accumulates round-off over many steps.
**How to avoid:** `skip_f64_with_log` on every f64 oracle case (recurring gate). Document an f32-on-rocm band for `coef_`/`intercept_` (recurring gate explicitly names "documented f32-on-rocm band for weights"). Exact predicted labels stay the strict hard gate for classifiers (integer, no band).

### Pitfall 7: Pinned oracle defaults vs builder defaults (D-03)
**What goes wrong:** The oracle fixtures are generated with `shuffle=False, tol=0, fixed eta0` but the builder's DEFAULT must equal sklearn's *default* (`shuffle=True, tol=1e-3, optimal`), so a test that constructs `builder().build()` and compares to a pinned fixture will mismatch.
**Why it happens:** D-03 separates the builder default (sklearn default) from the oracle pin (deterministic override).
**How to avoid:** The oracle test constructs the estimator with EXPLICIT pinned setters (`.shuffle(false).tol(0.0).max_iter(K).learning_rate(Constant).eta0(...)`), not the bare default. A SEPARATE test asserts `builder().build()` reproduces sklearn's default-constructor params (the D-03 litmus). (This mirrors the 09-03 lesson: an oracle pinning a DEFAULT-constructor must verify the path reproduces that parameterization.)

## Code Examples

### Per-sample dloss host helper (mirrors the §SGD Math table)
```rust
// Source: derived from sklearn _sgd_fast dloss + the host_to_f64 helper in logistic.rs
// [CITED: _sgd_fast.pyx.tp] — computed host-side per minibatch sample, fed to sgd_weight_update as g[].
fn dloss(loss: Loss, p: f64, y: f64, epsilon: f64) -> f64 {
    match loss {
        Loss::Hinge        => { let z = p*y; if z <= 1.0 { -y } else { 0.0 } }
        Loss::SquaredHinge => { let z = 1.0 - p*y; if z > 0.0 { -2.0*y*z } else { 0.0 } }
        Loss::Log          => { -y / (1.0 + (y*p).exp()) }
        Loss::SquaredLoss  => { p - y }
        Loss::EpsilonInsensitive => {
            if y - p > epsilon { -1.0 } else if p - y > epsilon { 1.0 } else { 0.0 }
        }
        Loss::SquaredEpsilonInsensitive => {
            let z = y - p;
            if z > epsilon { -2.0*(z-epsilon) } else if z < -epsilon { 2.0*(-z-epsilon) } else { 0.0 }
        }
    }
}
```

### Optimal-schedule t0 init (host f64)
```rust
// Source: sklearn BaseSGD._init_t + _sgd_fast optimal_init [ASSUMED — see A1]
fn optimal_t0(loss: Loss, alpha: f64) -> f64 {
    let typw = (1.0 / alpha.sqrt()).sqrt();
    let initial_eta0 = typw / dloss(loss, -typw, 1.0, 0.1).abs().max(1.0);
    1.0 / (initial_eta0 * alpha)            // == optimal_init; eta(t)=1/(alpha*(t0+t-1))
}
```

### TryFrom enum (D-05, single source in mlrs-algos)
```rust
// Source: mirrors KernelKind (kernel_ridge.rs) + replaces kernel.rs parse_kernel_kind
impl TryFrom<&str> for Loss {
    type Error = BuildError;
    fn try_from(s: &str) -> Result<Self, BuildError> {
        match s {
            "hinge" => Ok(Loss::Hinge),
            "log" | "log_loss" => Ok(Loss::Log),
            "squared_hinge" => Ok(Loss::SquaredHinge),
            "squared_error" | "squared_loss" => Ok(Loss::SquaredLoss),
            "epsilon_insensitive" => Ok(Loss::EpsilonInsensitive),
            "squared_epsilon_insensitive" => Ok(Loss::SquaredEpsilonInsensitive),
            other => Err(BuildError::UnknownLoss { value: other.to_string() }),
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `loss='log'` | `loss='log_loss'` | sklearn 1.1 deprecated `'log'`, 1.3 removed | `TryFrom` should accept both spellings (alias `log`→`Log`) for sklearn-faithfulness across versions |
| `loss='squared_loss'` | `loss='squared_error'` | sklearn 1.0 renamed | `TryFrom` accepts both → `SquaredLoss` |
| `dual=True` default (LinearSVC) | `dual='auto'` | sklearn 1.3 | D-03 default is `'auto'`; resolution rule in §LinearSVC |
| `multi_class='ovr'`/`SGDClassifier` OvR | unchanged for these losses | — | MBSGDClassifier binary is the Phase-10 scope; multiclass OvR is per-class SGD (note for fixtures) |

**Deprecated/outdated:**
- `loss='log'`, `loss='squared_loss'` spellings — accept as aliases, emit the modern enum.
- The oracle's `tol`/`n_iter_no_change` early-stopping — deliberately disabled (`tol=0`) for determinism.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `optimal` `t0 = 1/(initial_eta0·alpha)`, `initial_eta0 = typw/max(1,dloss(-typw,1))`, `typw=sqrt(1/sqrt(alpha))`, `eta(t)=1/(alpha·(t0+t-1))` | SGD Math / Code Examples | If the `t-1` offset or the `dloss(-typw,1)` term is wrong, the `optimal`-schedule oracle won't match. **Mitigation:** the Wave-0 live-sklearn oracle is authoritative; pin a `constant`-schedule case first to isolate. |
| A2 | The minibatch gradient is *averaged* (×1/B) per cuML MBSGD semantics | Pattern 1 | If cuML/sklearn sums rather than averages, `eta` is effectively scaled by B. **Mitigation:** confirm against the oracle (a 1-sample minibatch reduces to plain SGD and disambiguates). |
| A3 | `intercept_decay = 1.0` for the dense path (0.01 only for sparse) | SGD Math | Wrong decay → intercept drifts. **Mitigation:** mlrs densifies at ingress, so dense (1.0) is the path; confirm in oracle. |
| A4 | `cd_solve` can be extended (or a thin SVM-CD added) to express squared_hinge/squared_epsilon_insensitive for LinearSVC/SVR | LinearSVC §, Open Q1 | If CD can't reach the SVM optimum, LinearSVC/SVR need their own solver path. **Mitigation:** Open Q1 — planner spikes the CD objective fit early. |
| A5 | The recalled cpu-MLIR constraint list (no SharedMemory/atomics/INFINITY/bool/shift) still holds in the current cubecl 0.10 pin | Pattern 1 / Pitfall 1 | If a constraint relaxed, the kernel is over-constrained (harmless) or under-constrained (launch panic). **Mitigation:** the cpu-launch gate is the authoritative test; write SharedMemory-free by construction. |
| A6 | MBSGDClassifier Phase-10 scope is binary (±1 margin); multiclass is OvR if needed | Pitfall 4 / State of the Art | If multiclass is required, the fixture/label-encoding scope grows. **Mitigation:** ROADMAP success criteria say `predict`/`predict_proba` under a pinned oracle — start binary, confirm scope with discuss-phase. |

## Open Questions

1. **Can `cd_solve` express the LinearSVC/SVR (squared-hinge / squared-epsilon-insensitive) objective, or is a thin SVM-CD path needed?**
   - What we know: `cd_solve` is the Lasso/ElasticNet soft-threshold CD; sklearn LinearSVC/SVR are liblinear hinge/eps-CD — a different per-coordinate update.
   - What's unclear: whether the existing CD bookkeeping can be reused with an SVM loss, or a new per-coordinate update is required.
   - Recommendation: planner spikes the LinearSVC fit against the oracle in an early wave; if CD-reuse can't reach the optimum, add a small SVM-CD update (still host-orchestrated over device dot/axpy). [A4]

2. **Minibatch size default + averaging convention.**
   - What we know: cuML `MBSGD*` exposes `batch_size`; sklearn `SGD*` is pure online (B=1).
   - What's unclear: the exact `batch_size` default and whether the gradient is summed or averaged (A2).
   - Recommendation: pin `batch_size` explicitly in the oracle; a 1-sample batch reduces to plain SGD and disambiguates the averaging. Confirm the cuML default with discuss-phase.

3. **Is MBSGDClassifier multiclass in scope for Phase 10?**
   - What we know: ROADMAP says `predict`/`predict_proba` under a pinned oracle; binary is the minimal scope.
   - What's unclear: whether multiclass OvR fixtures are required.
   - Recommendation: scope binary first (the hard-gate is exact labels); escalate multiclass to discuss-phase if needed. [A6]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cpu (cubecl-cpu MLIR) backend | PRIM-10 cpu-launch gate; f64 oracle | ✓ (project gate) | cubecl ^0.10 | — (this is the primary gate) |
| rocm (HIP gfx1100) backend | f32 weight band gate | ✓ (project gate) | cubecl-hip 0.10 | f64 skips-with-log on rocm |
| scikit-learn (oracle generation) | pinned-deterministic fixtures | ✓ via /tmp venv | 1.9.0 (per prior phases) | regen in /tmp venv (PEP 668) |
| numpy/scipy (oracle) | fixture generation | ✓ via /tmp venv | numpy 2.4.6 / scipy 1.18.0 | /tmp venv |
| cuda backend | compile-only | compile-only | — | opportunistic, not gated |

**Missing dependencies with no fallback:** none — all gates (cpu f64, rocm f32) are available in this environment per Phase-3 bring-up.
**Missing dependencies with fallback:** f64-on-rocm (skips-with-log, the established pattern); cuda (compile-only, opportunistic).

**Oracle generation note:** `scripts/gen_oracle.py` (1874 lines) is the generator; fixtures are committed `.npz` blobs in `tests/fixtures/`. Add `gen_mbsgd_classifier`, `gen_mbsgd_regressor`, `gen_linear_svc`, `gen_linear_svr` generators following the existing `gen_logistic`/`gen_elastic_net` shape (seed42, both dtypes, pinned `shuffle=False`/`tol=0`/fixed `max_iter`/explicit schedule). Regen in the /tmp venv in ISOLATION (per the 08-01/09-01 lesson — don't churn other phase blobs). [VERIFIED: gen_oracle.py + MEMORY oracle-fixture-regen]

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`). This section maps each success criterion to an automated test.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (integration tests in `crates/*/tests/`, AGENTS.md §2 — never in-source `#[cfg(test)] mod tests`) + Python pytest smoke for the PyO3 surface |
| Config file | none — `cargo test` per crate; tests gated by `--features cpu` / `--features rocm` |
| Quick run command | `cargo test -p mlrs-backend --features cpu sgd` (targeted; full cpu suite is slow ~6min per MEMORY) |
| Full suite command | `cargo test -p mlrs-backend -p mlrs-algos --features cpu` (background it; targeted post-merge gates per MEMORY) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-10 | SGD prim minimizes a convex objective standalone (host ref) | unit | `cargo test -p mlrs-backend --features cpu sgd_test::sgd_convex_objective` | ❌ Wave 0 |
| PRIM-10 | SGD kernels LAUNCH on cpu (not just compile) | unit (launch) | `cargo test -p mlrs-backend --features cpu sgd_test::sgd_cpu_launch` | ❌ Wave 0 |
| PRIM-10 | SharedMemory/INFINITY/atomic grep gate on new kernel source | static | `grep -c SharedMemory crates/mlrs-kernels/src/sgd.rs` == 0 | ❌ Wave 0 |
| PRIM-10 | PoolStats memory gate (bounded alloc, peak plateaus, read_backs bounded) | unit | `cargo test -p mlrs-backend --features cpu memory_gate_test::memory_gate_sgd_bounded` | ❌ Wave 0 |
| SGDSVM-01 | MBSGDClassifier weights match pinned sklearn oracle (f64 strict, f32 band) | oracle | `cargo test -p mlrs-algos --features cpu mbsgd_classifier_test::oracle` | ❌ Wave 0 |
| SGDSVM-01 | MBSGDClassifier predict labels EXACT (hard gate) | oracle | `..::mbsgd_classifier_test::exact_labels` | ❌ Wave 0 |
| SGDSVM-01 | MBSGDClassifier(log) predict_proba within tol | oracle | `..::mbsgd_classifier_test::proba` | ❌ Wave 0 |
| SGDSVM-02 | MBSGDRegressor predict matches pinned oracle | oracle | `..::mbsgd_regressor_test::oracle` | ❌ Wave 0 |
| SGDSVM-03 | LinearSVC predict labels exact (CD-reuse, dual='auto', intercept_scaling) | oracle | `..::linear_svc_test::oracle` | ❌ Wave 0 |
| SGDSVM-04 | LinearSVR predict matches oracle | oracle | `..::linear_svr_test::oracle` | ❌ Wave 0 |
| D-03 | `builder().build()` reproduces sklearn default params (litmus) | unit | `..::*_test::default_matches_sklearn` | ❌ Wave 0 |
| D-08/D-09 | `build()` rejects bad data-independent params → BuildError; PyO3 → ValueError | unit + pytest | `..::*_test::build_rejects_bad_alpha` + py smoke | ❌ Wave 0 |
| D-05 | enum `TryFrom<&str>` accepts sklearn spellings, rejects unknown | unit | `..::sgd_config_test::try_from` | ❌ Wave 0 |

### How each success criterion is validated (ROADMAP criteria)
- **Criterion 1 (standalone SGD prim):** a convex-objective test fits the prim on a host-generated convex problem (e.g. squared loss on a known-optimum linear system) and asserts the SGD iterate converges to the host closed-form optimum within tolerance — BEFORE any estimator is wired (primitive-first gate). Plus the cpu-launch gate + the SharedMemory grep gate + the PoolStats memory gate.
- **Criterion 2/3 (MBSGD* pinned oracle):** fixtures generated from sklearn `SGDClassifier`/`SGDRegressor` with `shuffle=False, tol=0, fixed max_iter, fixed eta0/schedule`; Rust constructs the estimator with the SAME explicit pins (NOT the bare default — Pitfall 7), fits, and compares `coef_`/`intercept_` (f64 strict 1e-5, documented f32-on-rocm band) and — for the classifier — EXACT predicted labels (hard gate) + `predict_proba` (log loss) within tol.
- **Criterion 4 (LinearSVC/SVR):** fixtures from sklearn `LinearSVC`/`LinearSVR` (converged, default `dual='auto'`); Rust CD-reuse path compares `predict` (exact labels for SVC) + `coef_`/`intercept_` (with the `intercept_scaling` recovery).
- **Recurring gates:** `skip_f64_with_log` on every f64 oracle case; f32-on-rocm weight band; exact-labels hard gate; per-prim PoolStats gate; cpu-launch gate.

### Sampling Rate
- **Per task commit:** `cargo test -p <crate> --features cpu <targeted test>` (the new test only).
- **Per wave merge:** `cargo test -p mlrs-backend -p mlrs-algos --features cpu <phase tests>` (targeted, NOT the full slow suite — MEMORY: full suite ~6min and full `cargo test --features cpu` can exhaust disk; run targeted + cargo clean to recover).
- **Phase gate:** targeted phase suite green on cpu(f64); rocm(f32) test target builds + f32 cases green, f64 skips-with-log; before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/sgd_test.rs` — PRIM-10 convex objective + cpu launch + grep gate
- [ ] `crates/mlrs-algos/tests/{mbsgd_classifier,mbsgd_regressor,linear_svc,linear_svr}_test.rs` — oracle cases
- [ ] `crates/mlrs-algos/tests/sgd_config_test.rs` — TryFrom + build() validation
- [ ] `crates/mlrs-backend/tests/memory_gate_test.rs` — add `memory_gate_sgd_bounded`
- [ ] `scripts/gen_oracle.py` — add `gen_mbsgd_classifier/regressor/linear_svc/linear_svr` (committed `.npz`, /tmp venv, isolated regen)
- [ ] Wave-0 scaffold: `BuildError` in `error.rs`; `sgd.rs` kernel/prim stubs; four estimator struct homes; enum + `SgdConfig` definitions; PyO3 `any_estimator!` stubs + `build_err_to_py`; `#[ignore]` Nyquist scaffolds asserting fixture-load+shape only

## Security Domain

> security_enforcement is enabled (config.json), ASVS level 1. This phase is a numerical compute library with a PyO3 boundary; the relevant ASVS surface is input validation and the absence of unsafe RNG.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (no auth surface) |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | **yes** | Validate-before-launch: `build()` rejects bad data-independent params (`alpha>=0`, `l1_ratio∈[0,1]`, `eta0>0`, valid enum); `fit()` rejects bad geometry/labels (`x.len()==n*d`, label integrality) as typed `AlgoError`/`BuildError` BEFORE any unsafe device launch (the established ASVS-V5 pattern in every prior prim). Untrusted hyperparameters from the Python boundary become typed `ValueError`, never an out-of-bounds device read. |
| V6 Cryptography | **yes (RNG only)** | Use `prims::rng::SplitMix64` for any shuffle — NEVER `OsRng`/`rand` (the established ASVS-V6 host-PRNG rule). Pinned oracle uses `shuffle=False` (identity), so no RNG in the gated path. Seed is a `u64` builder knob (reproducible). |
| V12 (deps) | yes | No new external dependency (v2 Cargo-unchanged contract); `default-features=false` pins intact. |

### Known Threat Patterns for {Rust compute lib + PyO3 boundary}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from a malformed shape/hyperparameter | Tampering | Validate geometry + params before the `unsafe` launch (V5); typed error, not a panic |
| DoS via unbounded iteration (huge `max_iter`) | Denial of Service | `max_iter` is a finite cap; `NotConverged` surfaced at the cap (T-05-10-03 precedent) |
| Panic across the PyO3 boundary on a device launch failure | Denial of Service | Capture prim errors into a typed `AlgoError`/`PyValueError`, never an unwinding panic across FFI (the logistic.rs `prim_err` slot + `ScratchGuard` RAII precedent) |
| Non-reproducible RNG (OsRng) leaking entropy / breaking determinism | Tampering | SplitMix64 seeded PRNG only (V6); never OsRng |

## Sources

### Primary (HIGH confidence — read from source this session)
- `crates/mlrs-kernels/src/{dbscan.rs, kmeans.rs, coordinate.rs}` — the cpu-MLIR-safe GATHER kernel idiom (Pattern 1, Pitfall 1)
- `crates/mlrs-algos/src/linear/{logistic.rs, coordinate_descent.rs}` — host orchestration, `host_to_f64`, classes_ remap, center-then-solve, NotConverged
- `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs` — `KernelKind` enum precedent (D-04)
- `crates/mlrs-algos/src/traits.rs` — Fit/PartialFit/Predict/PredictLabels/PredictProba contract
- `crates/mlrs-py/src/{dispatch.rs, estimators/linear.rs, estimators/kernel.rs, errors.rs}` — `any_estimator!` macro, builder integration site, `algo_err_to_py`, the `parse_kernel_kind` pattern D-05 replaces
- `crates/mlrs-algos/src/error.rs` — AlgoError variant set (where BuildError joins)
- `.planning/STATE.md` — the cpu-MLIR landmine history (05-02), the cpu(f64)+rocm(f32) gate, oracle regen lessons
- `.planning/{ROADMAP.md, REQUIREMENTS.md, phases/10-sgd-linear-svm/10-CONTEXT.md}`

### Secondary (MEDIUM confidence — WebFetch this session)
- scikit-learn.org/stable/modules/sgd.html — SGD objective, update rule, schedules, penalty forms
- github.com/scikit-learn/scikit-learn `sklearn/linear_model/_sgd_fast.pyx.tp` — per-loss `dloss` subgradients, per-sample update sequence (lazy wscale, L1 cumulative, intercept_decay)
- scikit-learn.org generated docs for `SGDClassifier`/`SGDRegressor`/`LinearSVC`/`LinearSVR` — defaults, `dual='auto'`, `intercept_scaling`

### Tertiary (LOW / ASSUMED — flagged in Assumptions Log)
- sklearn `BaseSGD._init_t` `optimal` `t0` formula (A1 — training knowledge, oracle is authoritative)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all first-party workspace crates read from source; no new deps (v2 contract).
- Architecture (two-pass GATHER, builder/PyO3 seam): HIGH — derived directly from 5 shipped prims + the shipped macro/estimator/error code.
- SGD math (dloss, schedules, defaults): MEDIUM — sklearn docs + Cython source via WebFetch, not run-in-session; the Wave-0 live-sklearn oracle is the authoritative pin.
- `optimal` t0 init: LOW/ASSUMED (A1) — flagged; isolate with a constant-schedule oracle case first.
- Pitfalls / cpu-MLIR constraints: HIGH — cross-confirmed against the 05-02 documented failure + 4 surviving prims; the cpu-launch gate is the final authority.

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (stable — sklearn SGD math is long-stable; cubecl pin is frozen at ^0.10; re-check if the cubecl pin moves or sklearn major bumps)
