# Phase 8: Kernel Family - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver kernel-based regression and density estimation assembled on v1's
validated primitive base, plus one new keystone **device** primitive and one
new trait. Scope is fixed by ROADMAP.md Phase 8 success criteria and
REQUIREMENTS PRIM-08, KERNEL-01, KERNEL-02 (and PY-06's incremental
per-phase share).

**In scope:**
- `prims/kernel_matrix.rs` — keystone kernel-matrix primitive: one small
  elementwise map over the v1 distance/Gram prims producing
  **linear / RBF / polynomial / sigmoid** kernels. Feature-free, **NO
  SharedMemory, NO atomics**; large `n×n` operands kept in global memory
  (gfx1100 LDS ≤ 65536 B). Validated standalone vs a host reference for f32
  and f64, with its build-failing PoolStats memory gate. This is the seam
  Phase 9 (spectral affinity = `kernel_matrix(Rbf)`) and future kernel-SVM
  reuse. (PRIM-08)
- `KernelRidge` — dual-coefficient solve of `(K + αI)` via the v1 Cholesky
  prim; kernels linear/rbf/polynomial/sigmoid with `gamma`/`degree`/`coef0`;
  `predict` matching scikit-learn ≤ 1e-5. (KERNEL-01)
- `KernelDensity` — kernels + `bandwidth`; `score_samples` returns log-density
  via a numerically-stable log-sum-exp, matching scikit-learn within a
  documented tolerance. (KERNEL-02)
- New `ScoreSamples<F>` trait alongside `Fit`/`Predict`/`Transform`/
  `PartialFit` in `crates/mlrs-algos/src/traits.rs`; `KernelDensity`
  implements it (length-`n` log-densities, NOT `Predict` semantics).
- PyO3 wrappers for both estimators reusing the shipped `any_estimator!`
  machinery (`score_samples` is the new exposed method; PY-06's final
  cross-cutting sign-off remains Phase 11).

**Out of scope (deferred / other phases):**
- `kernel='precomputed'` and per-target `alpha` arrays for KernelRidge.
- Tree-based KD acceleration (BallTree/KDTree) — brute-force exact only.
- Kernel SVC/SVR (SMO) — explicit v3 backlog (REQUIREMENTS §v3 notes).
- Graph-Laplacian / spectral estimators (Phase 9, hard-depends on this prim).

</domain>

<decisions>
## Implementation Decisions

### kernel_matrix primitive API (the keystone seam)
- **D-01:** Kernel selection is a **typed `Kernel<F>` enum with per-variant
  params** — `Linear`, `Rbf { gamma }`, `Poly { gamma, degree, coef0 }`,
  `Sigmoid { gamma, coef0 }`. One value carries everything; the prim matches on
  it. Type-safe and the cleanest reusable seam for Phase 9 / kernel-SVM.
- **D-02:** **Always compute the full general `K(X, Y)`** (`rows_x × rows_y`).
  Training `K(X, X)` simply passes `Y = X`; predict needs the general form
  anyway. No symmetry/upper-triangle special-case — one branch-free code path;
  the ~2× redundant compute on the symmetric case is acceptable at v2 sizes.
- **D-03:** The prim is **self-contained** — it internally dispatches the base
  op (v1 `distance` squared-euclidean for RBF; v1 `gemm` `XYᵀ` Gram for
  linear/poly/sigmoid) and then applies the elementwise map kernel. Callers
  (KernelRidge, spectral, future SVM) just call
  `kernel_matrix(X, Y, Kernel::…)`. Keep the map kernel feature-free,
  SharedMemory-free, no atomics, F/u32 accumulators only (cpu-MLIR landmines).

### KernelRidge fidelity
- **D-04:** **Support multi-target `y`** (`n_samples × n_targets`). `dual_coef_`
  is solved as `(K + αI)⁻¹ Y` with `Y` supplied as `n_targets` RHS columns —
  v1 `cholesky_solve` already takes multiple `rhs` columns, so this is near-free
  and gives full sklearn parity. `dual_coef_` is `n_samples × n_targets`.
- **D-05:** **Mirror sklearn's `gamma=None` semantics exactly:** `gamma=None →
  1/n_features` (computed at fit from `n_features`) for rbf/poly/sigmoid;
  explicit `gamma` used as-is. Oracle pins BOTH the None-default and the
  explicit-gamma paths to ≤ 1e-5.
- **D-06:** **Scalar `alpha` + the 4 computed kernels only** (linear/rbf/poly/
  sigmoid with `gamma`/`degree`/`coef0`). NO `kernel='precomputed'`, NO
  per-target `alpha` array. KernelRidge has **no intercept / no centering**
  (sklearn KernelRidge fits on raw data) — do not add one.

### KernelDensity kernel scope & architecture
- **D-07:** Ship **all 6 sklearn KD kernels** — gaussian, tophat, epanechnikov,
  exponential, linear, cosine — for full parity. Each is a small elementwise map
  of (raw) distance plus a dimension-dependent normalization constant; once the
  distance + normalize + log-sum-exp harness exists, the extra 5 are cheap. The
  compact-support kernels (tophat/epanechnikov/linear/cosine) yield exactly-zero
  density outside `bandwidth` → log(0) = −∞; this is handled **in the linear
  (non-log) domain, never with `F::INFINITY` in a kernel** (see D-11).
- **D-08:** KD is a **distinct kernel family** — its kernels are functions of
  raw euclidean **distance** with dimension-dependent normalization, NOT the
  prim's dot-product kernels. KD therefore **composes directly over the v1
  `distance` prim** (squared-euclidean, sqrt as needed) + the density-kernel
  map + normalization + log-sum-exp. KD does **NOT** route through
  `kernel_matrix.rs`. The `kernel_matrix` prim serves **KernelRidge + spectral
  (Phase 9)** only. (This clarifies ROADMAP SC-1's loose "serving … KernelDensity"
  wording — the shared base underneath both is the v1 distance prim.)
- **D-09:** Support **numeric `bandwidth` (float > 0) AND the `'scott'` /
  `'silverman'` auto-bandwidth string rules** (host-side closed-form formulas
  over `n_samples`/`n_features`/feature std). Fuller sklearn 1.x parity; pin
  the exact formulas from sklearn source during planning.

### KernelDensity oracle & numerics
- **D-10:** **Oracle = sklearn `KernelDensity` forced exact** (`rtol=0.0,
  atol=0.0` → the tree falls back to exact summation), keeping the "oracle =
  scikit-learn" rule intact. mlrs computes **brute-force exact** pairwise
  log-density and matches within the documented tolerance.
- **D-11:** **Device-side log-sum-exp**, device-resident, via the v1 `reduce`
  prim. **CRITICAL implementation constraint:** operate in the **linear kernel
  domain** — kernel values are non-negative with exact `0` for out-of-support
  points, so zeros are summed directly and **never become `F::INFINITY`**
  (the cpu-MLIR landmine per [[cubecl-cpu-no-shared-memory]]). For numerical
  stability over the large dynamic range, an optional **reduce-max rescale**
  (divide by the per-query max kernel value before summing, add `log(max)`
  back) gives the max-shift effect without ever touching ±∞; apply a single
  `log` at the end. The large-dynamic-range f32-on-rocm risk is covered by the
  documented log-density tolerance band (Claude's discretion, below).

### ScoreSamples trait (settled by precedent)
- **D-12:** Add `ScoreSamples<F>` next to `Fit`/`Predict`/`Transform`/
  `PartialFit` in `traits.rs`, same `<F: Float + CubeElement + Pod>` bound,
  same `pool`/`DeviceArray`/explicit-`(rows,cols)` device-resident convention
  (mirrors the Phase-7 `PartialFit<F>` addition, [[oracle-fixture-regen-needs-venv]]
  harness unchanged). Returns length-`n` log-densities; `KernelDensity`
  implements it. This is the KERNEL-02 / PY-06 `score_samples` contract.

### Claude's Discretion
- Exact f32-on-rocm tolerance bands for **KernelRidge predictions** and
  **KernelDensity log-density** (large dynamic range) — follow the v1
  per-family documented-band precedent (ROADMAP Phase 8 "Recurring gates");
  f64 stays strict (≤ 1e-5 / documented KD tolerance), gated by
  `skip_f64_with_log`.
- Whether the D-11 reduce-max rescale is actually needed (vs a plain linear
  reduce-sum) — decide from numerical testing during planning/execution.
- The precise `'scott'`/`'silverman'` formulas — pin from sklearn source.
- Whether the elementwise map is a single kernel parameterized by a kernel-type
  uniform vs one kernel per variant — planner's call (keep it
  SharedMemory/atomics-free either way).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` — Phase 8 "Kernel Family" success criteria, recurring
  gates, and the "research-phase can be skipped" note (kernel-matrix is a known
  elementwise map; no research flag).
- `.planning/REQUIREMENTS.md` — PRIM-08, KERNEL-01, KERNEL-02, PY-06 (exact
  wording, incl. the "within tolerance" vs "≤ 1e-5" distinction between
  KernelDensity and KernelRidge).
- `.planning/PROJECT.md` — milestone v2.0 goal, constraints, Key Decisions
  table (oracle = sklearn, gate = cpu(f64)+rocm(f32), primitive-first).
- `.planning/seeds/v2-breadth-roadmap.md` — v2 family/prim mapping.
- `.planning/research/SUMMARY.md` — v2 project research backing the roadmap.
- `.planning/phases/07-covariance-projection/07-CONTEXT.md` — prior-phase
  precedents carried forward (trait-addition shape, PoolStats gate, f32 bands,
  PyO3 `any_estimator!` reuse).

### Reusable primitive & estimator code (v1 + Phase 7, validated)
- `crates/mlrs-backend/src/prims/distance.rs` — pairwise squared-euclidean
  (GEMM-expansion, clamped ≥ 0). Base for RBF (kernel_matrix) AND for all 6 KD
  kernels (D-03/D-08); exposes optional sqrt at the boundary.
- `crates/mlrs-backend/src/prims/gemm.rs` — `XYᵀ` Gram for linear/poly/sigmoid
  base (D-03).
- `crates/mlrs-backend/src/prims/cholesky.rs` — SPD `(K + αI)` dual solve;
  `cholesky_solve` already takes multi-column `rhs` (enables D-04 multi-target)
  and an `out` working-buffer reuse path.
- `crates/mlrs-backend/src/prims/reduce.rs` — device max + sum for the D-11
  device-side log-sum-exp.
- `crates/mlrs-backend/src/prims/covariance.rs` — centered-Gram precedent for
  the in-place-scale / GEMM-output-buffer-reuse idiom.
- `crates/mlrs-backend/src/prims/mod.rs` — register the new `kernel_matrix`
  module.
- `crates/mlrs-algos/src/traits.rs` — `Fit`/`Predict`/`Transform`/`PartialFit`
  surface to extend with `ScoreSamples<F>` (D-12).
- `crates/mlrs-algos/src/linear/ridge.rs` — closed-form `(XᵀX + αI)` Cholesky
  solve skeleton KernelRidge's dual solve mirrors.
- `crates/mlrs-algos/src/error.rs` — `AlgoError` (extend with Phase-8
  hyperparameter guards: bandwidth > 0, degree ≥ 1, kernel-name validation).
- `crates/mlrs-py/src/dispatch.rs` + `crates/mlrs-py/src/estimators/` — the
  `any_estimator!` Unfit/F32/F64 machinery + dtype-suffixed accessors,
  `py.detach` GIL release, `guard_f64()`; add a `kernel.rs` estimator module
  here (mirrors `covariance.rs` / `decomposition.rs`).
- `tests/` + `crates/*/tests/` + `gen_oracle.py` — committed-`.npz` oracle
  harness ([[oracle-fixture-regen-needs-venv]]: regen needs a `/tmp` venv with
  numpy+scipy+sklearn, PEP 668).

### Kernel / build guidance
- `/home/user/Documents/workspace/cubecl_manual/` — CubeCL manuals (the
  Phase-7 `manual/Cubecl/` subpath has moved; use the dir root). Relevant for
  the one new device kernel (`kernel_matrix` elementwise map).
- `/home/user/Documents/workspace/cintx/docs/cubecl_error_guideline.md` —
  CubeCL error guideline referenced by `AGENTS.md`.
- `AGENTS.md` — tests separated from source; consult the CubeCL error guideline
  on any build error; generics-over-float protocol.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **v1 distance prim** (`prims/distance.rs`): squared-euclidean is the shared
  base for BOTH the RBF kernel_matrix branch AND all 6 KernelDensity kernels —
  KD does not need `kernel_matrix.rs` at all (D-08).
- **v1 GEMM** (`prims/gemm.rs`): `XYᵀ` Gram is the base for the linear / poly /
  sigmoid kernel_matrix branches (D-03).
- **v1 Cholesky** (`prims/cholesky.rs`): `cholesky_solve` already accepts a
  multi-column RHS, so KernelRidge multi-target (D-04) needs zero new solver
  work; also has an `out`-buffer reuse path for the `n×n` Gram.
- **v1 reduce prim** (`prims/reduce.rs`): max + sum for the D-11 device-side
  log-sum-exp.
- **PyO3 `any_estimator!` machinery** (`mlrs-py/src/dispatch.rs` +
  `estimators/*.rs`): v2 adds zero binding infra; KernelRidge/KernelDensity get
  the Unfit/F32/F64 enum + hand-written `#[pymethods]` with dtype-suffixed
  accessors, `py.detach` GIL release, `guard_f64()`. `score_samples` is the one
  new method to expose.

### Established Patterns
- **Primitive-first discipline:** land + standalone-validate `kernel_matrix.rs`
  (f32 + f64 vs host reference) with its build-failing PoolStats memory gate
  BEFORE wiring KernelRidge/KernelDensity (mirrors Phase 7's rng/incremental_svd
  gating).
- **cpu-MLIR-safe kernels:** feature-free, SharedMemory-free, no cross-unit
  atomics, F/u32 accumulators only; **never `F::INFINITY`** — the D-11
  linear-domain log-sum-exp exists specifically to avoid it
  ([[cubecl-cpu-no-shared-memory]]).
- **LDS budget:** large `n×n` kernel operands stay in global memory
  (gfx1100 LDS ≤ 65536 B); LDS-budget audit on any SharedMemory tile (there
  should be none).
- **f64 gated by `skip_f64_with_log`** (cpu runs f64, rocm skips); documented
  per-family f32-on-rocm bands for KernelRidge predictions & KernelDensity
  log-density ([[rocm-is-runnable-gpu-gate]]).
- **Backend test suite is slow** ([[backend-test-suite-slow]]) — run targeted
  gates for `kernel_matrix_test` / KernelRidge / KernelDensity; background the
  full run.

### Integration Points
- `crates/mlrs-backend/src/prims/mod.rs` — register `kernel_matrix`.
- `crates/mlrs-algos/src/traits.rs` — add `ScoreSamples<F>`.
- `crates/mlrs-algos/src/` — new estimator module group for KernelRidge /
  KernelDensity (file-disjoint, register in `lib.rs`; e.g. a `kernel_ridge`
  module + KernelDensity under a neighbors/density home — planner's call).
- `crates/mlrs-py/src/estimators/` — new `kernel.rs` wrapping both estimators.

</code_context>

<specifics>
## Specific Ideas

- KernelRidge fits raw data with **no intercept and no centering** (unlike
  sklearn `Ridge`) — a pure dual solve; do not add centering (D-06).
- KD's gaussian kernel is mathematically rbf-like, but it is still implemented
  through the **distance prim + KD normalization**, NOT `kernel_matrix(Rbf)`,
  to keep all 6 KD kernels on one consistent code path (D-08).
- KernelDensity oracle is sklearn at `rtol=0, atol=0` (exact), small `n` so the
  comparison is deterministic (D-10).
- Multi-target KernelRidge falls out of the existing multi-RHS Cholesky solve —
  pin a 2-target oracle case alongside the single-target one (D-04).

</specifics>

<deferred>
## Deferred Ideas

- **`kernel='precomputed'` + per-target `alpha` array** (KernelRidge) — rare at
  v2; out of KERNEL-01's gated surface.
- **Tree-based KD acceleration** (BallTree/KDTree with rtol/atol) — brute-force
  exact is correct and sufficient at v2 sizes.
- **Kernel SVC/SVR (SMO)** — explicit v3 hard-algorithm backlog
  (`.planning/notes/v3-hard-algorithm-backlog.md`).
- **A bespoke fused kernel-matrix-then-reduce device kernel** — only if the
  compose-over-distance/GEMM path proves a memory/perf problem (default: no).
- None outside phase scope surfaced during discussion — stayed within Phase 8.

</deferred>

---

*Phase: 8-kernel-family*
*Context gathered: 2026-06-21*
