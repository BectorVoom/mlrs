# Phase 7: Covariance & Projection - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver covariance estimators and projection transformers assembled on v1's
validated primitive base, plus two new **host-side** primitives and one new
trait. Scope is fixed by ROADMAP.md Phase 7 success criteria and
REQUIREMENTS PRIM-06, PRIM-07, COV-01, COV-02, DECOMP-03, PROJ-01, PROJ-02.

**In scope:**
- `prims/rng.rs` — reproducible seeded RNG-matrix primitive (host SplitMix64
  promoted from k-means++, **no `OsRng`** per ASVS V6): Gaussian + Achlioptas-sparse
  projection matrices and shuffle permutations; PoolStats memory gate. (PRIM-06)
- `prims/incremental_svd.rs` — incremental-SVD merge composed over the v1 Jacobi
  `svd` (mean-correction row, `svd_flip(u_based_decision=False)`, ddof=1),
  validated standalone against a **2+ batch** host reference. (PRIM-07)
- New `PartialFit<F>` trait alongside the existing `Fit`/`Predict`/`Transform`
  surface in `crates/mlrs-algos/src/traits.rs`.
- `EmpiricalCovariance` (ddof=0 MLE), `LedoitWolf` — `covariance_`/`location_`/
  `precision_`/`shrinkage_` matching sklearn ≤ 1e-5. (COV-01, COV-02)
- `IncrementalPCA` via `partial_fit` over batches — `components_`,
  `explained_variance_`, `explained_variance_ratio_`, `singular_values_`,
  `mean_`, `var_`, `transform`/`inverse_transform` ≤ 1e-5 after `svd_flip`
  (V-based) sign alignment. (DECOMP-03)
- `GaussianRandomProjection`, `SparseRandomProjection` (`n_components='auto'`
  via `johnson_lindenstrauss_min_dim`) + `transform` — **property-gated**, not
  1e-5; `johnson_lindenstrauss_min_dim` itself value-matched; sparse input
  densified at the Python ingress. (PROJ-01, PROJ-02)

**Out of scope (deferred / other phases):**
- A dedicated device RNG kernel (host-generate-then-upload is the v1 idiom; the
  `[v2-P1]` spike confirms whether a device kernel is ever needed — default NO).
- A dedicated incremental rank-update SVD kernel unless the `[v2-P1]` spike
  proves full-Jacobi-per-batch is unstable on f32/rocm.
- Kernel/spectral/SGD/Naive-Bayes families (Phases 8–11).

</domain>

<decisions>
## Implementation Decisions

### PartialFit trait & IncrementalPCA streaming contract
- **D-01:** Add a new `PartialFit<F>` trait next to `Fit`/`Predict`/`Transform`
  (same `<F: Float + CubeElement + Pod>` bound, same `pool`/`DeviceArray`/explicit
  `(rows,cols)` shape convention, device-resident state per D-03). It is the new
  cross-cutting PY-06 contract (also reused by Phase 10 MBSGD).
- **D-02:** `fit()` is **sklearn-faithful**: it resets fitted state, then iterates
  `partial_fit` over `gen_batches(n_samples, batch_size)`. This exercises the
  PRIM-07 multi-batch merge *inside* `fit`, not only via explicit user batching.
- **D-03:** `batch_size=None → 5·n_features` (sklearn IncrementalPCA default);
  `n_samples_seen_` accumulated across `partial_fit` calls and exposed.
- **D-04:** The `[v2-P1]` incremental-SVD merge algorithm (full Jacobi re-SVD of
  the stacked `[prev_singular·V; mean-correction; batch]` per batch vs a dedicated
  rank-update kernel) is **handed to the research spike** before planning. No
  strong user preference; default leaning is full-Jacobi-per-batch reusing v1
  `svd` (zero new kernel), to be confirmed for f32-on-rocm stability by the spike.

### precision_ (inverse covariance)
- **D-05:** Compute `precision_` as a **symmetric pseudo-inverse via the v1 `eig`
  prim** (pinvh-equivalent: `V·diag(1/λ_i, with a near-zero floor)·Vᵀ`), matching
  sklearn's `linalg.pinvh`. Singular-safe — handles the EmpiricalCovariance MLE
  rank-deficient case (`n_samples ≤ n_features`) without raising. Do **not** use
  the v1 Cholesky prim for the inverse (SPD-only, fails on singular covariance).
  Reuse the v1 cutoff/near-zero-floor convention (cf. 04-03 σ⁺ RCOND pattern).

### sklearn parameter fidelity (implement all four now)
- **D-06:** `whiten` (IncrementalPCA) — whitened transform output (components
  scaled by `1/sqrt(explained_variance_)`); `inverse_transform` un-whitens.
- **D-07:** `assume_centered` (EmpiricalCovariance + LedoitWolf) — when true, skip
  mean subtraction and set `location_ = 0`; covariance computed about the origin.
- **D-08:** `store_precision` / `precision_` accessor (default `True`) — required
  by COV-01's `precision_` criterion regardless; gates whether D-05 runs.
- **D-09:** `batch_size` (IncrementalPCA) — explicit control for the D-02/D-03
  `partial_fit` batching loop.

### RandomProjection property-gate (PROJ-01/PROJ-02)
- **D-10:** Use **strict (tight) property-gate thresholds** — JL distortion checked
  close to the theoretical bound; matrix-distribution moments held to tight
  tolerances. This is the user's explicit choice over looser flake-resistant bands.
- **D-11:** **Mitigate the flakiness risk of strict bands** with deterministic
  seeding (fixed SplitMix64 seed → identical matrix across runs/backends per
  PRIM-06) and **averaging the distortion/moment statistics over many trials**, so
  tight thresholds stay reproducible across cpu/rocm rather than seed-fragile.
  Researcher/planner pins the exact threshold numbers and trial count.
- **D-12:** RandomProjection correctness gate is the **structural property set**
  (JL distortion bound, matrix-distribution stats, seed-reproducibility,
  `transform == X·componentsᵀ` self-consistency), explicitly **NOT** a 1e-5 value
  oracle (mlrs SplitMix64 RNG ≠ NumPy MT19937). `johnson_lindenstrauss_min_dim`
  *is* value-matched to sklearn. SparseRandomProjection `components_` are stored
  **dense** (no sparse device kernels in v2; acceptable at v2 sizes).

### Claude's Discretion
- Exact f32-on-rocm tolerance bands for LedoitWolf / IncrementalPCA (components
  band + sign; explained_variance band) — follow the v1 per-family documented-band
  precedent (Recurring gates in ROADMAP Phase 7).
- Whether `EmpiricalCovariance`/`LedoitWolf` expose `error_norm`/`mahalanobis`
  helpers — only if cheap and within the COV-01/02 surface; otherwise defer.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` — Phase 7 "Covariance & Projection" success criteria,
  recurring gates, and the `[v2-P1]` research flag.
- `.planning/REQUIREMENTS.md` — PRIM-06, PRIM-07, COV-01, COV-02, DECOMP-03,
  PROJ-01, PROJ-02 (exact wording incl. property-gate vs 1e-5 distinction).
- `.planning/PROJECT.md` — milestone v2.0 goal, constraints, Key Decisions table
  (gate D-07, oracle, primitive-first discipline).
- `.planning/seeds/v2-breadth-roadmap.md` — v2 family/prim mapping.

### Open research (resolve before/at planning)
- `.planning/research/questions.md` §`[v2-P1]` — incremental-SVD merge in CubeCL
  and RNG-matrix generator on device. **Run the research spike before planning**
  (D-04).
- `.planning/research/SUMMARY.md` — v2 project research backing the roadmap.

### Reusable primitive & estimator code (v1, validated)
- `crates/mlrs-algos/src/traits.rs` — `Fit`/`Predict`/`Transform` surface to
  extend with `PartialFit<F>` (D-01); conventions (generic-over-F, device-resident,
  flat row-major + explicit shape).
- `crates/mlrs-backend/src/prims/covariance.rs` — centered-AᵀA/(n-ddof) Gram with
  in-place scale + GEMM-output-buffer reuse (consumed by COV-01/02).
- `crates/mlrs-backend/src/prims/svd.rs` — v1 Jacobi SVD (descending S, thin-U via
  GEMM, `svd_flip` applied estimator-side); base for `incremental_svd.rs` + PCA.
- `crates/mlrs-backend/src/prims/eig.rs` — symmetric eig (descending), base for the
  D-05 pinvh `precision_`.
- `crates/mlrs-backend/src/prims/kmeans.rs` — host SplitMix64 PRNG to **promote**
  into `prims/rng.rs` (PRIM-06).
- `crates/mlrs-algos/src/decomposition/pca.rs` — v1 PCA skeleton (center →
  svd → align_rows(Vᵀ) → truncate; explained_variance_ host pass) that
  IncrementalPCA mirrors.
- `crates/mlrs-algos/src/error.rs` — `AlgoError` (extend with any new Phase-7
  hyperparameter guards in the existing struct-variant style).
- `tests/` + `crates/*/tests/` + `gen_oracle.py` — committed-`.npz` oracle harness
  (regen needs a `/tmp` venv with numpy+scipy+sklearn, PEP 668).

### Kernel / build guidance
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — CubeCL manuals
  (only relevant if the `[v2-P1]` spike forces a new kernel; both new Phase-7
  prims are host-side glue otherwise).
- `AGENTS.md` — tests separated from source; consult CubeCL error guideline on any
  build error; generics-over-float protocol.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **v1 covariance prim** (`prims/covariance.rs`): centered Gram with ddof folded
  into the scale factor — directly serves EmpiricalCovariance (`covariance_`).
- **v1 svd prim** (`prims/svd.rs`): the base PRIM-07 composes over; PCA-style
  `svd_flip` is applied estimator-side (primitive stays raw).
- **v1 eig prim** (`prims/eig.rs`): serves the D-05 pinvh `precision_`.
- **Host SplitMix64** inside `prims/kmeans.rs` (`kmeanspp_sample`): the exact PRNG
  to promote into `prims/rng.rs` — already hand-rolled, no `rand`/`OsRng`.
- **PyO3 `any_estimator!` machinery** (`crates/mlrs-py/src/dispatch.rs`): v2 adds
  zero binding infrastructure — each estimator gets the Unfit/F32/F64 enum + a
  hand-written `#[pymethods]` body with dtype-suffixed accessors, `py.detach` GIL
  release, and `guard_f64()` before the F64 arm.

### Established Patterns
- **GATHER/host-glue discipline:** both new prims are host-side (RNG generate +
  SVD merge orchestration) — avoids the cpu-MLIR SharedMemory/atomics/`F::INFINITY`
  landmines entirely. Keep any new kernel (if the spike forces one) feature-free,
  SharedMemory-free, F/u32-accumulators-only.
- **Build-failing PoolStats memory gate per new prim** (D-10 precedent): one gate
  for `rng.rs`, one for `incremental_svd.rs`.
- **f64 oracle cases gated by `skip_f64_with_log`** (cpu runs f64, rocm skips):
  mirror `gemm_test.rs` for every f64 case.
- **Documented per-family f32 tolerance bands** (D-08 growth point): LedoitWolf /
  IncrementalPCA get f32-on-rocm bands; f64 stays strict 1e-5.

### Integration Points
- `crates/mlrs-algos/src/traits.rs` — add `PartialFit<F>`.
- `crates/mlrs-algos/src/{decomposition,covariance?}/` — new estimator modules
  (decomposition for IncrementalPCA; a new covariance/projection module group —
  file-disjoint, register in `lib.rs` index).
- `crates/mlrs-backend/src/prims/mod.rs` — register `rng` + `incremental_svd`.
- `crates/mlrs-py/src/` — wrap the new estimators incrementally (PY-06 final
  sign-off is formally Phase 11, but each phase wraps its own).

</code_context>

<specifics>
## Specific Ideas

- `fit()` must behave like sklearn IncrementalPCA: loop `partial_fit` over
  `gen_batches`, default `batch_size = 5·n_features` (D-02/D-03).
- `precision_` must match `sklearn.covariance` semantics via `linalg.pinvh`
  (eig-based), not a Cholesky inverse (D-05).
- Strict property-gate bands, made reproducible by deterministic seeding +
  many-trial averaging (D-10/D-11) — user explicitly wants the tight bar.

</specifics>

<deferred>
## Deferred Ideas

- **Device RNG kernel** — only if the `[v2-P1]` spike shows host-generate-then-upload
  is a bottleneck (default: not needed in v2).
- **Dedicated incremental rank-update SVD kernel** — only if full-Jacobi-per-batch
  is unstable on f32/rocm per the spike.
- **Sparse device kernels for SparseRandomProjection** — out of v2 scope; sparse
  input densified at ingress, `components_` stored dense.
- None outside phase scope surfaced during discussion — stayed within Phase 7.

</deferred>

---

*Phase: 7-covariance-projection*
*Context gathered: 2026-06-14*
