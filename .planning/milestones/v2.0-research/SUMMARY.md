# Project Research Summary

**Project:** mlrs — cuML in Rust (v2.0 Breadth Sweep milestone)
**Domain:** scikit-learn-compatible ML estimator library (Rust/CubeCL rewrite of RAPIDS cuML)
**Researched:** 2026-06-14
**Confidence:** HIGH (three scoped MEDIUM unknowns to resolve at plan time)

## Executive Summary

v2.0 is a **breadth sweep**: ~16 sklearn-compatible estimators across five families (Covariance & Projection, Kernel, Spectral, SGD/linear-SVM, Naive Bayes) added on top of v1's shipped, validated five-crate stack. The single most important conclusion from all four research files is that **this milestone requires no new compute dependency**. Every estimator is *assembly* over v1's existing primitives (GEMM, reductions, pairwise distance, covariance/Gram, Jacobi SVD + symmetric eig, Cholesky/triangular-solve, top-k, L-BFGS, coordinate descent) plus exactly five new *feature-free* mlrs-authored CubeCL primitives — one (or zero) per phase: RNG-matrix + incremental-SVD (P7), kernel-matrix (P8), graph-Laplacian (P9), SGD solver (P10), and none for Naive Bayes (P11, reductions only). The workspace `Cargo.toml` is unchanged; pyo3 stays pinned at 0.28 (arrow-59 transitively pins it), `cubek-random` is rejected (no caller seed → breaks ASVS-V6 reproducibility; shared-memory Tausworthe → breaks cpu-MLIR), and sparse input is densified at the Python ingress rather than adding a CSR device path (a v3 line item).

The recommended approach reuses v1's proven discipline verbatim: **primitive-first** (land the prim + its build-failing PoolStats memory gate, then the estimators as thin assembly), **file-disjoint parallelism** (each estimator is a new file + one `pub mod`/`pub use` line; the only shared-edit points are `mlrs-py/src/lib.rs` registration and family `mod.rs` files), and **scikit-learn as the sole oracle** at abs/rel ≤ 1e-5 on f64. The build order from the seed roadmap is sound and dependency-correct: 7→8→9→10→11, with one dependency that must be made explicit — **P9 (Spectral) hard-depends on P8's kernel-matrix prim** because the graph affinity *is* `kernel_matrix(Rbf)`. P7 and P11 are pure assembly (lowest risk, ideal confidence-building bookends); P10 (the one genuinely new solver) is the highest risk and unblocks four estimators.

The risk profile is dominated by two cross-cutting concerns. First, **cpu-MLIR safety**: the new SGD, Naive-Bayes, and Laplacian kernels all have a naive formulation that wants cross-unit atomics or SharedMemory (scatter-add into weight/class/degree bins), which compiled-but-panicked-at-launch in v1. The fix is the v1 GATHER idiom — invert parallelism to one-thread-per-output-cell with F/u32 accumulators, if-guards, no `bool`/`F::INFINITY`/descending-shift loops, no SharedMemory, no atomics. Second, **oracle exceptions**: most estimators value-match sklearn at 1e-5, but RandomProjection is *property-gated* (Johnson–Lindenstrauss distortion + matrix distribution, NOT value-match, because mlrs's seeded PRNG ≠ NumPy MT19937), and SGD/LinearSVM need a *pinned deterministic self-reference oracle* (shuffle off, fixed schedule, fixed `max_iter`, `tol=0`) exactly as v1 did for LogReg's L-BFGS. cuML is NOT the oracle — it diverges from sklearn on the SGD loss set/schedule and on LinearSVC's solver. Three genuine unknowns (incremental-SVD f32 stability, smallest-eigenpair extraction, SGD under cpu-MLIR) must get a research spike before planning P7/P9/P10.

## Key Findings

### Recommended Stack

No new runtime crate is added. All v2 compute composes from v1 prims plus five mlrs-authored, feature-free CubeCL primitives. RNG is host-side SplitMix64 (promoted from `prims/kmeans.rs` into a shared `prims/rng.rs`) generating the projection matrix / epoch-shuffle permutation on host and uploading once — reproducible (ASVS-V6), backend-independent, and cpu-MLIR-safe. Sparse input is densified at the Python wrapper before the existing dense Arrow ingress; a native CSR device path is explicitly deferred to v3. See [STACK.md](STACK.md).

**Core technologies (all UNCHANGED from v1):**
- cubecl 0.10.0 (`default-features=false`): device-kernel layer, generic over float + runtime — entire v2 surface is feature-free kernels over the existing pattern.
- cubek-matmul / cubek-reduce 0.2.0: GEMM and reductions backing KernelRidge, RandomProjection, SGD, the full Naive-Bayes family, and LedoitWolf shrinkage — already wired.
- arrow 59 (`pyarrow`) + pyo3 0.28 (`abi3-py312`): dense Float32/Float64 ingress is sufficient (sparse densified at ingress). **pyo3 stays pinned at 0.28** — arrow-59 transitively pins it; bumping to 0.29 links a second PyInit ABI and crashes the wheel at import (D-09/PY-05).
- **NOT used:** `cubek-random` (no caller seed → ASVS-V6 fail; shared-memory Tausworthe → cpu-MLIR fail), `rand`/`getrandom` (OsRng forbidden by ASVS-V6), `ndarray`/`nalgebra` (math runs on device via existing prims).

**Five new mlrs-authored primitives (NOT crates):** `prims/rng.rs` (host PRNG, no device kernel), `prims/incremental_svd.rs` (glue over v1 `svd`), `prims/kernel_matrix.rs` (one elementwise map over distance/Gram → linear/RBF/poly/sigmoid), `prims/laplacian.rs` (reduce + elementwise over affinity), `prims/sgd.rs` (host epoch loop + GATHER per-minibatch device passes).

### Expected Features

The oracle is **scikit-learn**, not cuML. Where cuML diverges (SGD loss set/schedule, LinearSVC solver), match sklearn. Every estimator exposes sklearn-named params/defaults, the appropriate `fit`/`predict`/`transform`/`score` surface, sklearn fitted attributes (trailing-underscore), `n_features_in_`, and passes the v1 `estimator_checks` harness. See [FEATURES.md](FEATURES.md).

**Must have (the 16 firm estimators = the MVP):**
- Covariance: EmpiricalCovariance, LedoitWolf (ddof=0 MLE covariance; LW shrinkage clipped to [0,1]).
- Projection: IncrementalPCA (incremental-SVD merge, V-based `svd_flip`, ddof=1 explained_variance), Gaussian/SparseRandomProjection (property-gated).
- Kernel: KernelRidge (Cholesky dual solve), KernelDensity (brute-force kernel-sum, log-sum-exp).
- Spectral: SpectralEmbedding, SpectralClustering (full-spectrum eig → take smallest nontrivial → KMeans).
- SGD/linear-SVM: MBSGDClassifier/Regressor (sklearn schedules incl. `optimal` + Bottou t0), LinearSVC/SVR (sklearn liblinear objective: `squared_hinge`/`squared_epsilon_insensitive` defaults, regularized intercept via `intercept_scaling`).
- Naive Bayes: Gaussian/Multinomial/Bernoulli/Complement/CategoricalNB (per-variant smoothing; log-sum-exp predict_proba).

**Should have (differentiators):** f64 device path for all 16 (most GPU libs are f32-only); a single generic kernel-matrix prim reused by KRR/KDE and future kernel-SVM; exact (not approximate) spectral.

**Defer (v3):** `crammer_singer` multiclass, callable/numba kernels, KDE `sample`, spectral `discretize`/`cluster_qr`, tree-based KDE, native sparse interchange, Lanczos/shift-invert smallest-eigenpair solver, Nyström kernel approximation.

### Architecture Approach

The v1 five-crate seam is REUSED unchanged and the dependency graph stays acyclic. Every v2 addition is a new file plus a `pub mod`/`pub use` line; no v2 work edits a v1 estimator file. Two new traits are added (`PartialFit<F>` for IncrementalPCA, `ScoreSamples<F>` for KernelDensity); all other traits (`Fit`, `Predict`, `Transform`, `PredictLabels`, `PredictProba`) are reused, and covariance estimators need no new trait (`Fit` + accessors, no `predict`). See [ARCHITECTURE.md](ARCHITECTURE.md).

**Major components:**
1. `mlrs-kernels` — feature-free `#[cube]` kernels (new: kernel-matrix elementwise map, Laplacian D^{-1/2} scale, SGD per-minibatch grad/update, NB per-class reductions).
2. `mlrs-backend/src/prims/` — the only kernel-launch site (D-13); five new prim files following the `validate_geometry → compose prims → in-place scale on reused out → return exact handle` contract, each with a PoolStats memory gate.
3. `mlrs-algos` — estimator structs that COMPOSE prims (never launch kernels); new modules `covariance/`, `projection/`, `kernel/`, `manifold/`, + spectral in `cluster/`, MBSGD/SVM in `linear/`, `naive_bayes/`.
4. `mlrs-py` — `#[pyclass]` via `any_estimator!` enum (Unfit/F32/F64), dtype-suffixed accessors, `py.detach` + `guard_f64`, pure-Python sklearn shim reusing `MlrsBase`.
5. `scripts/gen_oracle.py` + committed `tests/fixtures/*.npz` — per-estimator fixtures, with the RandomProjection property-gate exception.

### Critical Pitfalls

Top pitfalls, all grounded in v1 idioms + project memory. See [PITFALLS.md](PITFALLS.md).

1. **cpu-MLIR scatter-add panic (SGD/NB/Laplacian)** — naive kernels accumulate into shared weight/class/degree bins (cross-unit atomics or SharedMemory) → compiles under cuda/wgpu but PANICS at launch under `--features cpu`. **Avoid:** GATHER rewrite — one thread per output cell (weight coord / (class,feature) / row), ascending scan, F/u32 accumulators, if-guards, no SharedMemory/atomics.
2. **RandomProjection value-unmatchable (wrong oracle)** — mlrs's seeded PRNG ≠ NumPy MT19937, so a value oracle reports total failure on correct code. **Avoid:** property gate (JL distortion within `eps`, matrix distribution, seed-reproducibility, `transform == X@components_.T` self-consistency); value-match only `johnson_lindenstrauss_min_dim` (pure arithmetic).
3. **SGD oracle nondeterminism (and using cuML)** — shuffle RNG + `optimal` schedule + early-stop make a default-config oracle a moving target; cuML diverges. **Avoid:** pinned self-reference oracle (`shuffle=False`, fixed `eta0`/schedule, fixed `max_iter`, `tol=0`, `n_iter_no_change=max_iter`) referencing sklearn, mirroring v1 LogReg; separate looser convergence property test.
4. **Spectral takes the WRONG eigenvectors** — v1 eig returns full *descending* spectrum; spectral needs *smallest* nontrivial. **Avoid:** sort ascending, drop index 0 (trivial ≈0 vector), take next k; `_deterministic_vector_sign_flip` for embedding output; clustering is sign-immune via `label_perm`; subspace test for degenerate spectra.
5. **f64-on-rocm runs instead of skipping (gate violation)** — every new f64 oracle case must begin `if capability::skip_f64_with_log() { return; }` (cubecl-cpp 0.10 has F64 unregistered for HIP). Hits EVERY new test file in P7–11; make it a per-file checklist line.

Additional recurring traps: IncrementalPCA merge (mean-correction factor, stack order, V-based `svd_flip`, ddof=1) — test with 2+ batches; LedoitWolf/EmpCov ddof=0 normalization + `shrinkage_ ∈ [0,1]`; NB underflow/smoothing/`var_smoothing` from *global* feature variance; large n×n kernel/Gram/Laplacian tiles overflowing gfx1100 LDS (65536 B) → keep big operands in global memory; per-family f32-on-rocm bands (classifiers/clusterers keep exact label/argmax as the hard gate).

## Implications for Roadmap

Build order 7→8→9→10→11, continuing v1's phase numbering. The seed roadmap order is dependency-correct; the one addition is making the P8→P9 dependency explicit.

### Phase 7: Covariance & Projection
**Rationale:** Pure assembly on v1 covariance + SVD — lowest-risk opener, builds confidence and lands the reusable RNG-matrix prim early. First introduces the `PartialFit` trait.
**Delivers:** `prims/rng.rs` (host PRNG), `prims/incremental_svd.rs` (glue over v1 svd), `PartialFit<F>` trait; EmpiricalCovariance, LedoitWolf, IncrementalPCA, GaussianRandomProjection, SparseRandomProjection.
**Addresses (FEATURES):** Covariance + Projection families.
**Avoids (PITFALLS):** RandomProjection wrong-oracle (set property-gate contract up front); IncrementalPCA merge (V-based svd_flip, ddof=1, 2+ batch oracle); LedoitWolf ddof=0 + shrinkage∈[0,1]; f64-on-rocm skip guard.

### Phase 8: Kernel Family
**Rationale:** Lands the keystone kernel-matrix prim (linear/RBF/poly/sigmoid) that P9 and future kernel-SVM reuse. Introduces the `ScoreSamples` trait.
**Delivers:** `prims/kernel_matrix.rs` (one elementwise map over distance/Gram), `ScoreSamples<F>` trait; KernelRidge (reuses `cholesky_solve`), KernelDensity (reuses distance + log-sum-exp).
**Uses (STACK):** existing distance/gemm/cholesky prims + one new elementwise kernel.
**Implements (ARCH):** `prims/kernel_matrix.rs` + `kernel/` estimator module.
**Avoids (PITFALLS):** per-kernel log-norm constants (KDE); singular-K fallback (KRR); LDS budget overflow on dense Gram tiles (keep in global); f64-on-rocm skip guard.

### Phase 9: Spectral Family [HARD DEP on Phase 8]
**Rationale:** Cashes in v1's hardest-won prim (`eig`) for two estimators cheaply; the Laplacian's affinity *is* `kernel_matrix(Rbf)` from P8 — order is mandatory, not just convenient.
**Delivers:** `prims/laplacian.rs` (affinity → degree row-reduction → normalized Laplacian); SpectralEmbedding (Fit+Transform), SpectralClustering (Fit+PredictLabels, reuses v1 KMeans).
**Uses (STACK):** P8 kernel_matrix + v1 eig + v1 kmeans.
**Avoids (PITFALLS):** Laplacian degree scatter + zero-degree (`d_inv_sqrt` guard, NO `F::INFINITY`); wrong/sign eigenvectors (ascending + drop-0 + sign_flip; clustering via label_perm); LDS overflow on dense Laplacian; f64-on-rocm skip guard.

### Phase 10: SGD / Linear-SVM
**Rationale:** The single genuinely new solver; unblocks four estimators at once and carries the highest cpu-MLIR risk. Budget a research spike before planning.
**Delivers:** `prims/sgd.rs` (hinge/log/squared/ε-insensitive losses; l1/l2/elasticnet; LR schedules incl. `optimal` + Bottou t0); MBSGDClassifier/Regressor, LinearSVC, LinearSVR (sklearn liblinear converged-optimum via v1 CD).
**Uses (STACK):** gemm + reductions + host SplitMix64 shuffle.
**Avoids (PITFALLS):** SGD cross-unit atomics (two-pass GATHER kernel); nondeterministic oracle (pinned shuffle=False/fixed-schedule self-reference, sklearn NOT cuML); squared_hinge default + regularized intercept (LinearSVC); f64-on-rocm skip guard; named f32 band with exact predicted labels as hard gate.

### Phase 11: Naive Bayes
**Rationale:** Wide-but-shallow (reductions only) — highest coverage per unit effort, ideal closing bookend. Five mutually-independent, parallel-buildable estimators.
**Delivers:** Gaussian/Multinomial/Bernoulli/Complement/CategoricalNB (all Fit+PredictLabels+PredictProba); no new prim.
**Avoids (PITFALLS):** per-class scatter-add (one-owner-per-(class,feature) GATHER); underflow/smoothing/`var_smoothing` from global variance; log-sum-exp predict_proba (rows sum to 1); per-variant quirks (Bernoulli `(1-x)log(1-p)`, Complement argmin/L1-norm, Categorical ragged list); f64-on-rocm skip guard.

### Phase Ordering Rationale

- **Dependency-driven:** RNG/incremental-SVD (P7) → IncrementalPCA/RandomProjection; kernel-matrix (P8) → spectral affinity (P9) AND future kernel-SVM; Laplacian+eig (P9) reuse v1 eig/KMeans; SGD solver (P10) → all four SGD/SVM estimators; reductions-only (P11) needs nothing new.
- **Risk-tiered:** P7 and P11 are pure assembly (confidence bookends); P8/P9 are MEDIUM (kernel-matrix + smallest-eig); P10 is the HIGH-risk single new solver isolated to one phase.
- **Pitfall-driven:** the P8→P9 hard dependency prevents duplicating the kernel-matrix map in the Laplacian; isolating SGD in P10 contains the cpu-MLIR GATHER risk; bookending with assembly phases keeps momentum either side of the hard solver.

### Research Flags

Phases needing a research spike during planning (the three genuine unknowns from `research/questions.md`):
- **Phase 7:** `[v2-P1]` incremental-SVD merge — settle "full Jacobi per batch vs dedicated rank-update kernel" and the f32-on-rocm stability of the stacked re-SVD.
- **Phase 9:** `[v2-P3]` smallest-eigenpair extraction — confirm full-spectrum-then-slice is acceptable at v2 sizes (vs Lanczos/shift-invert); document the problem-size cap.
- **Phase 10:** `[v2-P4]` SGD under cpu-MLIR — spike the two-pass GATHER kernel and the pinned deterministic oracle before wiring any of the four estimators.

Phases with standard/established patterns (research-phase can be skipped):
- **Phase 8:** kernel-matrix is a known elementwise map over distance/Gram; design is settled in research.
- **Phase 11:** Naive Bayes is reductions-only over the validated v1 reduce prim; per-variant math is fully specified in FEATURES.md.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | crates.io versions verified 2026-06-14; cubek-random/pyo3 decisions grounded in Context7 + transitive-pin analysis + direct source reads. |
| Features | HIGH | sklearn semantics verified live against docs + cuML source; parity-sensitive defaults (SGD `optimal`/t0, LinearSVC `squared_hinge`/`dual='auto'`, NB smoothing) explicitly confirmed. |
| Architecture | HIGH for placement/signatures/traits/dispatch/oracle (mirrors shipped files); MEDIUM for incremental-SVD merge stability, SGD-under-cpu-MLIR, smallest-eigenpair — the three open questions. |
| Pitfalls | HIGH for backend/cpu-MLIR + oracle traps (v1 idioms + memory) and sklearn parity math; MEDIUM for exact f32-on-rocm band magnitudes (must be measured empirically per family, as in v1). |

**Overall confidence:** HIGH

### Gaps to Address

- **Incremental-SVD f32 stability (P7):** the stacked re-SVD merge may compound f32 error on rocm. Handle: research spike at plan time; reuse v1 Jacobi SVD; expect a documented f32 band for components/explained_variance; gate with a 2+ batch oracle.
- **Smallest-eigenpair extraction (P9):** full-spectrum-then-slice is the v2 decision but has an O(n³) size ceiling. Handle: confirm and document the `n_samples` cap during planning; Lanczos/shift-invert deferred to v3.
- **SGD under cpu-MLIR (P10):** the two-pass GATHER kernel is designed but unproven on the MLIR backend. Handle: spike the cpu launch (not just compile) before wiring estimators; verify no Atomic/SharedMemory imports.
- **f32-on-rocm band magnitudes (P7–11):** predicted per-family (LedoitWolf, IncrementalPCA, KernelRidge/Density, Spectral, SGD/SVM, GaussianNB likely need bands) but actual magnitudes must be measured on hardware. Handle: measure during each phase's validation; classifiers/clusterers keep exact label/argmax as the hard gate; never loosen the global tolerance.
- **Sparse densify memory cost:** densifying large term-count matrices can blow the per-phase memory gate. Handle: document in estimator docstrings; let the existing BufferPool gate catch regressions; native CSR is v3.

## Sources

### Primary (HIGH confidence)
- Context7 `/tracel-ai/cubek` + `/tracel-ai/cubecl` — cubek-random API (no seed arg; shared-memory Tausworthe); CubeCL 0.10 multi-runtime.
- crates.io API (2026-06-14) — cubecl 0.10.0, cubek-matmul/reduce 0.2.0, arrow 59.0.0, pyo3 0.28/0.29.
- scikit-learn docs verified live 2026-06-14 — SGDClassifier (optimal/t0/schedules/stopping), LinearSVC/LinearSVR (squared_hinge/dual/liblinear), IncrementalPCA merge, LedoitWolf shrinkage, RandomProjection JL + sparse density, KDE kernels, spectral normalized-Laplacian + deterministic sign flip.
- cuML v26.08 source (read-only reference) — ledoit_wolf, kernel_ridge, sgd.pyx (confirms cuML is a *subset* of sklearn), naive_bayes, manifold, svm, linear_model.
- mlrs shipped code — `traits.rs`, `prims/{covariance,distance,kmeans,lbfgs,cholesky,reduce}.rs`, `dispatch.rs`, `estimators/decomposition.rs`, `python/mlrs/{base,decomposition,linear}.py`, `gen_oracle.py`, `Cargo.toml`.
- Project memory (HIGH) — cubecl-cpu no-SharedMemory/no-atomics; rocm f64-unsupported gate; gfx1100 LDS 65536 B; f32 band policy; oracle /tmp-venv regen; cuML diverges from sklearn.

### Secondary (MEDIUM confidence)
- `research/questions.md` open unknowns `[v2-P1]`/`[v2-P3]`/`[v2-P4]` — incremental-SVD merge stability, smallest-eigenpair approach, SGD-under-cpu-MLIR — to resolve before P7/P9/P10.
- Predicted f32-on-rocm band needs per family — must be measured empirically on rocm hardware during each phase (as in v1).
- RAPIDS issues #2113/#2114 — cuML MBSGD default-schedule divergence from sklearn.

### Tertiary (LOW confidence)
- None — all findings traced to verified sources, shipped code, or scoped open questions.

---
*Research completed: 2026-06-14*
*Ready for roadmap: yes*
