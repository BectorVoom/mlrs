# Retrospective — mlrs

Living retrospective across milestones.

## Milestone: v1.0 — Core ML Library

**Shipped:** 2026-06-14
**Phases:** 6 | **Plans:** 38 | **Commits:** 233 | **Timeline:** 3 days (2026-06-11 → 2026-06-14)

### What Was Built

A ground-up Rust rewrite of cuML's scikit-learn-core surface: 12 sklearn-compatible estimators on a single CubeCL codebase generic over `f32`/`f64` and runtime, validated against scikit-learn within 1e-5 on the cpu(f64) + rocm(f32) gate, shipped as four per-backend abi3 wheels with zero-copy Arrow ingest.

### What Worked

- **Primitive-first horizontal build order.** Validating GEMM/reduce/distance/SVD/eig/Cholesky standalone before any estimator meant estimators were "mostly assembly." The SVD/eig hard gate (Phase 3) unblocked 4 estimators; distance unblocked 3.
- **Build-failing memory gates per phase.** Encoding PoolStats assertions (bounded reuse, GEMM-output reuse, read-back counts) as compile/test gates kept memory efficiency from being deferred — it was provable at every phase.
- **scikit-learn as the oracle.** A CPU reference that runs without a GPU made the 1e-5 contract checkable in CI and decoupled correctness from hardware availability.
- **Committed .npz fixtures.** Oracle fixtures generated once (via a /tmp venv) and committed as blobs kept tests hermetic and fast.

### What Was Inefficient

- **cpu-MLIR kernel constraints discovered reactively.** Several kernels compiled but panicked at launch on the cpu backend (SharedMemory + mutable bool / F::INFINITY / descending-shift loops; cross-unit atomics). Each cost a rewrite to the GATHER/no-SharedMemory idiom. Knowing the constraint up front would have saved iterations.
- **Fortran-vs-C contiguity bug latency.** sklearn PCA `components_` is Fortran-order; the scaffold only asserted fixture *length*, so a transposed-read bug stayed latent until Phase 4.
- **f64-on-rocm reality surfaced mid-project.** The gate moved from wgpu+cpu to cpu+rocm at Phase 3 once it was confirmed cubecl-cpp 0.10 doesn't register F64 for HIP.

### Patterns Established

- GATHER kernels (single-owner outputs, F/u32 accumulators, if-guards, no SharedMemory/atomics) as the cpu-MLIR-safe default.
- Validate-geometry-before-unsafe-launch returning typed `PrimError`/`AlgoError` (ASVS V5).
- Estimator-side `svd_flip`/`align_rows` keeping primitives raw.
- f64 oracle cases capability-gated via `skip_f64_with_log` (skip, never fail).
- Two-threshold Jacobi convergence (rotation-skip vs noise-floor break with √pairs scaling).

### Key Lessons

1. Encode environment constraints (cpu-MLIR, f64-on-rocm) as explicit gates/notes early — they reshape kernel design more than algorithm choice does.
2. Scaffold tests that assert *shape and contiguity*, not just length — length-only checks hide layout bugs.
3. For gauge-redundant objectives (symmetric softmax LogReg), pick the correctness witness deliberately (gauge-fixed predict_proba), and document the divergence from sklearn's native parameterization.

### Cost Observations

- Sessions: multi-session over 3 days; model mix not instrumented.
- Notable: primitive-first ordering front-loaded the hard work (Phase 3 SVD/eig) so later estimator phases were fast and low-risk.

## Milestone: v2.0 — Breadth Sweep

**Shipped:** 2026-06-22
**Phases:** 5 (7–11) | **Plans:** 27 | **Tasks:** 52 | **Commits:** ~192 (45 feat) | **Timeline:** built 2026-06-20 → 2026-06-22 (3 days; planned 2026-06-14)

### What Was Built

18 new sklearn-compatible estimators across five families — covariance/projection (EmpiricalCovariance, LedoitWolf, IncrementalPCA, Gaussian/SparseRandomProjection), kernel (KernelRidge, KernelDensity), spectral (SpectralEmbedding, SpectralClustering), SGD/linear-SVM (MBSGDClassifier/Regressor, LinearSVC/SVR), and Naive Bayes (Gaussian/Multinomial/Bernoulli/Complement/Categorical) — built as assembly on v1's primitive base plus five new feature-free CubeCL prims (RNG-matrix, incremental-SVD, kernel-matrix, graph-Laplacian, two-pass SGD solver). Zero new compute dependency; pyo3 stayed 0.28. Total estimator surface now 30.

### What Worked

- **Wave-0 shared-seam scaffold per phase.** Front-loading every shared-file edit (traits, error variants, module index, oracle generators, #[ignore] test scaffolds) into one wave made Waves 1/2/3 strictly file-disjoint and parallel-buildable — the pattern repeated cleanly across all five phases.
- **Primitive-first discipline carried from v1.** Each phase landed + standalone-validated its one new prim (with a PoolStats memory gate) before any estimator consumed it; the highest-risk prim (the SGD device solver) launched on cpu-MLIR first try because the GATHER idiom was applied by construction, not discovered reactively.
- **Reusing the shipped PyO3 layer.** v1's `any_estimator!` machinery generalized to all 18 v2 estimators with zero new binding infrastructure — each phase wrapped its own estimators incrementally.
- **Exact-label hard gate + documented coef bands.** For iterative/host-order solvers (SGD, SVM, NB) the integer-exact label decision was the strict correctness witness while coefficients agreed to a measured band — a clean, honest contract.

### What Was Inefficient

- **Oracle-fixture re-generation mid-phase.** The Phase-9 SpectralEmbedding default-constructor fixture pinned a disconnected kNN graph whose degenerate zero-eigenspace a dense Jacobi eig can't reproduce; the oracle had to be regenerated with an explicit connected parameterization + a subspace gate. Lesson: an oracle pinning an estimator's *default* params must verify the dense-eig path actually reproduces that parameterization.
- **Literal-grep acceptance criteria vs intent.** Several plans had `grep == 0` gates (e.g. `to_host`, `cholesky`, `SharedMemory`) that the correct implementation legitimately tripped (host-side sorts, doc-comments); reconciling literal-grep-vs-intent recurred and cost cycles.
- **Live Python FFI remained environment-gated.** The maturin+pyarrow path can't run here, so PY-06's live `estimator_checks` re-triage + FFI smoke are carried forward as deferred (Rust-side pyclass smoke compensates) — same shape as v1's deferred CUDA-host checks.

### Patterns Established

- Wave-0 shared-seam scaffold (all cross-cutting edits + test scaffolds + committed fixtures in one file-disjoint wave).
- Builder-fronted estimators with sklearn-default field initializers + a split validation contract (data-independent at `build()` → BuildError; data-dependent at `fit()` → AlgoError).
- Property-gate exception for RNG-dependent estimators (RandomProjection: JL distortion + distribution + seed-reproducibility, not 1e-5 value match), with trial-count averaging for reproducible bands.
- Ragged per-feature fitted tables (`Vec<Vec<f64>>`) + per-feature host lookup-and-sum for structurally-distinct estimators (CategoricalNB) where the GEMM joint-LL shape doesn't fit.

### Key Lessons

1. A default-constructor oracle is only meaningful if the implementation's solver path can actually reproduce that default — verify, don't assume (SpectralEmbedding).
2. Prefer behavioral acceptance criteria over literal source greps; when a grep gate is unavoidable, scope it to non-comment lines and document the legitimate exceptions up front.
3. The cpu-MLIR GATHER idiom, applied proactively, eliminated the reactive kernel-rewrite cost that dogged v1 — environment constraints belong in the design, not the debugging.

### Cost Observations

- Model mix: planner + executor both opus (model_profile=quality); mix not finely instrumented.
- Sessions: multi-session over ~3 active build days.
- Notable: the five-prim primitive-first structure again front-loaded risk (SGD solver in Phase 10) so the closing Naive-Bayes phase was wide-but-shallow and fast.

## Milestone: v3.0 — Manifold Algorithms & Rust-Native API

**Shipped:** 2026-06-26
**Phases:** 5 (12–16) | **Plans:** 34 | **Tasks:** 63 | **Commits:** 248 | **Timeline:** built 2026-06-23 → 2026-06-26 (~4 days)

### What Was Built

The UMAP + HDBSCAN manifold/clustering pair on a single shared, multi-metric KNN-graph primitive (euclidean/manhattan/cosine/chebyshev/minkowski-p, cpu-MLIR-safe), plus a Rust-native builder + compile-time fit/unfit typestate API additively retrofitted across all 32 estimators, a pure-Python sklearn shim (verbatim `__init__` + get/set_params/clone, AST-purity gated), and PyO3-wrapped UMAP/HDBSCAN. Oracle broadened to umap-learn 0.5.12 (property gate for the stochastic SGD layout, ≤1e-5 for deterministic stages); HDBSCAN keeps an exact-label hard gate. Zero new compute dependencies.

### What Worked

- **Primitive-first keystone, spike-validated first.** The KNN-graph prim (the milestone's feasibility risk) was spiked (the new Manhattan/Chebyshev/Minkowski-p direct kernels + the in-kernel `F::powf` cpu-MLIR unknown) BEFORE planning, then landed + standalone-gated per metric before UMAP/HDBSCAN touched it. Both consumers were then "mostly assembly" on a proven substrate.
- **Convention-before-retrofit sequencing.** Establishing the builder/typestate *convention* in Phase 12 (so the new estimators were born idiomatic) and isolating the broad, parallel-unsafe 30-estimator *retrofit sweep* to the last phase (16) protected file-disjoint discipline and every shipped 1e-5/exact gate — the additive "builder fronts the existing config, fit body byte-identical" rule meant zero numeric regressions.
- **Gate-type honesty for stochastic output.** UMAP's SplitMix64 ≠ NumPy MT means coordinates can't match; the property gate (trustworthiness/kNN-overlap within margin of umap-learn, byte-identical per seed) was the right contract, reusing the v2 RandomProjection D-12 precedent rather than forcing a doomed value match.
- **File-disjoint parallel estimator phases.** UMAP (14) and HDBSCAN (15) were feature-disjoint and built in parallel after the shared prim landed.

### What Was Inefficient

- **Auto-extracted milestone accomplishments were noisy.** `milestone.complete` scraped stray SUMMARY lines (`[Rule 1 - Bug]`, `Verified GREEN`, `Task 1 — Guard`) into MILESTONES.md; the entry had to be hand-curated. SUMMARY one-liners aren't reliably machine-delimited.
- **A pre-close gate was deferred on a stale environment assumption.** Phase 12's live-PyO3-FFI UAT/verification item sat `human_needed` on the "no maturin/pyarrow host" assumption — but PyPI was reachable, so it was genuinely resolved at close (venv + `maturin develop` + a live UMAP/HDBSCAN script, 22/22 f32+f64). The assumption should have been re-checked at phase time, not carried to milestone close.
- **`min_dist > spread` and `n_components < n` guards surfaced as verification gaps.** UMAP needed CR-01/CR-02/CR-03 gap-closure plans (cross-cube write race, force double-count, n_components guard) after the first verification — the owner-only `move_other=0` fix landed in 14-07.

### Patterns Established

- Spike-before-planning for a feasibility-keystone primitive, then primitive-first land + per-metric standalone gate before any consumer.
- Convention-first / retrofit-last for cross-cutting API changes (born-idiomatic new code; additive, gate-preserving sweep isolated to one parallel-unsafe phase).
- Compile-time typestate (`T<Unfit>`→`T<Fitted>`) with a trybuild compile-fail gate as the predict-before-fit guard, collapsing behind the unchanged PyO3 `any_estimator!` enum.
- Live FFI is runnable here when PyPI is reachable (venv + maturin + pyarrow) — verify, don't auto-defer SHIM-03-class gates.

### Key Lessons

1. Re-check environment-limitation assumptions at the moment a gate is written — "untestable here" can silently become false (a reachable PyPI made the whole live-FFI path runnable in ~15 min).
2. For a feasibility-critical shared primitive, the spike + per-metric standalone gate BEFORE consumers is what makes the downstream estimators low-risk assembly — the v1/v2 primitive-first lesson held a third time.
3. Curate milestone accomplishments by hand — auto-extraction from SUMMARY files captures intra-document scaffolding noise, not the headline deliverable.

### Cost Observations

- Model mix: planner + executor both opus (model_profile=quality); mix not finely instrumented.
- Sessions: multi-session over ~4 active build days; the close itself resolved a carried gate inline rather than deferring it.
- Notable: the largest milestone by diff (426 files, +45k) yet zero numeric regressions, because the broad edit (builder retrofit) was constrained to be additive and gate-preserving.

## Cross-Milestone Trends

| Milestone | Phases | Plans | Days | Estimators |
|-----------|--------|-------|------|------------|
| v1.0 | 6 | 38 | 3 | 12 |
| v2.0 | 5 | 27 | 3 | 18 (30 total) |
| v3.0 | 5 | 34 | 4 | 2 + KNN-graph prim (32 total) |
