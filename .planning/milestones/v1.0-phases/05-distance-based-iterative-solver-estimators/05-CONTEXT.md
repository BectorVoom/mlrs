# Phase 5: Distance-Based & Iterative-Solver Estimators - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 delivers the remaining v1 estimators in the `mlrs-algos` crate — the
**distance-based** family (`KMeans`, `DBSCAN`, `NearestNeighbors`,
`KNeighborsClassifier`, `KNeighborsRegressor`) and the **iterative-solver**
linear models (`Lasso`, `ElasticNet`, `LogisticRegression`) — each generic over
`<F: Float>` and over the CubeCL runtime, assembled on the validated Phase-2/3
primitives (pairwise distance, reductions, argmin, GEMM, SVD, Cholesky) **plus a
set of NEW Phase-5 primitives** (see D-01). Each estimator `fit`s and exposes the
fitted attributes named in the success criteria, matching scikit-learn **up to
label permutation where applicable** (clustering) and within **1e-5** elsewhere.
This phase completes the v1 algorithm surface in Rust.

Covers requirements **LINEAR-03, LINEAR-04, LINEAR-05, CLUSTER-01, CLUSTER-02,
NEIGH-01, NEIGH-02, NEIGH-03**.

**Scope anchors (carried forward from Phase 4 — NOT re-decided):**
- **Rust-only this phase.** PyO3 `#[pyclass]` bindings, the Python surface,
  `get_params`/`set_params`, Arrow PyCapsule ingest, and per-backend wheels are
  **Phase 6**. Phase 5 ships the Rust estimator layer + Rust oracle tests only.
  The Rust API shape (traits) is chosen now (D-05..D-08) so Phase 6 wraps it
  cleanly, but no Python is written this phase.
- **Gate = cpu(f64) + rocm(f32) (D-07 from Phase 3).** f64 validates on cpu, f32
  on rocm, f64-on-rocm **skips-with-log** (cubecl-cpp 0.10 does not register F64
  for the HIP backend — expected, not a defect). wgpu is opportunistic only.
  ROADMAP/PROJECT still carry old "cpu+wgpu" wording in places — read it as
  "cpu+rocm".
- **Primitive-first discipline.** New compute lands as validated standalone
  primitives in `mlrs-backend` (feature-free `#[cube]` kernels in `mlrs-kernels`)
  with their own oracle + the build-failing PoolStats memory gate, BEFORE the
  estimator consumes them — exactly as Phase-4 treated the new Cholesky primitive.

**The single highest correctness risk in the whole project lives here:**
`LogisticRegression` matching sklearn's `lbfgs` solver within tolerance across
penalty types and the multinomial formulation. The ROADMAP flags it for
`/gsd-plan-phase --research-phase 5`. CD convergence for Lasso/ElasticNet is
medium-risk — validate tolerance during implementation.

</domain>

<decisions>
## Implementation Decisions

### New-primitive boundary (what becomes a gated `mlrs-backend` primitive)
- **D-01: Aggressive promotion — every device-compute kernel is its own
  validated standalone primitive.** Each new piece of device compute lands in
  `mlrs-backend/src/prims/` (feature-free `#[cube]` kernel in `mlrs-kernels`),
  validated standalone f32+f64 cpu+rocm against a numpy/sklearn reference and/or
  an algebraic invariant, with the build-failing PoolStats memory gate, BEFORE
  any estimator consumes it. Candidate primitives (planner to confirm the exact
  set/granularity): **top-k selection** (D-02), **k-means++ D²-weighted sampling**,
  **Lloyd centroid-update / label-assignment**, **eps-region query + core-point
  mask** (DBSCAN), **coordinate-descent step** (Lasso/EN, D-03), **L-BFGS
  direction/update** (LogReg, D-03). Maximum reuse + isolation testability;
  accept the larger number of upfront primitive plans. **Exception:** inherently
  sequential graph traversal (DBSCAN cluster expansion) is NOT a device kernel —
  it runs host-side (D-04).
- **D-02: New top-k selection primitive for KNN.** A partial-select-k kernel over
  the pairwise-distance rows, returning **k indices + k distances** per query row
  with a **lowest-index tie-break** (consistent with the Phase-2 `argmin_rows`
  convention). Shared by `NearestNeighbors`, `KNeighborsClassifier`, and
  `KNeighborsRegressor`. Built on the existing Phase-2 squared-Euclidean distance
  primitive (PRIM-03). NEIGH-01 brute-force only — no spatial index in v1.

### Iterative-solver structure (Lasso / ElasticNet / LogisticRegression)
- **D-03: Two separate solvers — do NOT unify.** A **shared coordinate-descent
  kernel** serves Lasso and ElasticNet (Lasso = `l1_ratio == 1`), per the ROADMAP
  success criterion. An **independent L-BFGS** solver serves LogisticRegression
  (gradient + two-loop recursion). Coupling the highest-risk estimator (LogReg)
  to a generic optimizer frame was rejected — too large a blast radius.
- **D-10: Host-driven iteration loop over device kernels.** The host runs the
  convergence loop; each iteration launches device kernels (gradient /
  coordinate-update) and reads back **exactly one scalar convergence metric**
  (duality gap for CD; gradient-norm / pgtol for L-BFGS). This matches how
  sklearn/scipy structure these solvers and is the easiest way to reproduce their
  exact stopping behavior. In-kernel iteration (the Phase-3 Jacobi precedent) was
  rejected for CD/L-BFGS as far harder and worse at matching sklearn's stopping
  rule.
  - **⚠ MEMORY-GATE RECONCILIATION (planner MUST encode this):** the per-iteration
    scalar convergence readback is an **explicit, documented exception** to the
    no-mid-pipeline-readback rule of the Phase-2/3/4 memory gate. For iterative
    solvers the gate asserts **bounded allocation** instead — solver buffers
    (residuals, gradients, L-BFGS history `(s, y)` pairs, coordinate state) are
    pool-managed and **reused across iterations** (allocation count flat after
    warmup), and there is no per-iteration *array* readback beyond the single
    scalar. State this exception in the gate test so it does not silently regress.
- **D-11: Match scikit-learn's exact convergence criteria.** Reproduce sklearn's
  per-solver stopping rules — Lasso/EN **duality-gap < tol** (defaults `tol=1e-4`,
  `max_iter=1000`); LogReg L-BFGS **gradient-norm / pgtol** (default
  `max_iter=100`). Exact constants pinned by research. A simple
  max-coefficient-change tol was rejected — risks missing sklearn's exact iterate
  count / final `coef_` and the 1e-5 gate.
- **D-12: LogisticRegression = multinomial softmax.** Numerically-stable softmax +
  cross-entropy, matching sklearn's `lbfgs` solver default (lbfgs → multinomial
  for multiclass). **Binary is the 2-class case of the same code path** — one
  formulation, not a separate binary branch. One-vs-rest was rejected (doesn't
  match the LINEAR-05 `lbfgs` contract).
- **D-13: Match scikit-learn's exact objectives / penalty scaling.**
  - Lasso/ElasticNet: `(1/2n)·‖y − Xw‖² + α·penalty` (data term divided by
    `n_samples`), `l1_ratio` mixing the L1/L2 penalty, **intercept via centering**
    (unpenalized), reusing the Phase-2 column-mean reduction + the Phase-4 D-05
    center-then-solve pattern.
  - LogisticRegression: **L2 penalty default**, strength via **`C`** (inverse
    regularization), **intercept unpenalized**.
  - Exact scalings pinned by research — prerequisites for 1e-5 agreement.

### Estimator output API surface (extends the Phase-4 D-04 trait surface)
- **D-05: New label-returning trait(s) for clustering/classification.** Integer
  labels (`KMeans.labels_`, `DBSCAN.labels_`, `KNeighborsClassifier.predict`)
  cannot use the F-typed `Predict<F>`. Add a clustering/classify trait returning
  an **integer `DeviceArray`**; keep `Predict<F>` for the regressors
  (`KNeighborsRegressor`, the linear models). Clean type separation mirroring
  sklearn's Classifier/Cluster vs Regressor mixin split; Phase-6 wraps each trait
  family generically. (Exact trait names/method signatures = Claude's discretion.)
- **D-06: Integer labels and neighbor indices are `i32` everywhere.** DBSCAN noise
  = `-1` forces signed; use `i32` uniformly for all cluster ids, class
  predictions, and neighbor indices even though KMeans/KNN ids are non-negative —
  one label type across the surface, matching sklearn's int labels. Phase-6 maps
  to numpy `int32`. (`DeviceArray<ActiveRuntime, i32>` — confirm the pool/bridge
  support the integer element type during planning.)
- **D-07: Formalize `KNeighbors` and `PredictProba` traits.**
  `NearestNeighbors.kneighbors` returns BOTH distances and indices, and
  `KNeighborsClassifier` needs `predict_proba`. Formalize a **`KNeighbors` trait**
  (returns `(DeviceArray<F>` distances, `DeviceArray<i32>` indices`)`) and a
  **`PredictProba` trait** so Phase-6 wraps the KNN family uniformly.
- **D-08: Match scikit-learn's API shape per estimator family.** Clustering
  estimators implement `Fit` (storing `labels_`/`inertia_`/`cluster_centers_`/
  `core_sample_indices_` **device-resident**, D-03 carried forward) + a
  `fit_predict` convenience. **`KMeans` also implements `Predict`** (assign new
  points to fitted centers via the distance + argmin primitives). **`DBSCAN` does
  NOT implement a standalone `predict`** (no transductive predict — exactly like
  sklearn). Fitted attributes are device-resident accessors materialized lazily
  (D-03).

### KMeans stochastic-oracle strategy (CLUSTER-01)
- **D-09: Inject fixed init centers for a deterministic 1e-5 oracle.** k-means++
  uses an RNG stream that cannot be reproduced bit-for-bit from sklearn, so the
  oracle **supplies the initial centers**; both mlrs and sklearn run Lloyd from
  identical init, then compare `cluster_centers_`/`labels_`/`inertia_` **up to
  label permutation** within 1e-5 (reuse the Phase-1 `label_perm` helper). This
  tests the Lloyd iteration math deterministically, decoupled from RNG
  reproduction. A quality-bound (`inertia ≤ sklearn·(1+ε)`) comparison was
  rejected as too loose.
  - **D-09a: Still implement k-means++ (CLUSTER-01 names it as the sklearn
    default).** Build the D²-weighted sampling primitive (per D-01) so the
    estimator's real default init is k-means++; drive the *deterministic oracle*
    from injected init, and separately sanity-check that k-means++ produces valid,
    seed-reproducible centers.
  - **D-09b: `n_init = 1`** (sklearn's current `'auto'` default for k-means++).
    `n_init=10` (legacy) deferred.
  - **D-09c: Host-side seeded RNG** draws the next center from the
    device-computed D² weights (read back per center — **init only, not the hot
    Lloyd loop**). Backend-independent reproducibility, consistent with the
    Phase-1 seeded-RNG fixture philosophy. Device-side RNG rejected
    (backend-divergent streams, not seed-reproducible).

### DBSCAN execution split (CLUSTER-02)
- **D-04: Device computes, host expands.** The device computes the pairwise
  distance matrix + eps-threshold + **core-point mask**; the **host runs the
  BFS/union-find** cluster expansion (inherently sequential pointer-chasing — a
  poor GPU fit, and the easiest way to reproduce sklearn's exact labels up to
  permutation). Labels follow sklearn: noise = `-1`, plus `core_sample_indices_`.
  - **Memory note for the gate:** the **n² distance matrix** is the dominant
    allocation; the gate should bound it (and confirm buffer reuse) rather than
    expect a device-resident no-readback pipeline — DBSCAN deliberately reads the
    mask/distances back to host for the graph walk.

### Carried forward from Phases 1–4 (reaffirmed, not re-decided)
- Estimators generic over `<F: Float + CubeElement + Pod>`; flat row-major
  `DeviceArray` with explicit `(rows, cols)` per call (P2 D-04); device-resident
  in/out + optional caller-out + pooled scratch (P2 D-05/D-11); `fit` returns
  `&mut self` (P4 D-04); device-resident fitted state, lazy host-materialize
  (P4 D-03). Feature-free `#[cube]` kernels in `mlrs-kernels`; launch wrappers +
  host orchestration in `mlrs-backend`; estimators in `mlrs-algos`. sklearn oracle
  fixtures via `scripts/gen_oracle.py` (regen needs a /tmp venv with
  numpy+scikit-learn per PEP 668 — committed blobs, not test-time); `assert_close`
  1e-5 abs+rel with near-zero floor + the per-family looser bound as the escape
  hatch (P3 D-10); `label_perm` helper for clustering comparisons (P1);
  `skip_f64_with_log` for f64-on-rocm (P3 D-07); `thiserror` in libs / `anyhow` at
  boundaries; deps track latest; source/test separation per AGENTS.md (tests in
  `crates/*/tests/`).

### Claude's Discretion
- Exact set/granularity of the new primitives under D-01 (e.g. is "Lloyd update"
  one primitive or assignment + centroid-recompute split; is the CD step one
  primitive per the shared kernel) — subject to the memory gate, tolerance policy,
  and no-hardcoded-plane-width rule. Researcher-flagged.
- Module/file layout within `mlrs-algos` (e.g. `cluster/`, `neighbors/`,
  `linear/` modules) and exact trait names/method signatures for D-05/D-07.
- The L-BFGS history size `m` and line-search details (D-03/D-11) that reproduce
  scipy's L-BFGS-B behavior sklearn relies on — pick what holds tolerance.
- Exact random shapes/seeds for the oracle sweeps and which cases get committed
  sklearn fixtures vs algebraic-invariant-only checks.
- Naming of new estimator/primitive error variants (extend the `thiserror` enums).
- Distance-metric scope for KNN/DBSCAN (v1 leans Euclidean/squared via PRIM-03);
  `weights='uniform'` vs `'distance'` for KNN — default `'uniform'` unless a
  requirement forces otherwise; confirm during planning.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
  (NB: documents the cpu+wgpu gate — D-07 supersedes it with cpu+rocm)
- `.planning/REQUIREMENTS.md` — LINEAR-03, LINEAR-04, LINEAR-05, CLUSTER-01,
  CLUSTER-02, NEIGH-01, NEIGH-02, NEIGH-03 requirement text + traceability table
- `.planning/ROADMAP.md` §"Phase 5: Distance-Based & Iterative-Solver Estimators"
  — goal + 4 success criteria (the gate for this phase) + the LogReg research flag
- `.planning/phases/04-closed-form-estimators/04-CONTEXT.md` — the trait surface
  (D-04 Fit/Predict/Transform), device-resident state (D-03), center-then-solve
  intercept (D-05), the new-primitive precedent (Cholesky), sklearn-oracle policy
  (D-07), the cpu+rocm gate, the build-failing memory gate this phase extends
- `.planning/phases/03-svd-eigendecomposition-primitive-hard-gate/03-CONTEXT.md`
  — the cpu+rocm gate D-07, in-kernel iteration precedent (Jacobi), tolerance D-10,
  memory gate D-11
- `.planning/phases/02-core-compute-primitives/02-CONTEXT.md` — distance (PRIM-03),
  reductions/argmin/argmin_rows, GEMM transpose flags, device-resident in/out,
  optional-out + pooled scratch, the D-10 memory gate

### Build / kernel protocol (MANDATORY before writing any CubeCL code)
- `AGENTS.md` — source/test separation; CubeCL generics-over-float requirement;
  build-error protocol
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL
  manual index; read before writing the new top-k / k-means++ / CD / L-BFGS kernels
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — generics, plane/
  subgroup, shared-memory, matmul/gemm manuals
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  — MANDATORY troubleshooting reference on ANY CubeCL build/compile/feature/
  toolchain error (new primitives must run on the rocm/HIP gate)

### Memory-efficiency guidance (informs the device-resident state + memory gate)
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL,
  buffer-reuse patterns (device-resident fitted state, bounded scratch, the
  iterative-solver buffer-reuse the D-10 gate exception relies on)

### Reference implementation (read-only — behavior/convention reference)
- `cuml-main/` — RAPIDS cuML v26.08.00; KMeans/DBSCAN/KNN/Lasso/ElasticNet/
  LogisticRegression solver behavior reference (NOT code to port verbatim;
  numerical agreement is with **scikit-learn**, not cuML)
- scikit-learn source/docs — the oracle contract: `KMeans` (k-means++,
  `n_init='auto'`), `DBSCAN` (`eps`/`min_samples`, brute algorithm),
  `NearestNeighbors`/`KNeighborsClassifier`/`KNeighborsRegressor` (brute-force,
  `weights='uniform'`), `Lasso`/`ElasticNet` (cyclic coordinate descent, duality-gap
  stopping, `(1/2n)` data-term scaling), `LogisticRegression` (`solver='lbfgs'`,
  multinomial, `C`)
- `.planning/codebase/*.md` — codebase maps (ARCHITECTURE, CONVENTIONS, STACK,
  TESTING, STRUCTURE, INTEGRATIONS, CONCERNS)

### Existing source this phase consumes / extends
- `crates/mlrs-algos/src/lib.rs`, `src/traits.rs` (Fit/Predict/Transform — extend
  with D-05/D-07 label/kneighbors/proba traits), `src/error.rs` (extend AlgoError)
- `crates/mlrs-backend/src/prims/distance.rs` — pairwise squared-Euclidean
  (KMeans assignment, DBSCAN eps-query, KNN top-k input)
- `crates/mlrs-backend/src/prims/reduce.rs` — `argmin_rows`/`argmax_rows`
  (KMeans label assignment, KNN voting), `column_reduce`/`mean` (centering,
  centroid recompute, inertia)
- `crates/mlrs-backend/src/prims/{gemm.rs, covariance.rs, svd.rs, cholesky.rs}` —
  available primitives (CD/L-BFGS may reuse GEMM for `Xw`/gradients)
- `crates/mlrs-core/src/{oracle.rs, compare.rs, tolerance.rs, label_perm.rs}` —
  oracle harness, `assert_close`, 1e-5 policy, label-permutation comparison (D-09)
- `crates/mlrs-backend/src/{device_array.rs, pool.rs}` — DeviceArray + BufferPool +
  PoolStats (D-10 gate; confirm i32 element-type support for D-06)
- `crates/mlrs-backend/src/{runtime.rs, capability.rs}` — `ActiveRuntime`,
  `skip_f64_with_log`
- `crates/mlrs-backend/tests/memory_gate_test.rs` — the hard PoolStats gate to
  extend (with the D-10 iterative-solver exception + the DBSCAN n² note)
- `crates/mlrs-kernels/src/{reduce.rs, ...}` — feature-free kernel home for the new
  top-k / k-means++ / CD / L-BFGS / region-query kernels
- `scripts/gen_oracle.py` — oracle fixture generation (extend with sklearn KMeans/
  DBSCAN/KNN/Lasso/ElasticNet/LogReg fixtures; KMeans fixtures carry injected init)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Pairwise squared-Euclidean distance (PRIM-03, Phase 2):** the shared input to
  KMeans assignment, DBSCAN eps-query, and KNN top-k. One primitive, three families.
- **`argmin_rows` / `argmax_rows` (Phase 2):** KMeans per-sample label assignment
  and KNN majority voting reuse the per-row index reductions (lowest-index
  tie-break already pinned — the same convention D-02 adopts for top-k).
- **`column_reduce` / `mean` (Phase 2):** KMeans centroid recompute + inertia,
  Lasso/EN/LogReg centering for the intercept (D-13, reusing Phase-4 D-05).
- **GEMM (Phase 2):** `Xw` residuals and gradients for CD/L-BFGS.
- **`label_perm` + oracle harness + `assert_close` (Phase 1):** clustering
  comparison up to permutation (D-09); the per-family looser bound is the escape
  hatch for ill-conditioned iterative cases.
- **`Fit`/`Predict`/`Transform` traits + device-resident state (Phase 4 D-03/D-04):**
  the surface Phase 5 extends with integer-label, kneighbors, and proba traits.
- **DeviceArray + BufferPool + PoolStats + the build-failing memory gate:** the
  verification surface; Phase 5 extends it with the D-10 iterative-solver exception
  and the DBSCAN n²-matrix bound.

### Established Patterns
- Feature-free kernels in `mlrs-kernels`; runtime-bound launch wrappers in
  `mlrs-backend/prims/`; estimators in `mlrs-algos` consume the backend primitive
  API — the new Phase-5 primitives (D-01) honor the same split.
- scikit-learn/numpy/LAPACK conventions are the contract (duality-gap stopping,
  multinomial softmax, `(1/2n)` scaling, noise=-1, label permutation), NOT cuML's.
- Primitive-first: a new primitive passes its own standalone validation (f32+f64,
  cpu+rocm) + memory gate BEFORE the estimator consumes it (Phase-4 Cholesky
  precedent). Sequence primitives first within each estimator's plan set.

### Integration Points
- **`mlrs-algos` already has the Phase-4 estimators + trait surface** — Phase 5
  adds modules (cluster/neighbors/linear) and extends `traits.rs`/`error.rs`.
- **The new primitives (D-01) are the critical path** — top-k, k-means++ sampling,
  CD step, and especially L-BFGS must each pass standalone validation before the
  estimator built on them. L-BFGS + LogReg is the project's highest-risk pairing
  (ROADMAP research flag).
- **Phase 6 (PyO3) consumes the extended trait surface (D-05/D-07) + i32 labels
  (D-06)** — chosen now so the zero-copy Arrow + GIL-release wrapping stays generic.

</code_context>

<specifics>
## Specific Ideas

- **KMeans oracle is deliberately init-injected (D-09), not RNG-matched** — the
  Lloyd math is tested deterministically; k-means++ is implemented separately to
  satisfy CLUSTER-01 and sanity-checked for validity/seed-reproducibility. This is
  intentional, not a shortcut.
- **The two iterative-solver families use deliberately different solvers (D-03):**
  shared coordinate-descent for Lasso/ElasticNet (Lasso = `l1_ratio==1`),
  independent L-BFGS for LogisticRegression. Do not unify them.
- **LogisticRegression is multinomial-by-default (D-12)**, with binary as the
  2-class special case of the same softmax path — matching sklearn `lbfgs`, not OvR.
- **Host-driven iteration with a single scalar readback per iteration (D-10)** is
  the visible departure from the strict device-resident pipeline — it is a
  documented, gate-encoded exception, not a memory-gate regression.
- **DBSCAN is a device-compute + host-graph-walk hybrid (D-04)** — the n² distance
  matrix is the accepted memory cost for brute-force v1.

</specifics>

<deferred>
## Deferred Ideas

- **KMeans `n_init=10` / multi-restart-keep-best** — v1 ships `n_init=1` ('auto',
  D-09b). Add if a case needs robustness to bad init.
- **KNN `weights='distance'`, non-Euclidean metrics, spatial indices (kd-tree/
  ball-tree)** — v1 is brute-force Euclidean, `weights='uniform'` (NEIGH-01).
  Spatial acceleration and metric/weight knobs are later work.
- **DBSCAN device-side label propagation** — rejected for v1 (D-04 host expansion);
  revisit if the n² host round-trip becomes a bottleneck on large n.
- **Additional sklearn constructor knobs** — `algorithm`/`leaf_size`/`p` (neighbors),
  `selection='random'`/`positive`/`warm_start` (CD), `penalty='l1'/'elasticnet'`/
  `solver` choices / `class_weight` / `multi_class='ovr'` (LogReg), `metric`
  variants (DBSCAN). Out of v1 scope; revisit per Phase-6 estimator-checks needs.
- **Reusing the new optimizer primitives elsewhere** — top-k could serve future
  ranking/ANN paths; L-BFGS could serve other GLMs; CD could serve other sparse
  models. Built for their v1 consumers; no obligation to generalize now.

### Reviewed Todos (not folded)
None — no pending todos matched this phase.

## Open Questions for Research (run `/gsd-plan-phase --research-phase 5`)
- **L-BFGS + LogisticRegression matching sklearn `lbfgs` within tolerance (D-03/
  D-11/D-12/D-13)** — THE highest project risk. L-BFGS history size, line search,
  multinomial cross-entropy gradient/Hessian-free direction, stable softmax,
  penalty `C` scaling, intercept handling, exact pgtol/max_iter stopping. Validate
  the L-BFGS primitive standalone (on a convex test objective with a known
  minimizer) BEFORE LogReg consumes it.
- **Coordinate-descent convergence for Lasso/ElasticNet (D-03/D-11/D-13)** —
  cyclic CD, soft-thresholding, duality-gap stopping with `tol=1e-4`/`max_iter=1000`,
  the `(1/2n)` data-term scaling and `l1_ratio` mixing, intercept via centering.
  Confirm `coef_` (including exact zeros / sparsity pattern) matches sklearn's CD.
- **Top-k selection determinism (D-02)** — confirm the k indices + distances match
  sklearn's `kneighbors` with the lowest-index tie-break, including exact-tie cases.
- **KMeans Lloyd-from-injected-init agreement (D-09)** — confirm centers/labels/
  inertia hold 1e-5 up to label permutation; confirm k-means++ init is valid and
  seed-reproducible.
- **DBSCAN label agreement (D-04)** — confirm host BFS/union-find reproduces
  sklearn's `labels_` (noise=-1) and `core_sample_indices_` up to permutation.
- **i32 DeviceArray support (D-06)** — confirm the pool/bridge/kernels handle the
  integer element type for labels and neighbor indices.

</deferred>

---

*Phase: 5-Distance-Based & Iterative-Solver Estimators*
*Context gathered: 2026-06-12*
