# Phase 4: Closed-Form Estimators - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4 delivers four sklearn-compatible **Rust** estimators in the `mlrs-algos`
crate — **`LinearRegression`**, **`Ridge`**, **`PCA`**, **`TruncatedSVD`** —
each generic over `<F: Float>` and over the CubeCL runtime, **assembled on the
already-validated Phase-2/3 primitives** (thin SVD, symmetric eig, covariance,
GEMM, reductions). Each estimator `fit`s and exposes the fitted attributes named
in the success criteria, matching scikit-learn within **1e-5** after `svd_flip`
sign alignment. This phase exercises the full **Arrow → kernel →
device-state → materialize → oracle** pipeline with **no convergence risk** —
the iterative linear-algebra risk was absorbed by the Phase-3 SVD/eig hard gate.

Covers requirements **LINEAR-01, LINEAR-02, DECOMP-01, DECOMP-02**.

**Scope anchors (carried forward — NOT re-decided):**
- **Rust-only this phase.** PyO3 `#[pyclass]` bindings, the Python `fit`/`predict`/
  `transform`/`score` surface, `get_params`/`set_params`, Arrow PyCapsule ingest,
  and per-backend wheels are **Phase 6**. Phase 4 ships the Rust estimator layer
  + Rust oracle tests **only**. The Rust API shape is chosen now (D-04) so Phase 6
  wraps it cleanly, but no Python is written this phase.
- **Gate = cpu + rocm (D-07, from Phase 3).** The ROADMAP §"Phase 4" success
  criterion 1 wording "via cpu and **wgpu**" is **superseded** by Phase-3 D-07:
  f64 validates on **cpu**, f32 on **rocm**, f64-on-rocm **skips-with-log**
  (cubecl-cpp 0.10 does not register F64 for the HIP backend — expected, not a
  defect). wgpu is opportunistic only. ROADMAP/PROJECT still carry the old
  cpu+wgpu wording in places — planner should read "cpu+wgpu" as "cpu+rocm".

`mlrs-algos/src/lib.rs` is currently an empty skeleton ("estimators land in
Phase 4+"). This phase fills it. The workspace DAG already reserves the crate's
place, so **only `mlrs-algos` (and its tests) plus possibly one new primitive in
`mlrs-backend` — see D-02** — are edited here.

</domain>

<decisions>
## Implementation Decisions

### PCA solver path (DECOMP-01)
- **D-01:** **PCA `full` = SVD of centered X** (NOT eig-of-covariance). Match
  scikit-learn's *actual* `svd_solver='full'` arithmetic: center X by column
  means → run the Phase-3 **thin SVD** on the centered matrix → derive
  `explained_variance_ = S²/(n−1)`, `explained_variance_ratio_ =
  explained_variance_ / total_variance`, `components_ = Vᵀ` (after `svd_flip`),
  `singular_values_ = S`, `mean_ = column means`, `transform(X) = (X − mean)·V`,
  `inverse_transform(Z) = Z·Vᵀ + mean`. Chosen over the eig-of-covariance path
  (which Phase 3 D-01/D-06 *anticipated*) because SVD-of-centered-X is more
  numerically faithful to sklearn's literal arithmetic and more robust near
  rank-deficiency — the strongest way to hold 1e-5.
  - **Consequence:** the Phase-3 **eig primitive is NOT consumed by PCA** in v1.
    It remains a validated standalone primitive (reusable later); do not treat
    its non-use as a defect. The **SVD primitive is the PCA workhorse**.
  - The `svd_flip` convention is applied **by the estimator** to canonicalize
    `components_`/`transform` to match sklearn (D-03 P3: the primitive stays raw;
    the estimator flips). Reuse `mlrs-core/src/sign_flip.rs` (`align_rows`).

### Ridge solver (LINEAR-02)
- **D-02:** **Ridge = Cholesky normal-equations.** Solve
  `(XᵀX + αI)·coef = Xᵀy` via Cholesky factorization + triangular solve, matching
  scikit-learn's dense `solver='auto'`→Cholesky default. Reuses the Phase-2
  covariance/Gram (`XᵀX`) primitive for the normal matrix.
  - **⚠ IN-PHASE PRIMITIVE CONSEQUENCE (planner MUST scope this):** this path
    needs a **NEW linear-solve primitive** — Cholesky factorization of an SPD
    matrix + triangular (forward/back) substitution — which does **NOT** exist in
    the Phase-2/3 primitive set. This is an explicit Phase-4 sub-deliverable
    living in `mlrs-backend/src/prims/` (e.g. `cholesky.rs` / `solve.rs`),
    feature-free `#[cube]` kernel in `mlrs-kernels`, validated standalone (f32+f64,
    cpu+rocm) against a numpy/sklearn reference + algebraic invariant
    (`‖L·Lᵀ − A‖`, `‖A·x − b‖`) before Ridge consumes it. Treat it with the same
    primitive-first rigor as Phase 2/3 (memory gate, tolerance policy) even though
    it lands inside an estimator phase. **This is the single highest implementation
    risk in Phase 4** — flag for research depth in `/gsd-plan-phase`.
  - **LinearRegression (LINEAR-01) is SEPARATE and SVD-based** (pinned by the
    requirement: "SVD-based to match sklearn's default lstsq"). It uses the
    Phase-3 SVD pseudo-inverse `coef = V·diag(1/σ)·Uᵀ·y` (with sklearn's
    small-singular-value cutoff), NOT the Cholesky path. So the two linear models
    deliberately use different solvers — do not unify them.

### Fitted-state representation
- **D-03:** **Fitted attributes are device-resident (`DeviceArray`).** `coef_`,
  `intercept_`, `components_`, `mean_`, `singular_values_`, `explained_variance_`,
  etc. stay on-device as `DeviceArray<ActiveRuntime, F>` after `fit`.
  `predict`/`transform`/`inverse_transform` run **device-side** with no host
  round-trip; host materialization (to `Vec<F>`) happens **lazily** only when a
  Rust accessor is called or at oracle-comparison time. Satisfies the first-class
  memory-efficiency requirement, keeps the pipeline device-resident end-to-end,
  and sets up the Phase-6 zero-copy Arrow handoff. Host-materialize-at-fit was
  rejected (extra copies, breaks device-residency, fights the memory gate).
  - **Memory gate extends here:** the Phase-2 D-10 / Phase-3 D-11 build-failing
    PoolStats gate applies — `fit` + a `predict`/`transform` round must not host
    round-trip mid-pipeline, fitted-state buffers are pool-managed, and repeated
    same-shape `transform` calls drive reuse (bounded allocation). Assert it.

### Estimator API shape (consumed by Phase-6 PyO3 wrapping)
- **D-04:** **Shared traits — `Fit`, `Transform`, `Predict`** (sklearn-mixin-style)
  defined in `mlrs-algos`. Each estimator implements the traits relevant to it
  (`LinearRegression`/`Ridge`: `Fit` + `Predict`; `PCA`: `Fit` + `Transform`
  [+ inverse]; `TruncatedSVD`: `Fit` + `Transform`). Gives a uniform surface so
  Phase-6 PyO3 wrapping is generic, future Phase-5 estimators slot in, and the
  shape mirrors sklearn's mixin pattern (`RegressorMixin`/`TransformerMixin`).
  `fit` returns `&mut self` / `self` (sklearn convention: `fit` returns the
  estimator). Standalone-structs-without-traits was rejected (Phase 5 adds 7 more
  estimators — the shared surface pays off).

### Decided defaults (bounded by requirements — not full gray areas)
- **D-05:** **Intercept via center-then-solve.** `LinearRegression`/`Ridge` with
  `fit_intercept=true` (default) center X and y by column means, solve for `coef_`
  on centered data, then recover `intercept_ = ȳ − x̄·coef_` — sklearn's exact
  procedure. Reuses the Phase-2 column-mean reduction. (Ridge does NOT penalize
  the intercept — centering handles this automatically, matching sklearn.)
- **D-06:** **v1 constructor surface is minimal, `n_components` is an integer.**
  `LinearRegression { fit_intercept }`, `Ridge { alpha, fit_intercept }`,
  `PCA { n_components }`, `TruncatedSVD { n_components }`. `n_components` accepts
  an **int only** in v1 (`k ≤ min(n_samples, n_features)`). sklearn's float
  (variance-ratio), `'mle'`, and `None`-means-all semantics are **deferred** (see
  Deferred Ideas). Other sklearn knobs (`copy_X`, `tol`, `normalize`, `whiten`,
  random-state for randomized SVD) are out of v1 scope.
- **D-07:** **Oracle source = scikit-learn fixtures** (not bare numpy) for
  estimator-specific attributes. Phase 3 used numpy fixtures for the raw SVD/eig;
  Phase 4 needs **sklearn** as the reference because the contract includes
  sklearn-specific behaviors: Ridge intercept handling, `explained_variance_ratio_`,
  `svd_flip` canonicalization, and **TruncatedSVD's deterministic `algorithm='arpack'`
  path** (NOT the default `'randomized'`, which is non-deterministic). Generate
  committed fixtures via `scripts/gen_oracle.py` (reuse Phase-1 infra; regen needs
  a /tmp venv with numpy+scikit-learn per PEP 668 — fixtures are committed blobs,
  not test-time). Compare after `align_rows` sign alignment; hold the global 1e-5
  abs+rel policy with the per-family looser bound activating only if a real case
  forces it (carries Phase-3 D-10).

### Carried forward from Phases 1–3 (reaffirmed, not re-decided)
- **D-08:** Estimators generic over `<F: Float + CubeElement + Pod>`; thin SVD
  returns (U[m×k], S[k], Vᵀ[k×n]) with k=min(m,n) (P3 D-02); `eig` returns
  descending (w, V) (P3 D-04); `covariance` takes `ddof` (P2 D-09); explicit
  `(rows, cols)` per call, `DeviceArray` stays flat 1D (P2 D-04); device-resident
  in/out (P2 D-05); optional caller-out + pooled scratch (P2 D-11); `svd_flip`
  canonicalizes at comparison/estimator time, primitives stay raw (P3 D-03).
  Feature-free `#[cube]` kernels in `mlrs-kernels`; launch wrappers + host
  orchestration in `mlrs-backend`; estimators in `mlrs-algos` consume the
  `mlrs-backend` primitive API. `assert_close` 1e-5 abs+rel with near-zero floor;
  f64 capability-gated via `skip_f64_with_log`; `thiserror` in libs / `anyhow` at
  boundaries; deps track latest; source/test separation per AGENTS.md (no in-source
  `mod tests` — tests in `crates/*/tests/`).

### Claude's Discretion
- Module/file layout within `mlrs-algos` (e.g. `linear/`, `decomposition/` modules
  or per-estimator files) and the exact trait method signatures (D-04) — honor
  source/test separation.
- The new Cholesky/solve primitive's internal design (D-02): blocked vs
  unblocked Cholesky, in-place vs out-of-place, triangular-solve kernel structure
  — subject to the Phase-2/3 memory gate, tolerance policy, and no-hardcoded-
  plane-width rule. Researcher-flagged.
- LinearRegression's small-singular-value cutoff constant (D-02) to match
  sklearn/`scipy.linalg.lstsq` `rcond` default — pick the value that holds 1e-5.
- Exact random shapes/seeds for the estimator oracle sweep, and which cases get
  committed sklearn fixtures vs algebraic-invariant-only checks (D-07).
- Naming of new estimator/primitive error variants (extend the `thiserror` enums).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
  (NB: documents the cpu+wgpu gate — D-07 supersedes it with cpu+rocm; the
  ROADMAP Phase-4 criterion 1 "cpu and wgpu" wording is likewise superseded)
- `.planning/REQUIREMENTS.md` — LINEAR-01, LINEAR-02, DECOMP-01, DECOMP-02
  requirement text + traceability table
- `.planning/ROADMAP.md` §"Phase 4: Closed-Form Estimators" — goal + 4 success
  criteria (the gate for this phase)
- `.planning/phases/03-svd-eigendecomposition-primitive-hard-gate/03-CONTEXT.md`
  — Phase-3 SVD/eig primitive decisions this phase consumes (thin extent D-02,
  raw-output/sign-at-comparison D-03, descending order D-04, tall+wide D-05,
  the cpu+rocm gate D-07, tolerance D-10, memory gate D-11)
- `.planning/phases/02-core-compute-primitives/02-CONTEXT.md` — GEMM transpose
  flags (D-06), covariance/Gram + ddof (D-09), device-resident in/out (D-05),
  the D-10/D-11 memory gate this phase extends, optional-out + pooled scratch
- `.planning/phases/01-foundation-oracle-backend-abstraction-arrow-bridge/01-CONTEXT.md`
  — oracle harness, `sign_flip` helper (FOUND-08), capability gating, tolerance policy

### Build / kernel protocol (MANDATORY before writing any CubeCL code)
- `AGENTS.md` — source/test separation; CubeCL generics-over-float requirement;
  build-error protocol
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL
  manual index; read before writing the new Cholesky/solve kernel (D-02)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — generics, plane/
  subgroup, shared-memory, matmul/gemm manuals (the Cholesky + triangular-solve
  kernel needs these)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  — MANDATORY troubleshooting reference on ANY CubeCL build/compile/feature/
  toolchain error (the new primitive must run on the rocm/HIP gate)

### Memory-efficiency guidance (informs the D-03 device-resident state + memory gate)
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL,
  buffer-reuse patterns (device-resident fitted state, bounded scratch)

### Reference implementation (read-only — behavior/convention reference)
- `cuml-main/` — RAPIDS cuML v26.08.00; LinearRegression/Ridge/PCA/TruncatedSVD
  solver behavior reference (NOT code to port verbatim; numerical agreement is
  with **scikit-learn**, not cuML)
- scikit-learn source/docs for `LinearRegression`, `Ridge` (solver='cholesky'/'svd'),
  `PCA` (svd_solver='full'), `TruncatedSVD` (algorithm='arpack') — the oracle contract
- `.planning/codebase/*.md` — codebase maps (ARCHITECTURE, CONVENTIONS, STACK,
  TESTING, STRUCTURE, INTEGRATIONS, CONCERNS)

### Existing source this phase consumes / extends
- `crates/mlrs-algos/src/lib.rs` — currently an empty skeleton; this phase fills it
- `crates/mlrs-backend/src/prims/svd.rs` — thin SVD `svd()` (PCA workhorse D-01,
  LinearRegression pseudo-inverse D-02); `svd_with_max_sweeps` test hook
- `crates/mlrs-backend/src/prims/covariance.rs` — `covariance(ddof)` → Gram XᵀX
  for the Ridge normal matrix (D-02)
- `crates/mlrs-backend/src/prims/{eig.rs, gemm.rs, reduce.rs, distance.rs}` —
  eig (validated, NOT consumed by PCA in v1 per D-01), GEMM, column-mean reduction
  (centering D-05)
- `crates/mlrs-core/src/sign_flip.rs` — `align_rows`/`align_sign` (svd_flip applied
  by the estimator, D-01/D-03)
- `crates/mlrs-core/src/{oracle.rs, compare.rs, tolerance.rs, label_perm.rs}` —
  oracle harness, `assert_close`, 1e-5 policy (D-07)
- `crates/mlrs-backend/src/{device_array.rs, pool.rs}` — DeviceArray + BufferPool +
  PoolStats (the D-03 memory gate asserts on these)
- `crates/mlrs-backend/src/{runtime.rs, capability.rs}` — `ActiveRuntime` (rocm/HIP
  under `--features rocm`), `skip_f64_with_log` (f64-on-rocm skips-with-log, D-07)
- `crates/mlrs-backend/tests/memory_gate_test.rs` — the hard PoolStats gate to extend
- `crates/mlrs-core/examples/gen_fixture.rs` + `scripts/gen_oracle.py` — oracle
  fixture generation (D-07: now needs scikit-learn in the /tmp venv, not just numpy)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Thin SVD (Phase 3):** the workhorse for BOTH PCA (`SVD of centered X`, D-01)
  and LinearRegression (SVD pseudo-inverse, D-02) and TruncatedSVD (thin SVD of
  uncentered X). One primitive, three consumers.
- **covariance/Gram (Phase 2, ddof):** the Ridge normal matrix `XᵀX` (D-02);
  reuse its output buffer for the Cholesky factorization (memory gate).
- **column-mean reduction (Phase 2):** centering for PCA `mean_` and the
  LinearRegression/Ridge intercept recovery (D-05).
- **`sign_flip::align_rows` (Phase 1):** the estimator-side `svd_flip` for
  `components_`/`transform` canonicalization (D-01/D-03).
- **Oracle harness + npz loader + `assert_close` (Phase 1):** reused for the
  sklearn fixtures (D-07); the per-family looser bound (Phase-3 D-10) is the
  escape hatch if an ill-conditioned case can't hold 1e-5.
- **DeviceArray + BufferPool + PoolStats:** device-resident fitted state (D-03)
  and the build-failing memory gate the estimators extend.

### Established Patterns
- Feature-free kernels in `mlrs-kernels`; runtime-bound launch wrappers in
  `mlrs-backend/prims/`; estimators in `mlrs-algos` consume the `mlrs-backend`
  primitive API — the **new Cholesky/solve primitive (D-02) honors the same
  split** even though it lands in an estimator phase.
- scikit-learn/numpy/LAPACK conventions are the contract (svd_flip, descending
  order, explained_variance ddof=1, arpack determinism), NOT cuML's.
- The per-phase build-failing memory gate (P2 D-10 / P3 D-11) is the verification
  surface; estimators extend it to fit→predict/transform pipelines (D-03).

### Integration Points
- **`mlrs-algos` was empty until now** — Phase 4 is its first real content. The
  workspace DAG already places it downstream of `mlrs-backend`, so the estimator
  layer only edits `mlrs-algos` + the one new primitive in `mlrs-backend`.
- **The new Cholesky/solve primitive (D-02) is the critical-path risk** — it must
  pass its own standalone validation (f32+f64, cpu+rocm) BEFORE Ridge consumes it,
  mirroring the Phase-2/3 primitive-first discipline. Sequence it first among the
  Ridge plans.
- **Phase 6 (PyO3) consumes the D-04 trait surface** — the `Fit`/`Transform`/
  `Predict` traits and device-resident state (D-03) are chosen to make the
  Phase-6 zero-copy Arrow + GIL-release wrapping straightforward.

</code_context>

<specifics>
## Specific Ideas

- **PCA explicitly does SVD of centered X (D-01), NOT eig of covariance** — even
  though Phase 3 built the eig primitive "for the PCA full solver path." The user
  chose the more sklearn-faithful, more robust path. The eig primitive stays a
  validated, unused-in-v1 asset; this is intentional, not an oversight.
- **The two linear models use deliberately different solvers:** LinearRegression
  = SVD pseudo-inverse (D-02, pinned by LINEAR-01); Ridge = Cholesky
  normal-equations (D-02). Do not unify them into one solver path.
- **TruncatedSVD oracle uses `algorithm='arpack'`** for determinism (D-07), not
  the sklearn default `'randomized'`. components_/singular_values_/transform from
  thin SVD of *uncentered* X; explained_variance_ = variance of the transformed
  columns.
- Device-resident fitted state (D-03) is the visible signal that estimators
  compose on-device like the Phase-2/3 primitive pipeline — only accessor calls
  and oracle comparison read back to host.

</specifics>

<deferred>
## Deferred Ideas

- **`n_components` as float (variance-ratio), `'mle'`, or `None`=all** — sklearn
  PCA supports these; v1 takes an int only (D-06). Add when the Python surface
  (Phase 6) needs sklearn-API parity, or earlier if a consumer requires it.
- **Additional sklearn constructor knobs** — `copy_X`, `tol`, `whiten` (PCA),
  randomized SVD + `random_state` (PCA/TruncatedSVD `algorithm='randomized'`),
  `positive`/`normalize` (linear models). Out of v1 scope (D-06); revisit per
  Phase-6 estimator-checks needs.
- **Ridge alternative solvers** (`svd`, `lsqr`, `sag`, `saga`, `sparse_cg`) —
  v1 ships Cholesky only (D-02). sklearn exposes a `solver=` choice; defer the
  multi-solver surface to a later milestone.
- **PCA via eig-of-covariance as a selectable solver** — rejected as the v1
  workhorse (D-01) but the eig primitive exists; could become an alternate
  `svd_solver='covariance_eigh'`-style path later (sklearn added one) if a
  many-features case makes the covariance path cheaper.
- **Reusing the new Cholesky/solve primitive elsewhere** — it could serve future
  GLM/Gaussian-process/Mahalanobis paths; built generically here, no v1 consumer
  beyond Ridge.

### Reviewed Todos (not folded)
None — no pending todos matched this phase.

## Open Questions for Research (run `/gsd-plan-phase --research-phase 4`)
- **Cholesky/triangular-solve primitive in CubeCL 0.10 (D-02)** — the highest
  Phase-4 risk. Blocked vs unblocked Cholesky expressible in `#[cube]`; numerical
  stability for the `(XᵀX + αI)` SPD system; does it run f32 on rocm/HIP and f64
  on cpu within 1e-5; memory-gate compliance (reuse the Gram buffer). Validate
  standalone before Ridge consumes it.
- **LinearRegression rcond / small-σ cutoff (D-02)** — the singular-value
  threshold that reproduces `scipy.linalg.lstsq` / sklearn `LinearRegression`
  within 1e-5, including rank-deficient X.
- **PCA explained_variance edge cases (D-01)** — confirm `S²/(n−1)` +
  `explained_variance_ratio_` + `svd_flip` hold 1e-5 vs sklearn across tall/wide/
  n_features>n_samples, including when `n_components < min(m,n)` (truncation).
- **TruncatedSVD arpack determinism (D-07)** — confirm the committed `arpack`
  fixtures are reproducible and the thin-SVD-of-uncentered-X path matches them
  after sign alignment.

</deferred>

---

*Phase: 4-Closed-Form Estimators*
*Context gathered: 2026-06-12*
