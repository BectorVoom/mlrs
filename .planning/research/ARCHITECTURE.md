# Architecture Patterns — v2.0 Breadth Sweep

**Domain:** sklearn-compatible ML estimator library (Rust/CubeCL rewrite of cuML), v2.0 breadth sweep
**Researched:** 2026-06-14
**Supersedes for v2:** the v1.0 `ARCHITECTURE.md` (2026-06-11) covered crate layout / generic-over-runtime / Arrow zero-copy and is now VALIDATED (shipped). This file is the v2 integration architecture: how the ~16 new estimators + 5 new primitives slot into that shipped layering. The v1 architecture is REUSED, not re-researched.

This file answers: how do the v2 estimators and their new primitives integrate into the existing
five-crate layering and trait surface — what is NEW, what is MODIFIED, in what build order, and
how each threads through dispatch / shim / oracle. Every claim is grounded in the actual shipped
code read this pass (`crates/*/src/**`, `scripts/gen_oracle.py`).

> Confidence: **HIGH** for placement / host-API signature shape / trait deltas / dispatch / shim
> mixins / oracle (every claim mirrors a shipped file). **MEDIUM** for the incremental-SVD merge
> stability, SGD-under-cpu-MLIR fit, and smallest-eigenpair approach — the genuine unknowns in
> `research/questions.md` that need a per-phase research spike before planning P7/P9/P10.

---

## Recommended Architecture

### The fixed five-crate seam (REUSE — do not change)

```
mlrs-kernels   #[cube] generic-float kernels, BACKEND-FEATURE-FREE (no ActiveRuntime)
      │            new: kernel_matrix elementwise, laplacian helpers, sgd update, nb reductions
      ▼
mlrs-backend   prims/*  validate-geometry → unsafe launch → Result<_, PrimError>
      │            owns ActiveRuntime, BufferPool, DeviceArray; the ONLY kernel launch site (D-13)
      │            new: prims/rng.rs, prims/kernel_matrix.rs, prims/incremental_svd.rs,
      │                 prims/laplacian.rs, prims/sgd.rs (+ reuse reduce/distance/eig/cholesky/gemm)
      ▼
mlrs-algos     estimator structs<F>; impl Fit/Predict/Transform/PredictLabels/PredictProba
      │            COMPOSE prims, never launch kernels directly
      │            new modules: covariance/, projection/, kernel/, manifold/, naive_bayes/
      │                         (+ spectral in cluster/, + mbsgd & svm in linear/)
      ▼
mlrs-py        #[pyclass] via any_estimator! enum (Unfit/F32/F64); dtype-suffixed accessors;
      │            py.detach + guard_f64; pure-Python sklearn shim (python/mlrs/*.py)
      ▼
scripts/gen_oracle.py + tests/fixtures/*.npz  (committed blobs, no Python at test time)
```

The dependency arrows are **acyclic and unchanged**. Every v2 addition is a *new file plus a
`pub mod` / `pub use` line* in the relevant crate root; no v2 work edits a v1 estimator file
(the file-disjoint, parallel-safe discipline from v1 holds). The single shared-edit point is
`mlrs-py/src/lib.rs` (pyclass registration) and the family-module `mod.rs` files.

### Two structural decisions the roadmapper must make up front

1. **Promote host SplitMix64 to `prims/rng.rs` vs. a device RNG kernel.** v1 already has a
   host-side seeded PRNG inside `prims/kmeans.rs::kmeanspp_sample` (read back once per center,
   never `OsRng` — ASVS V6). RandomProjection needs a full `n_features × n_components` matrix.
   **Recommendation: host-generate-then-upload** (SplitMix64/Philox on host → single
   `BufferPool` upload), promoted into a shared `prims/rng.rs` so RandomProjection, future SGD
   shuffling, and any later sampler reuse one seeded generator. A device RNG kernel is not worth
   it at v2 sizes and would fight the cpu-MLIR no-atomics constraint. (Resolves `[v2-P1] RNG`.)

2. **Smallest-eigenpairs for spectral.** v1 `eig` (Jacobi) returns the *full* descending
   spectrum. SpectralEmbedding/Clustering need the *smallest* nontrivial eigenvectors of the
   Laplacian. **Recommendation: full-spectrum-then-take-smallest** at v2 problem sizes (no
   Lanczos/shift-invert) — it reuses the validated `eig` prim verbatim; the only new code is a
   host-side "drop the trivial near-zero eigenvector, take the next k ascending" slice in the
   estimator. Flag a problem-size cap in PITFALLS. (Resolves the `[v2-P3]` default.)

---

## (1) New primitives — placement + host-API signature shape

Every prim follows the shipped contract seen in `prims/covariance.rs` and `prims/distance.rs`:
**`pub fn name<F: Float + CubeElement + Pod>(pool, inputs, geometry, params, out: Option<DeviceArray>) -> Result<DeviceArray, PrimError>`** — `validate_geometry(...)` called BEFORE any
`unsafe` launch, `out`-buffer reuse via `BufferPool`, zero host round-trips (the device-residency
grep gate). Any new kernel lives in `mlrs-kernels/src/` and is re-exported from its `lib.rs`; the
prim wrapper is the launch site.

| New prim | Kernel (mlrs-kernels) | Prim file (mlrs-backend/src/prims) | Signature shape | Composition / new-kernel? |
|---|---|---|---|---|
| **RNG matrix** | none (host PRNG) | `rng.rs` | `fn random_matrix<F>(pool, shape:(usize,usize), kind: RngKind {Gaussian, SparseAchlioptas{density}}, seed:u64, out:Option<_>) -> Result<DeviceArray<F>, PrimError>` | Host SplitMix64/Philox fills `Vec<F>` → single pool upload. **No device kernel.** Promotes v1's kmeans++ host PRNG. |
| **Kernel matrix** | extend `elementwise.rs` (exp/poly/tanh map) | `kernel_matrix.rs` | `fn kernel_matrix<F>(pool, x,(nx,d), y,(ny,d), kind: KernelKind {Linear, Rbf{gamma}, Poly{degree,gamma,coef0}, Sigmoid{gamma,coef0}}, out) -> Result<DeviceArray<F>, PrimError>` | Linear = `gemm(transb=true)`; RBF = `distance(sqrt=false)`→`exp(-γ·d²)`; Poly/Sigmoid = `gemm`→`(γG+c0)^deg`/`tanh`. **One small elementwise map kernel**, NO SharedMemory/atomics (resolves `[v2-P2]`). |
| **Incremental SVD merge** | none (glue over `svd`/`gemm`) | `incremental_svd.rs` | `fn isvd_merge<F>(pool, prev:(components,S,mean,n_seen), x_batch,(nb,d), out) -> Result<(components, S, mean), PrimError>` | sklearn `IncrementalPCA._fit`: stack `[√(n/seen)·diag(S)·components ; X_batch_centered]`, `svd` it (reuse v1 `svd`). **Host-light glue**; open risk = f32-on-rocm stability of the stacked merge (`[v2-P1]`). |
| **Graph Laplacian** | extend `elementwise.rs` (D^{-1/2} scale) | `laplacian.rs` | `fn laplacian<F>(pool, affinity,(n,n), norm: LapNorm {Unnormalized, SymmetricNorm, RandomWalk}, out) -> Result<DeviceArray<F>, PrimError>` | Affinity = `kernel_matrix(Rbf)` (reuse P8 prim). Degree = `row_reduce(Sum)`; `L=D−W` or `I−D^{-1/2}WD^{-1/2}`. **Small reduce+elementwise composition.** |
| **SGD solver** | `sgd.rs` (per-minibatch grad/update, GATHER idiom) | `sgd.rs` | `fn sgd_fit<F>(pool, x,(n,d), y, loss: SgdLoss {Hinge, Log, SquaredLoss, EpsilonInsensitive{eps}}, penalty:{kind,alpha,l1_ratio}, lr_schedule, max_iter, tol, n_iter_no_change, seed, fit_intercept) -> Result<(coef, intercept), PrimError>` | The ONE genuinely new solver. Host owns the epoch/shuffle/LR-schedule loop (like `prims/lbfgs.rs`'s host driver); device does the per-minibatch dot/update. **Must fit the v1 GATHER idiom** (no SharedMemory / no cross-unit atomics — `[v2-P4]`). |

**Pattern to follow (from `covariance.rs`):** `validate_geometry → reduce/gemm/distance compose →
in-place scale on the REUSED out buffer → return the exact out handle`. Each new prim needs its
own `tests/<prim>_test.rs` with the PoolStats memory gate (bounded reuse, out-buffer reuse,
`read_backs == 0` inside the prim) — the v1 per-prim gate discipline continues.

**Anti-pattern to avoid:** launching a kernel from `mlrs-algos` (estimators compose prims only —
D-13). Also avoid a caller-visible `ReducePath` parameter on INTERNAL reductions (the `CR-01`
plane-path-`None` panic on cpu — see `covariance.rs`: internal reductions are unconditionally
`ReducePath::Shared`).

---

## (2) New / modified traits

v1 trait surface (`mlrs-algos/src/traits.rs`): `Fit`, `Predict<F>` (continuous), `Transform<F>`
(+ default-`Unsupported` `inverse_transform`), `PredictLabels<F>` (i32), `KNeighbors<F>`,
`PredictProba<F>`. v2 needs **two new traits and reuses the rest**.

| Estimator family | Trait surface | NEW or REUSE |
|---|---|---|
| EmpiricalCovariance, LedoitWolf | `Fit` only + host accessors (`covariance_`, `location_`, `precision_`) | **REUSE `Fit`.** No `Predict`/`Transform`. A "covariance estimator without predict" is just `Fit` + accessors — **no new trait.** |
| IncrementalPCA | `Fit` + `Transform` + **`PartialFit`** | **NEW `PartialFit<F>`**: `fn partial_fit(&mut self, pool, x, y:Option, shape) -> Result<&mut Self, AlgoError>` — same shape as `Fit::fit` but accumulates running SVD state instead of resetting (`Fit::fit` = `partial_fit` over one batch). |
| Gaussian/SparseRandomProjection | `Fit` + `Transform` | **REUSE.** `fit` builds the random matrix from `n_features` (+seed); `transform` = `gemm(X, Rᵀ)`. No `inverse_transform`. |
| KernelRidge | `Fit` + `Predict` | **REUSE.** `fit` solves `(K+αI)·dual = y` via `cholesky_solve`; `predict` = `kernel_matrix(Xq,Xfit)·dual`. |
| KernelDensity | `Fit` + **`ScoreSamples`** | **NEW `ScoreSamples<F>`**: `fn score_samples(&self, pool, x, shape) -> Result<DeviceArray<F>, AlgoError>` → length-`n` log-densities (sklearn `KernelDensity.score_samples`; not `Predict` semantics). |
| SpectralEmbedding | `Fit` + `Transform` (transductive) | **REUSE `Transform`.** `fit` computes affinity→Laplacian→eig, stores embedding; `transform` returns it. sklearn SpectralEmbedding has no out-of-sample transform; shim exposes `fit_transform` via `TransformerMixin`. |
| SpectralClustering | `Fit` + `PredictLabels` | **REUSE `PredictLabels`** (i32, like KMeans/DBSCAN). `fit` = embedding → KMeans on embedding; labels stored. |
| MBSGDClassifier, LinearSVC | `Fit` + `PredictLabels` (+ `PredictProba` for log-loss MBSGDClassifier) | **REUSE.** Hinge/log via SGD prim. |
| MBSGDRegressor, LinearSVR | `Fit` + `Predict` (continuous) | **REUSE.** Squared / ε-insensitive via SGD prim. |
| 5× NaiveBayes | `Fit` + `PredictLabels` + `PredictProba` | **REUSE** all three. Pure reductions; proba = normalized class posteriors, labels = argmax. |

**Trait deltas summary:**
- **NEW:** `PartialFit<F>` (IncrementalPCA), `ScoreSamples<F>` (KernelDensity). Both placed in
  `traits.rs` next to existing traits, same `<F: Float + CubeElement + Pod>` bound, same
  `(pool, x, shape)` convention, errors via `AlgoError`. Re-export from `mlrs-algos/src/lib.rs`'s
  `pub use traits::{...}` (one-line MODIFY).
- **REUSE unchanged:** `Fit`, `Predict`, `Transform`, `PredictLabels`, `PredictProba`.
- **Covariance estimators need no new trait** — `Fit` + accessors, just don't impl `Predict`.

---

## (3) Threading through dispatch + accessors + shim mixins

### PyO3 layer (`mlrs-py/src/`)

Each estimator follows the **three-part pattern** in `estimators/decomposition.rs`:

1. **`any_estimator! { any: AnyX, algo: mlrs_algos::family::X, unfit: { hp: ty, ... } }`** — emits
   the `Unfit{..}/F32(X<f32>)/F64(X<f64>)` enum (REUSE `dispatch.rs` macro verbatim).
2. **`#[pyclass(name="SklearnName")] struct PyX { inner: AnyX }`** + `#[pymethods]`: `#[new]`
   stores hyperparameters into `Unfit`; `fit` does `float_dtype → match → guard_f64()` on the
   F64 arm BEFORE upload → `py.detach(|| { global_pool().lock(); est.fit(...) })`.
3. **dtype-suffixed accessors** (`coef_f32`/`coef_f64`, `transform_f32`/`_f64`, …) because a
   `#[pyclass]` method can't be generic over `F`; plus `dtype()`, `is_fitted()`, and the
   trait-specific method (`predict_labels` is i32 / unsuffixed; `predict_proba_f32/f64`).

New estimator modules in `mlrs-py/src/estimators/`: `covariance.rs`, `projection.rs`, `kernel.rs`,
`manifold.rs`; add spectral-clustering to `cluster.rs`, MBSGD/LinearSVM to `linear.rs`, new
`naive_bayes.rs`. Register every `#[pyclass]` in `mlrs-py/src/lib.rs` module init (the one shared
MODIFIED file).

**Per-family dispatch notes:**
- Covariance: `fit(x, rows, cols)` unsupervised (no `y`); accessors `covariance_f32/f64`,
  `location_f32/f64`, `precision_f32/f64`. No `predict`.
- IncrementalPCA: add `partial_fit(x, rows, cols)` `#[pymethod]` — first call picks dtype arm,
  later calls dispatch to the fitted arm's `PartialFit`; enforce dtype-consistency across batches
  (new guard).
- RandomProjection: `fit(rows, cols)` data-independent (matrix from `n_features`+seed); expose
  `components_f32/f64` and `transform_f32/f64`.
- KernelDensity: `score_samples_f32/f64`. KernelRidge: `predict_f32/f64`, `dual_coef_f32/f64`.
- NaiveBayes: `predict_labels` (i32, unsuffixed), `predict_proba_f32/f64`, `n_classes()` int
  accessor (mirror LogisticRegression's `classes_ = arange(n_classes())`).

### Pure-Python shim (`mlrs-py/python/mlrs/`)

`MlrsBase` (REUSE unchanged) already gives `_normalize`/`_normalize_y`, `_to_output`,
`_suffix()`/`_suffixed()`/`_np_float()` dtype routing, `_post_fit(n_features)`, `_check_fitted`,
and `__sklearn_tags__` (sparse/nan/array-api off). Each shim = `MlrsBase` + the sklearn family
mixin; the `fit` body is the v1 boilerplate verbatim (`_normalize → _ext().PyX(...) →
obj.fit(...) → self._mlrs_obj = obj → _post_fit(cols) → return self`).

| Family | sklearn mixin(s) | Shim file | Notes |
|---|---|---|---|
| EmpiricalCovariance, LedoitWolf | **none** (bare `MlrsBase`/`BaseEstimator`) | `covariance.py` (new) | `fit(X, y=None)`; `covariance_`/`location_`/`precision_` props. No mixin ⇒ no injected predict/transform. |
| IncrementalPCA | `TransformerMixin` | `decomposition.py` (extend) | add `partial_fit(X, y=None)`; `transform`, `components_`, `mean_`, `explained_variance_`. |
| Gaussian/SparseRandomProjection | `TransformerMixin` | `projection.py` (new) | `fit(X, y=None)` (uses only `X.shape[1]`); `transform`; `components_`. |
| KernelRidge | `RegressorMixin` | `kernel.py` (new) | `fit(X, y)`; `predict`; `dual_coef_`. |
| KernelDensity | **none** (sklearn KDE is bare `BaseEstimator`) | `kernel.py` or `neighbors.py` | `fit(X, y=None)`; `score_samples(X)`; `score(X)`. |
| SpectralEmbedding | `TransformerMixin` | `manifold.py` (new) | `fit_transform(X)` primary; `embedding_`. |
| SpectralClustering | `ClusterMixin` | `cluster.py` (extend) | `ClusterMixin` gives `fit_predict`; `labels_` (int). |
| MBSGDClassifier, LinearSVC | `ClassifierMixin` | `linear.py` (extend) | `classes_=arange(n_classes())`; `predict`→int; LinearSVC no `predict_proba`; MBSGDClassifier(log) has it. |
| MBSGDRegressor, LinearSVR | `RegressorMixin` | `linear.py` (extend) | `predict`→float; `coef_`/`intercept_`. |
| 5× NaiveBayes | `ClassifierMixin` | `naive_bayes.py` (new) | `classes_`; `predict`→int; `predict_proba`; per-variant fitted attrs (`theta_`/`var_` Gaussian; `feature_log_prob_`/`class_log_prior_` discrete). |

**No new shim machinery is required** — `MlrsBase` already covers dtype-suffix routing + output
mirroring. The v1 `__sklearn_tags__` (dense-float-only, nulls rejected) carries over to every
new family.

---

## (4) Suggested build order (primitive-first dependency graph)

The seed roadmap's phase order is sound and respects the prim→estimator graph. Phase numbering
continues from v1 (next = 7).

```
Phase 7  Covariance & projection
   prim: rng.rs (promote host PRNG)         → Gaussian/SparseRandomProjection
   prim: incremental_svd.rs (merge glue)    → IncrementalPCA
   reuse: covariance prim                    → EmpiricalCovariance, LedoitWolf (Fit-only)
   NEW trait: PartialFit                     → IncrementalPCA
   estimators: EmpiricalCovariance, LedoitWolf, IncrementalPCA, GaussianRP, SparseRP

Phase 8  Kernel family
   prim: kernel_matrix.rs (linear/rbf/poly/sigmoid)  → reused by P9 + future SVM
   NEW trait: ScoreSamples                   → KernelDensity
   reuse: cholesky_solve                      → KernelRidge
   estimators: KernelRidge (Fit+Predict), KernelDensity (Fit+ScoreSamples)

Phase 9  Spectral family   [HARD DEP on Phase 8]
   prim: laplacian.rs                         → consumes kernel_matrix(P8) + eig(v1) + kmeans(v1)
   reuse: eig (full-spectrum→take-smallest), kmeans
   estimators: SpectralEmbedding (Fit+Transform), SpectralClustering (Fit+PredictLabels)

Phase 10 SGD / linear-SVM
   prim: sgd.rs (hinge/log/squared/epsilon)   → one new solver; unblocks 4 estimators
   reuse: gemm, reductions
   estimators: MBSGDClassifier (Fit+PredictLabels[+PredictProba]), MBSGDRegressor (Fit+Predict),
               LinearSVC (Fit+PredictLabels), LinearSVR (Fit+Predict)

Phase 11 Naive Bayes
   prim: none (reductions only)
   estimators: Gaussian/Multinomial/Bernoulli/Complement/CategoricalNB
               (all Fit+PredictLabels+PredictProba)
```

**Critical ordering facts:**
- **P8 (kernel_matrix) must precede P9 (spectral)** — the Laplacian's affinity matrix *is*
  `kernel_matrix(Rbf)`. The seed lists them adjacent; make the dep explicit.
- **P7 and P11 are pure assembly** (covariance / reductions already validated) → lowest risk;
  ideal confidence-building bookends.
- **P10 (SGD) is the single new-solver risk** with the most research-gated unknown (`[v2-P4]`
  cpu-MLIR fit) and unblocks the most estimators (4). Budget a research spike before planning it.
- Within a phase, land the **prim + its gate test first**, then the consuming estimators (the v1
  "prim gated, estimator is assembly" pattern).

---

## (5) gen_oracle.py additions + the RandomProjection fixture exception

Each estimator adds a `gen_<name>(seed, dtype)` to `scripts/gen_oracle.py` writing a committed
`tests/fixtures/<name>_<dtype>_seed42.npz` (both f32 + f64), registered in `main()`. Follow v1
conventions: `np.ascontiguousarray(...).astype(dtype)` on every array (the PCA Fortran-order trap,
04-04 Rule-1); store sklearn fitted attrs + a held-out `predict`/`transform`; pin deterministic
solver settings (`algorithm='arpack'` not `'randomized'`; tight `tol` where the estimator
converges deeper than sklearn's default — cf. the LogReg `tol=1e-10` note).

| Estimator | Fixture arrays | Reference | Gate notes |
|---|---|---|---|
| EmpiricalCovariance | `X`, `covariance_`, `location_` | `sklearn.covariance.EmpiricalCovariance` | strict 1e-5; mirrors `gen_covariance`. |
| LedoitWolf | `X`, `covariance_`, `shrinkage_` | `sklearn.covariance.LedoitWolf` | pin shrinkage scalar too. |
| IncrementalPCA | `X` + batch sizes, `components_`, `explained_variance_`, `mean_`, `transform` | `sklearn.decomposition.IncrementalPCA(batch_size)` | **sign-align components rows (`align_rows`)** like PCA; pin batch schedule so the merge order matches. f32-on-rocm merge stability is the watch item. |
| Gaussian/SparseRandomProjection | see exception below | — | **PROPERTY GATE, not value-match.** |
| KernelRidge | `X`, `y`, `Xq`, `alpha`, kernel params, `dual_coef_`, `predict` | `sklearn.kernel_ridge.KernelRidge` | strict 1e-5; pin kernel + gamma. |
| KernelDensity | `X`, `Xq`, `bandwidth`, kernel, `score_samples` | `sklearn.neighbors.KernelDensity` | compare log-densities; pin kernel='gaussian'. |
| SpectralEmbedding | `X`/affinity, `embedding_` | `sklearn.manifold.SpectralEmbedding` | eigenvectors sign+order ambiguous → sign-align AND allow subspace comparison; pin `affinity`, `n_components`, drop trivial eigenvector. |
| SpectralClustering | `X`/affinity, `labels` | `sklearn.cluster.SpectralClustering` | **label-permutation invariant** (reuse v1 `label_perm` helper, like KMeans). |
| MBSGDClassifier/Regressor | `X`, `y`, `Xq`, hp, `coef_`, `intercept_`, `predict`(+`predict_proba`) | `sklearn.linear_model.SGDClassifier/SGDRegressor` | SGD parity is path-dependent (LR schedule, shuffle seed) → may need a SELF-REFERENCE (cf. LogReg binary) or fixed-seed+matched-schedule. Flag in PITFALLS. |
| LinearSVC/SVR | `X`, `y`, `Xq`, `C`, `coef_`, `intercept_`, `predict` | `sklearn.svm.LinearSVC/LinearSVR` (liblinear) | liblinear objective differs slightly → predict/label is the robust gate; `coef_` looser. |
| GaussianNB | `X`, `y`, `Xq`, `theta_`, `var_`, `predict`, `predict_proba` | `sklearn.naive_bayes.GaussianNB` | strict on proba; predict label-exact. |
| Multinomial/Bernoulli/Complement/CategoricalNB | `X` (counts/binary/categorical), `y`, `Xq`, `feature_log_prob_`/`class_log_prior_`, `predict`, `predict_proba` | corresponding `sklearn.naive_bayes.*` | discrete inputs; pin `alpha` smoothing. |

### The RandomProjection exception (explicit)

`GaussianRandomProjection` / `SparseRandomProjection` **cannot be value-matched against sklearn**:
the projection matrix is RNG-drawn and sklearn's NumPy `default_rng` stream is NOT reproducible by
mlrs's host SplitMix64/Philox (the same reason KMeans injects a fixed `init`). So instead of a
`coef_ ≤ 1e-5` value oracle, the test is a **property gate**:

- **Johnson–Lindenstrauss distance preservation:** for the *mlrs-drawn* matrix, pairwise distances
  after projection are preserved within the JL `eps` bound (sklearn's
  `johnson_lindenstrauss_min_dim` is the oracle for the required `n_components`).
- **Matrix distribution:** Gaussian RP — entries `~N(0, 1/n_components)` (mean/var within tol);
  Sparse RP — Achlioptas density + the `±sqrt(s)/sqrt(n_components)` value set at the expected
  nonzero fraction.
- **Seed reproducibility:** same seed → identical matrix across runs/backends (ASVS-V6).
- **Shape/transform correctness:** `transform(X) == X @ components_.T` value-exact against the
  *mlrs* matrix (self-consistency, not a sklearn match).

`gen_random_projection` (if any committed blob at all) stores only the seeded `X` + the
JL/`n_components` params; the test draws the matrix and asserts the *properties*. This is the one
v2 estimator whose correctness gate is structurally different from the 1e-5 value oracle — call it
out loudly in REQUIREMENTS and PITFALLS.

---

## Scalability Considerations

| Concern | v2 oracle sizes | Larger | Notes |
|---|---|---|---|
| Spectral eig | full Jacobi spectrum, take smallest | dense eig O(n³) — cap `n_samples` | Lanczos/shift-invert deferred to v3. |
| Kernel matrix | dense `n×n` | O(n²) memory | OK at v2 sizes; Nyström deferred. |
| IncrementalPCA | batched merge, O(batch·d) memory | the streaming point | f32 merge stability is the watch item. |
| SGD | host epoch loop, device minibatch | GATHER idiom (cpu-MLIR) | averaging/LR-schedule parity is the correctness risk, not throughput. |

---

## Patterns to Follow / Anti-Patterns

**Follow:** prim `validate_geometry → compose prims → in-place scale on reused `out` → return exact
handle` (`covariance.rs`); estimator stores fitted state as device-resident `Option<DeviceArray>`,
host-materializes only at accessors (`pca.rs`); PyO3 `any_estimator!` enum + `py.detach` +
`guard_f64()` before F64 arm (`decomposition.rs`); shim = `MlrsBase` + family mixin, `fit` returns
`self`, `_post_fit(cols)` (`linear.py`/`base.py`); fixtures sign-align via `align_rows` and
label-permute via `label_perm` where the math is gauge/permutation ambiguous.

**Avoid:** kernel launch from `mlrs-algos`; caller-visible `ReducePath` on internal reductions
(cpu plane-path-`None` panic); in-source `#[cfg(test)] mod tests` (AGENTS.md §2 — tests in
`tests/`); SharedMemory/atomics in any new kernel (cpu-MLIR MLIR backend panics — MEMORY note
`cubecl-cpu-no-shared-memory`); `OsRng` (ASVS-V6 — seeded reproducible PRNG only).

---

## Sources

- Shipped code (HIGH): `crates/mlrs-algos/src/traits.rs`,
  `crates/mlrs-backend/src/prims/{mod,covariance,distance,kmeans,lbfgs,cholesky}.rs`,
  `crates/mlrs-py/src/dispatch.rs`, `crates/mlrs-py/src/estimators/decomposition.rs`,
  `crates/mlrs-py/python/mlrs/{base,decomposition,linear}.py`,
  `crates/mlrs-algos/src/{lib.rs,decomposition/pca.rs}`, `crates/mlrs-kernels/src/lib.rs`,
  `scripts/gen_oracle.py`.
- Planning: `.planning/PROJECT.md`, `.planning/seeds/v2-breadth-roadmap.md`,
  `.planning/notes/cuml-mlrs-gap-inventory.md`, `.planning/research/questions.md`.
- cuML reference (read-only): `cuml-main/python/cuml/cuml/{covariance,random_projection,
  kernel_ridge,naive_bayes,manifold,svm,linear_model}/`.
- Open unknowns to resolve before P7/P9/P10 (MEDIUM): `[v2-P1]` incremental-SVD merge stability,
  `[v2-P3]` smallest-eigenpair approach, `[v2-P4]` SGD-under-cpu-MLIR — all in `research/questions.md`.
