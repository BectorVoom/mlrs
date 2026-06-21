---
phase: 09-spectral-family
plan: 04
type: execute
wave: 3
depends_on: ["09-03"]
files_modified:
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-algos/tests/spectral_clustering_test.rs
  - crates/mlrs-py/src/estimators/spectral.rs
  - crates/mlrs-py/tests/spectral_smoke_test.rs
autonomous: true
requirements: [SPECTRAL-02, PRIM-09, SPECTRAL-01]
must_haves:
  truths:
    - "SpectralClustering.fit(X) builds rbf affinity (default, D-01) → laplacian → eig → recovery (drop_first=FALSE, D-11) → KMeans::new → labels_"
    - "labels_ matches sklearn up to label permutation on a well-separated fixture (D-10) — EXACT labels, no band"
    - "n_components defaults to n_clusters (D-11); gamma defaults to literal 1.0 (D-04); KMeans::new is used (NOT with_init, D-10)"
    - "n_samples > 64 is rejected with AlgoError::NSamplesExceedsMaxDim BEFORE any device work (D-06)"
    - "PySpectralEmbedding and PySpectralClustering are #[pyclass]-registered with sklearn-named hyperparameters, f32/f64 dispatch, GIL release, and the f64 guard before upload"
    - "A PyO3 smoke test drives fit + embedding_/labels_ accessors for f32 and f64 (f64 backend-gated)"
  artifacts:
    - path: "crates/mlrs-algos/src/cluster/spectral_clustering.rs"
      provides: "SpectralClustering Fit + labels_ accessor / fit_predict over v1 KMeans"
      contains: "KMeans::new"
    - path: "crates/mlrs-algos/tests/spectral_clustering_test.rs"
      provides: "label_perm exact-label test on the well-separated fixture (un-ignored)"
      contains: "spectral_clustering"
    - path: "crates/mlrs-py/src/estimators/spectral.rs"
      provides: "PySpectralEmbedding/PySpectralClustering any_estimator! wrappers (fit + accessors)"
      contains: "any_estimator"
    - path: "crates/mlrs-py/tests/spectral_smoke_test.rs"
      provides: "PyO3 fit + embedding_/labels_ smoke (f32+f64)"
      contains: "spectral"
  key_links:
    - from: "crates/mlrs-algos/src/cluster/spectral_clustering.rs"
      to: "mlrs_algos::cluster::kmeans::KMeans"
      via: "KMeans::new(n_clusters, seed).fit(maps)"
      pattern: "KMeans::new"
    - from: "crates/mlrs-py/src/estimators/spectral.rs"
      to: "mlrs_algos::cluster::{SpectralEmbedding, SpectralClustering}"
      via: "any_estimator! algo binding"
      pattern: "SpectralClustering"
    - from: "crates/mlrs-py/src/lib.rs"
      to: "crates/mlrs-py/src/estimators/spectral.rs"
      via: "add_class (registered in Wave 0)"
      pattern: "PySpectralClustering"
---

<objective>
SPECTRAL-02 + the PY-06-share PyO3 wrapping: implement `SpectralClustering`
(spectral embedding → v1 KMeans) and wire BOTH spectral estimators into the
`_mlrs` Python surface.

SpectralClustering pipeline (RESEARCH §D-11): rbf affinity (DEFAULT, D-01; gamma
literal 1.0, D-04) → `laplacian` → v1 `eig` → recovery with `n_components =
n_clusters` and `drop_first=FALSE` (KEEP the trivial eigenvector, D-11) →
`KMeans::new(n_clusters, seed)` (kmeans++, n_init=1; NOT `with_init`, D-10) →
`labels_`. The exact-label gate comes from FIXTURE DESIGN (a well-separated
partition is unique → any KMeans converges to the same labels up to permutation,
D-10) — NOT from RNG matching.

The PyO3 layer mirrors `estimators/kernel.rs`: two `any_estimator!` invocations,
`py.detach` GIL release, `guard_f64()?` before upload, dtype-suffixed accessors.
Zero new binding infrastructure (the per-phase incremental wrap; PY-06 final
sign-off stays Phase 11).

Purpose: the spectral clustering label gate + the full Python surface for the phase.
Output: SpectralClustering estimator, label_perm test, PyO3 wrappers, smoke test.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/09-spectral-family/09-RESEARCH.md
@.planning/phases/09-spectral-family/09-PATTERNS.md
@.planning/phases/09-spectral-family/09-VALIDATION.md
@AGENTS.md

# Analogs (READ before editing):
@crates/mlrs-algos/src/cluster/kmeans.rs
@crates/mlrs-algos/src/cluster/spectral_embedding.rs
@crates/mlrs-py/src/estimators/kernel.rs
@crates/mlrs-py/src/estimators/cluster.rs
@crates/mlrs-py/src/dispatch.rs
@crates/mlrs-algos/tests/kmeans_test.rs
@crates/mlrs-algos/tests/dbscan_test.rs

# Phase 9 deps (READ — consumed here):
@crates/mlrs-backend/src/prims/laplacian.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: SpectralClustering (rbf → laplacian → eig → recovery drop_first=FALSE → KMeans) + label_perm test</name>
  <read_first>
    - crates/mlrs-algos/src/cluster/spectral_embedding.rs (the affinity builders + post-eig recovery from Plan 09-03 — reuse the recovery host math)
    - crates/mlrs-algos/src/cluster/kmeans.rs (:112 KMeans::new; :220 Fit; :400 PredictLabels; :449 fit_predict)
    - RESEARCH §D-11 (n_components default n_clusters, drop_first=FALSE, KMeans::new n_init) + Pitfall 5 (RNG flakiness avoided by fixture separation)
    - crates/mlrs-algos/tests/kmeans_test.rs / dbscan_test.rs (best_match_accuracy / label_perm compare)
    - The committed well-separated spectral_clustering .npz fixture (Wave 0, D-10)
  </read_first>
  <behavior>
    - validate-before-launch: reject n_samples > 64 → AlgoError::NSamplesExceedsMaxDim{estimator:"SpectralClustering",..} (D-06); reject n_clusters via InvalidK (1..=n_samples); reject non-finite gamma via InvalidGamma. All BEFORE device work.
    - fit: affinity = kernel_matrix(X,X,Rbf{gamma:1.0}) (default rbf, D-01/D-04) OR kNN-connectivity if affinity="nearest_neighbors"; (L,dd)=laplacian; eig; recovery with n_components = n_components.unwrap_or(n_clusters) (D-11) and drop_first=FALSE (keep the trivial eigenvector — the `maps` is n × n_components INCLUDING row 0); maps = recovered embedding; KMeans::new(n_clusters, seed).fit(maps); labels_ = kmeans labels_ (i32, kmeans.rs idiom).
    - labels_ test: matches sklearn up to label permutation on the well-separated fixture (D-10) — EXACT (best-match accuracy == 1.0), no band, sign-immune via label_perm.
  </behavior>
  <action>
    Implement spectral_clustering.rs fit (replace the Wave-0 todo!()). REUSE the affinity
    builders and post-eig recovery host math from spectral_embedding.rs (Plan 09-03) — factor
    the recovery into a shared helper if clean, or replicate the pinned order. The ONLY
    differences from SpectralEmbedding:
    - default affinity "rbf" + gamma literal 1.0 (D-01/D-04 — NO None→1/n_features fork for SC).
    - n_components = n_components.unwrap_or(n_clusters) (D-11).
    - drop_first = FALSE → keep ALL n_components rows (including the trivial ≈0 eigenvector) in
      `maps` for an exact match against sklearn's k_means input (RESEARCH §D-11).
    Then maps → KMeans::new(n_clusters, seed).fit(maps) (kmeans++, n_init=1; NOT with_init —
    D-10 rejects init-injection). Store labels_; expose via PredictLabels / a labels_ accessor /
    fit_predict, mirroring KMeans (no new trait).

    Copy the kmeans.rs validate-before-launch block: n_samples>64 → NSamplesExceedsMaxDim
    (D-06) BEFORE any affinity/Laplacian/eig/KMeans call.

    Un-ignore + implement spectral_clustering_test: load the well-separated fixture (D-10),
    fit, compare labels_ to sklearn labels_ up to permutation via best_match_accuracy/label_perm
    (copy from kmeans_test/dbscan_test) — assert EXACT (accuracy == 1.0). EXACT labels is the
    HARD gate (no band). f64 strict via skip_f64_with_log; f32 also exact (labels are integers).
  </action>
  <verify>
    <automated>cargo test --features cpu -p mlrs-algos spectral_clustering_test 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - spectral_clustering_test green: labels_ exact up to permutation (best-match accuracy == 1.0) on cpu f32+f64.
    - drop_first=FALSE (maps includes the trivial eigenvector); n_components defaults to n_clusters; KMeans::new (not with_init).
    - n_samples>64 rejected pre-launch with the spectral-domain typed error.
  </acceptance_criteria>
  <done>SpectralClustering labels_ match sklearn exactly up to permutation on the well-separated fixture (D-10); the D-11 drop_first=FALSE / n_components=n_clusters path is pinned; KMeans::new reused.</done>
</task>

<task type="auto">
  <name>Task 2: PyO3 wrappers for both spectral estimators (any_estimator! ×2) + smoke test</name>
  <read_first>
    - crates/mlrs-py/src/estimators/kernel.rs (any_estimator! invocations, #[pyclass] new/signature, fit body :159-213 with py.detach + guard_f64, dtype-suffixed accessors :266-280)
    - crates/mlrs-py/src/estimators/cluster.rs (PyKMeans labels_ Vec<i32> accessor — mirror for SC labels_)
    - crates/mlrs-py/src/dispatch.rs (:85 any_estimator! macro)
    - The Wave-0 spectral.rs stub (this plan fills its fit/accessor bodies)
  </read_first>
  <action>
    Fill the PyO3 wrapper bodies in estimators/spectral.rs (the Wave-0 stub already declared
    the two any_estimator! invocations + the two #[pyclass] shells + signatures). Copy the
    kernel.rs fit body verbatim shape: capsule_to_array → float_dtype → py.detach(|| { lock
    global_pool; match dt { F32 => validated_f32 + Estimator::<f32>::new(..).fit(..),
    F64 => guard_f64()? (BEFORE upload, D-04) + validated_f64 + ... } }). The Unfit arm stores
    the verbatim sklearn-named hyperparameters; the typed value (gamma Option<F>/literal,
    affinity String, n_components/n_neighbors/n_clusters/seed) is built at fit.

    Accessors (dtype-suffixed, copy kernel.rs:266-280): SpectralEmbedding → embedding_f32 /
    embedding_f64 (Vec<F> via to_host_metered); SpectralClustering → labels_ (Vec<i32>, mirror
    PyKMeans labels accessor in cluster.rs); plus is_fitted / dtype. Accessing before fit →
    not_fitted → PyValueError.

    The two add_class registrations + the estimators/mod.rs `pub mod spectral;` already landed
    in Wave 0 — do NOT re-edit lib.rs / mod.rs here (file-disjoint).

    Implement crates/mlrs-py/tests/spectral_smoke_test.rs (un-ignore the Wave-0 scaffold):
    drive the low-level _mlrs SpectralEmbedding/SpectralClustering classes directly via pyarrow
    capsules (mirror the 08-05 kernel smoke test) — fit + embedding_/labels_ accessors for
    f32 AND f64; f64 cases gated by backend_supports_f64() (skip on rocm, run on cpu). If the
    smoke test requires a built extension (maturin develop), gate it behind the same mechanism
    the kernel smoke test uses; otherwise a cargo-level test asserting the pyclass registration
    + signature is acceptable per the 08-05 precedent. pyo3 stays 0.28 (no bump).
  </action>
  <verify>
    <automated>cargo build -p mlrs-py 2>&1 | tail -3 && grep -q "any_estimator" crates/mlrs-py/src/estimators/spectral.rs && grep -q "guard_f64" crates/mlrs-py/src/estimators/spectral.rs && echo PYO3_OK</automated>
  </verify>
  <acceptance_criteria>
    - mlrs-py builds; estimators/spectral.rs has both any_estimator! wrappers with full fit bodies and dtype-suffixed accessors.
    - guard_f64()? gates the F64 arm BEFORE upload (statically grep-verified).
    - The smoke test drives fit + embedding_/labels_ for f32+f64 (f64 backend-gated) green, or asserts registration per the 08-05 precedent.
    - pyo3 stays 0.28; no new binding infrastructure.
  </acceptance_criteria>
  <done>Both spectral estimators are #[pyclass]-backed with sklearn-named hyperparameters, f32/f64 dispatch, GIL release, and the f64 pre-upload guard; the smoke test exercises fit + accessors.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python → PySpectral* constructor & fit | Untrusted hyperparameters cross the FFI boundary |
| host → SpectralClustering.fit | n_samples / n_clusters / gamma validated before device work |
| F64 dispatch arm → device | f64 on an f64-incapable backend must fail before upload |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-9-VAL | Tampering/DoS | SpectralClustering.fit + PyO3 ingress | mitigate | Reject `n_samples > 64` with `AlgoError::NSamplesExceedsMaxDim` (D-06) and `n_clusters` via `InvalidK` BEFORE any affinity/Laplacian/eig/KMeans device allocation; the PyO3 layer surfaces these as `PyValueError`. Mirrors `kmeans.rs:238` / `topk.rs:74`. |
| T-9-F64 | Information disclosure / silent downcast | F64 PyO3 dispatch arm | mitigate | `guard_f64()?` runs BEFORE any upload on the F64 arm (D-04 precedent); f64 on an incapable backend raises a clear `PyValueError`, never allocates a device buffer. Statically grep-verified. |
| T-9-LBL | Tampering (label flakiness) | KMeans RNG vs sklearn | mitigate | Exact-label gate comes from a WELL-SEPARATED fixture (D-10) so the partition is unique up to permutation — `KMeans::new` (not `with_init`); the SplitMix64-vs-MT19937 RNG gap is immaterial. Compared via `label_perm` (sign-immune). |
| T-9-SC | Tampering | npm/pip/cargo installs | accept | No package installs this phase; all deps first-party (RESEARCH Package Legitimacy Audit: N/A); pyo3 stays 0.28 (no bump). |
</threat_model>

<verification>
- `cargo test --features cpu -p mlrs-algos spectral_clustering_test` green: labels_ exact
  up to permutation (best-match accuracy == 1.0) on cpu f32+f64.
- `cargo build -p mlrs-py` green; estimators/spectral.rs has both any_estimator! wrappers
  with guard_f64 on the F64 arm.
- Smoke test exercises fit + embedding_/labels_ for f32+f64 (f64 backend-gated).
</verification>

<success_criteria>
- SPECTRAL-02: labels_ match sklearn up to label permutation (EXACT, no band) on the
  well-separated fixture (D-10).
- D-11 path pinned: n_components=n_clusters, drop_first=FALSE, assign_labels=kmeans-only.
- Both spectral estimators are #[pyclass]-backed (sklearn-named params, f32/f64 dispatch,
  GIL release, f64 pre-upload guard) — zero new binding infra, pyo3 0.28.
- n_samples > 64 rejected pre-launch with the spectral-domain typed error (D-06).
</success_criteria>

<output>
Create `.planning/phases/09-spectral-family/09-04-SUMMARY.md` when done.
</output>
