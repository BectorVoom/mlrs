---
phase: 5
slug: distance-based-iterative-solver-estimators
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-12
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `05-RESEARCH.md` §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (integration tests in `crates/*/tests/`, AGENTS.md §2 — no in-source `#[cfg(test)]`) |
| **Config file** | none (cargo default); fixtures in `tests/fixtures/*.npz` via `mlrs_core::oracle::load_npz` |
| **Quick run command** | `cargo test --features cpu -p mlrs-backend <prim>_test` (targeted prim) |
| **Full suite command** | `cargo test --features cpu` then `cargo test --features rocm` (cpu f64 + rocm f32 gate; f64-on-rocm skips-with-log) |
| **Estimated runtime** | mlrs-backend cpu suite ~6 min (existing reduce_test 248s / svd_test 99s dominate); Phase-5 prim oracles add to this — run TARGETED per-prim gates, background the full sweep |

---

## Sampling Rate

- **After every task commit:** Run the touched prim's targeted oracle — `cargo test --features cpu -p mlrs-backend <prim>_test` (or `-p mlrs-algos <estimator>_test`).
- **After every plan wave:** Run that estimator's `-p mlrs-algos` oracle test + the prim oracle tests it depends on, on cpu(f64); spot-check rocm(f32).
- **Before `/gsd-verify-work`:** Full `cargo test --features cpu` + `cargo test --features rocm` green (f64-on-rocm skips logged), including the extended `memory_gate_test`.
- **Max feedback latency:** targeted prim oracle < ~30s; full sweep backgrounded.

---

## Per-Task Verification Map

| Req / Artifact | Behavior | Test Type | Automated Command | File Exists | Status |
|----------------|----------|-----------|-------------------|-------------|--------|
| (NEW prim) top-k select | k indices+distances, lowest-index tie | unit/oracle | `cargo test --features cpu -p mlrs-backend topk_test` | ❌ W0 | ⬜ pending |
| (NEW prim) k-means++ D² | valid + seed-reproducible D² sampling | invariant | `cargo test --features cpu -p mlrs-backend kmeanspp_test` | ❌ W0 | ⬜ pending |
| (NEW prim) Lloyd update+inertia | centroid sum-by-label, inertia | oracle | `cargo test --features cpu -p mlrs-backend lloyd_test` | ❌ W0 | ⬜ pending |
| (NEW prim) DBSCAN eps-core-mask | core bit, eps-neighborhood incl self | oracle | `cargo test --features cpu -p mlrs-backend dbscan_mask_test` | ❌ W0 | ⬜ pending |
| (NEW prim) CD coordinate update | soft-threshold + residual update | oracle | `cargo test --features cpu -p mlrs-backend cd_test` | ❌ W0 | ⬜ pending |
| (NEW prim) L-BFGS dir + softmax loss/grad | convex-quadratic min; softmax grad | oracle/invariant | `cargo test --features cpu -p mlrs-backend lbfgs_test` | ❌ W0 | ⬜ pending |
| CLUSTER-01 | KMeans centers/labels/inertia up to perm | oracle | `cargo test --features cpu -p mlrs-algos kmeans_test` | ❌ W0 | ⬜ pending |
| CLUSTER-02 | DBSCAN labels(-1)+core_sample_indices_ | oracle | `cargo test --features cpu -p mlrs-algos dbscan_test` | ❌ W0 | ⬜ pending |
| NEIGH-01 | NearestNeighbors k dist+idx 1e-5 | oracle | `cargo test --features cpu -p mlrs-algos nearest_neighbors_test` | ❌ W0 | ⬜ pending |
| NEIGH-02 | KNeighborsClassifier predict/proba | oracle | `cargo test --features cpu -p mlrs-algos knn_classifier_test` | ❌ W0 | ⬜ pending |
| NEIGH-03 | KNeighborsRegressor predict | oracle | `cargo test --features cpu -p mlrs-algos knn_regressor_test` | ❌ W0 | ⬜ pending |
| LINEAR-03 | Lasso sparse coef_ | oracle | `cargo test --features cpu -p mlrs-algos lasso_test` | ❌ W0 | ⬜ pending |
| LINEAR-04 | ElasticNet coef_ | oracle | `cargo test --features cpu -p mlrs-algos elastic_net_test` | ❌ W0 | ⬜ pending |
| LINEAR-05 | LogReg predict/proba (binary+multiclass) | oracle | `cargo test --features cpu -p mlrs-algos logistic_test` | ❌ W0 | ⬜ pending |
| Memory gate (D-10) | iterative-solver bounded alloc + 1-scalar/iter readback | hard gate | `cargo test --features cpu -p mlrs-backend memory_gate_test` | extends existing | ⬜ pending |
| Memory gate (D-04) | DBSCAN n² bound + core-mask readback | hard gate | `cargo test --features cpu -p mlrs-backend memory_gate_test` | extends existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> **LogReg gauge-freedom escape hatch (research MEDIUM):** the symmetric over-parameterized multinomial softmax has gauge freedom in `coef_`, so the LINEAR-05 oracle gates on `predict_proba`/`predict` (gauge-invariant) as the primary assertion with `coef_` as a looser secondary check. Validate the L-BFGS primitive standalone on a convex quadratic with a known minimizer BEFORE LogReg consumes it.
> **KNN exact-tie (research MEDIUM):** sklearn's post-argpartition argsort is not tie-stable; oracle fixtures use distinct distances to sidestep tie ambiguity, with the lowest-index tie-break asserted on a separate constructed-tie invariant case.

---

## Wave 0 Requirements

- [ ] `tests/topk_test.rs`, `kmeanspp_test.rs`, `lloyd_test.rs`, `dbscan_mask_test.rs`, `cd_test.rs`, `lbfgs_test.rs` (mlrs-backend prim oracles)
- [ ] `tests/{kmeans,dbscan,nearest_neighbors,knn_classifier,knn_regressor,lasso,elastic_net,logistic}_test.rs` (mlrs-algos estimator oracles)
- [ ] `scripts/gen_oracle.py` extensions: `gen_kmeans` (INJECTED init centers — D-09), `gen_dbscan`, `gen_knn`, `gen_lasso`, `gen_elastic_net`, `gen_logistic` (binary + multiclass) — committed `.npz` blobs, regen in `/tmp/oracle-venv`
- [ ] `memory_gate_test.rs` extensions: iterative-solver bounded-allocation gate (CD + L-BFGS) + DBSCAN n²-bound gate (D-10/D-04 exceptions)
- [ ] `traits.rs` extensions: label-returning + KNeighbors + PredictProba traits (D-05/D-07)
- [ ] `error.rs` extensions: new hyperparameter-guard variants (InvalidK/InvalidEps/InvalidMinSamples/InvalidL1Ratio/InvalidC/NotConverged)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm(f32) gate for new prims/estimators | all | requires gfx1100/ROCm 7.1.1 hardware (not in default CI sandbox) | Run `cargo test --features rocm` on the ROCm host; confirm f32 oracles pass and f64 cases emit `skip_f64_with_log` |

*All numerical behaviors have automated cpu(f64) oracle verification; rocm(f32) is the hardware gate run opportunistically.*

---

## Validation Sign-Off

- [ ] All tasks have an `<automated>` verify command or a Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (prim + estimator oracles, gen_oracle fixtures)
- [ ] No watch-mode flags
- [ ] Feedback latency: targeted prim oracle < ~30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-12
