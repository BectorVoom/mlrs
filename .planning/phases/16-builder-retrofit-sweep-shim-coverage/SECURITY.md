---
phase: 16
slug: builder-retrofit-sweep-shim-coverage
status: secured
threats_open: 0
asvs_level: 1
created: 2026-06-26
---

# SECURITY.md — Phase 16: Builder Retrofit Sweep + Shim Coverage

**Audit date:** 2026-06-26
**ASVS Level:** 1
**block_on:** high
**Register origin:** `register_authored_at_plan_time: true` — all 13 PLAN.md files carried a parseable `<threat_model>` block. Verification = confirm declared mitigations exist (no new-threat scan).
**Result:** SECURED — all 10 declared threats verified CLOSED. The single open finding from the 2026-06-26 audit (poison-recovery `lock_pool()` half of T-16-GUARDF64 absent in covariance wrappers) was remediated the same day via quick task `260626-ktm` and re-verified. See Audit Trail.

This is a numeric-ML library with no network / auth / file-upload / session surface. The threat model is correctly scoped to input-geometry validation (V5), backend-capability guards (f64), typestate correctness, and mutex poison-recovery.

---

## Threat Verification — mitigate disposition

| Threat ID | Category | Status | Evidence |
|-----------|----------|--------|----------|
| T-16-V5 | Tampering/DoS | CLOSED | `validate_geometry` at TOP of fit before any device launch: ridge.rs:fit (validate at line 20, before first `to_host`); 28 estimator src files call `validate_geometry`/equivalent. Data-INDEPENDENT checks relocated to `build()→BuildError`: ridge `InvalidAlpha` (ridge.rs:189), neighbors `InvalidNNeighbors` (nearest.rs:192). Data-DEPENDENT checks retained in fit/accessor: KMeans `InvalidK` + injected-init dims (`init.len() != k*n_features`) kmeans.rs fit lines 15-33; k≤n_train in `neighbor_indices` (nearest.rs:25 `k < 1 || k > n_train` → InvalidK); IncrementalPCA `validate_batch` (incremental_pca.rs:247) called by both partial_fit arms via `merge_batch` (294). |
| T-16-GUARDF64 | DoS | CLOSED | `guard_f64()` half: present before every F64 upload across all 11 PyO3 estimator modules incl. PyUMAP.transform_f64 (manifold.rs:328), fit_transform_f64 (394), PyHDBSCAN.fit (474). `lock_pool()` half: REMEDIATED 2026-06-26 (quick task 260626-ktm, commits d3bcd72/e391eed) — all 12 covariance.rs sites converted from `global_pool().lock().expect("pool mutex")` to poison-recovering `crate::lock_pool()`; re-verified: `grep -c lock_pool() covariance.rs` = 12, zero live-code legacy hits across `estimators/` (only the naive_bayes.rs:19 doc counter-example remains), `cargo build -p mlrs-py --features cpu` clean. covariance was the sole remaining legacy holder; ZERO estimator modules now use the panicking lock form. |
| T-16-ARM | Tampering | CLOSED | `any_estimator_typestate!` macro used in every estimator module (linear ×9, decomposition ×3, cluster ×3, covariance ×2, neighbors ×3, projection ×2, kernel ×2, manifold ×2, spectral ×2, naive_bayes ×5); fitted arms spell `<F, Fitted>`. IncrementalPCA `partial_fit` on `Fitted` returns `type Fitted = Self` (incremental_pca.rs:391), Unfit arm → `IncrementalPCA<F, Fitted>` (371). |
| T-16-POISON | DoS | CLOSED | New UMAP/HDBSCAN methods (the declared scope of T-16-POISON, plan 16-10) use `lock_pool()` exclusively: manifold.rs (8 sites incl. fit 243, transform 307/325, fit_transform 357/393), cluster.rs PyHDBSCAN (387, 456, 506, 526, 535, 546, 555). No `.lock().expect()` in either. |
| T-16-NOTFIT | Tampering | CLOSED | Unfit arm returns `not_fitted(...)`: PyUMAP transform_f32/f64 (manifold.rs:315/334), embedding (147/155); PyHDBSCAN labels_inner (391), fit_predict (510), probabilities_f32/f64 (529/538), outlier_scores_f32/f64 (549/558). |
| T-16-00-PURITY | Tampering | CLOSED | `test_init_purity_ast` (test_params.py:377) parses each `cls.__init__` via `import ast` (line 18); asserts every statement is a bare `self.<name> = <name>` Assign, rejecting any `ast.Call`/`BinOp`/`Compare`/`if`/`for`/`raise` — impure body is a hard FAIL. Parametrized over `ALL_SHIMS` (32, derived from EXPECTED_PARAMS). |
| T-16-DEFCTOR | Tampering | CLOSED | New shims default every arg: UMAP (manifold.py — n_neighbors=15…b=None all defaulted) and HDBSCAN (cluster.py:13-20 — min_cluster_size=5, min_samples=None, … all defaulted). `check_parameters_default_constructible` ∈ `_FIT_FREE_CHECKS` (test_estimator_checks.py:227), never-xfailed (`test_fit_free_checks_never_xfailed`), recorded green (82 passed, per 16-12-SUMMARY:104 / VERIFICATION.md). Required-arg PCA/IncrementalPCA constructed with explicit params (test_estimator_checks.py:51) per sklearn instance API — not a defect. |
| T-16-PITFALL3 | Tampering/DoS | CLOSED | Empty-grep convergence gate holds: `grep -rn "use crate::traits\|crate::traits::"` across `crates/mlrs-algos/src/` + `crates/mlrs-py/src/` returns ZERO live references. `crates/mlrs-algos/src/traits.rs` hard-deleted (file absent). lib.rs:64 has only `pub mod typestate;`, no `pub mod traits`. (One comment-only string in random_projection_test.rs:31 — excluded, doc-comment, non-blocking per VERIFICATION.md.) |

## Threat Verification — accept disposition (accepted-risks log)

| Threat ID | Category | Status | Evidence / Acceptance basis |
|-----------|----------|--------|------------------------------|
| T-16-SEAL | Tampering | CLOSED (accepted, verified in code) | `State: sealed::Sealed` closed set: `mod sealed` private (typestate.rs:98-102); `State` supertrait-bound on it (112); only `Unfit` (118) and `Fitted` (125) implement `Sealed`+`State` (127-131). No third lifecycle state. A downstream crate cannot name `sealed::Sealed`, so the set is closed at the crate boundary. **Accepted:** no action required; seal not broken. |
| T-16-FFI-DEFER | — | CLOSED (accepted, documented) | Live `check_estimator`/`estimator_checks` over the compiled `_mlrs` extension deferred to UAT — no maturin+pyarrow host (confirmed: `import pyarrow` → ModuleNotFoundError in this env). Static + trybuild gates are the maximum verifiable. Documented in: 16-VALIDATION.md:28 & :74 (UAT row), 16-12-SUMMARY.md:128 & :138-140. **Accepted:** by-design deferral, honestly documented. |

---

## Open Threats (BLOCKER candidates)

None. The single finding from the 2026-06-26 audit (T-16-GUARDF64 `lock_pool()` half) was remediated the same day — see Resolved Findings and Audit Trail below.

---

## Resolved Findings

### T-16-GUARDF64 — `lock_pool()` half (covariance wrappers) — RESOLVED 2026-06-26

**Mitigation declared (16-06-PLAN `<threat_model>`):**
> PyKMeans/PyEmpiricalCovariance/PyLedoitWolf F64 arms — `guard_f64()` + `lock_pool()` preserved in migrated PyO3 fits

**Original finding:** `crates/mlrs-py/src/estimators/covariance.rs` used the FORBIDDEN panicking lock form `crate::global_pool().lock().expect("pool mutex")` at 12 code sites (PyEmpiricalCovariance::fit covariance.rs:85, PyLedoitWolf::fit covariance.rs:222, plus 10 fitted-attribute accessors) with ZERO `lock_pool()` calls — the sole estimator module still on the legacy form. Per lib.rs:96-117, `lock_pool()` is the sanctioned poison-recovering path (WR-02/WR-04); a device fault inside a covariance `fit`'s `py.detach` closure would poison the global pool mutex, after which every `.lock().expect()` re-panics — turning one recoverable device error into a process-wide brick for covariance ops until interpreter restart. Medium-severity DoS/poison-recovery (the `guard_f64()` half was already present).

**Resolution (chosen option 1 — close the threat):** Quick task `260626-ktm`. All 12 sites converted to `crate::lock_pool()`; the stale lib.rs:113-117 doc comment refreshed to record zero remaining legacy holders. Behavior-preserving lock-path swap — no compute/binding/control-flow change; existing oracle suites cover correctness.

**Re-verification (2026-06-26):**
- `grep -rnE "global_pool\(\)\.lock\(\)\.expect" crates/mlrs-py/src/estimators/` → only `naive_bayes.rs:19` (intentional doc counter-example); zero live-code hits.
- `grep -c "lock_pool()" covariance.rs` → 12.
- `cargo build -p mlrs-py --features cpu` → clean (2 pre-existing macro warnings, no errors).
- Commits: `d3bcd72` (code swap), `e391eed` (doc refresh).

---

## Audit Trail

### Security Audit 2026-06-26 (initial)
| Metric | Count |
|--------|-------|
| Threats found | 10 (8 mitigate + 2 accept) |
| Closed | 9 |
| Open | 1 (T-16-GUARDF64 lock_pool half, covariance.rs) |

### Re-verification 2026-06-26 (post-remediation, quick task 260626-ktm)
| Metric | Count |
|--------|-------|
| Threats found | 10 |
| Closed | 10 |
| Open | 0 |

---

## Unregistered Flags

None. All 11 `## Threat Flags` sections across 16-01..16-12 SUMMARY.md declare "None — trait-surface retrofit / deletion with byte-identical compute; no new network/auth/file/schema surface" (16-05 additionally notes `BuildError::InvalidEps` is a construction-time guard = a mitigation, not new surface). No new attack surface appeared during implementation without a threat mapping.

---

## Summary

| Disposition | Total | Closed | Open |
|-------------|-------|--------|------|
| mitigate | 8 | 8 | 0 |
| accept | 2 | 2 | 0 |

**threats_open: 0** — all 10 declared threats CLOSED. The lone audit finding (T-16-GUARDF64 `lock_pool()` half in covariance.rs) was remediated and re-verified same-day via quick task 260626-ktm.
