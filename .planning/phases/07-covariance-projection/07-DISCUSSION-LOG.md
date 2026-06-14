# Phase 7: Covariance & Projection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-14
**Phase:** 7-covariance-projection
**Areas discussed:** PartialFit & IncrementalPCA contract, precision_ inversion method, sklearn param fidelity scope, RandomProjection property-gate bar

---

## PartialFit & IncrementalPCA contract

| Option | Description | Selected |
|--------|-------------|----------|
| fit() loops partial_fit | Separate PartialFit<F> trait; fit() resets state then iterates batches through partial_fit (sklearn-faithful: IncrementalPCA.fit over gen_batches). batch_size=None → 5·n_features; n_samples_seen_ tracked. PY-06 exposes both. Exercises the PRIM-07 multi-batch merge in fit itself. | ✓ |
| partial_fit only; fit = 1 batch | fit() treats all of X as a single batch; user drives streaming via explicit partial_fit. Simpler, but fit() won't sub-batch like sklearn (fidelity gap). | |

**User's choice:** fit() loops partial_fit
**Notes:** Chosen for full sklearn fidelity. The [v2-P1] incremental-SVD merge algorithm itself is handed to the research spike before planning; no strong user preference, default leaning is full-Jacobi-per-batch reusing v1 svd (no new kernel).

---

## precision_ inversion method

| Option | Description | Selected |
|--------|-------------|----------|
| eig/SVD pinvh (singular-safe) | Symmetric pseudo-inverse via v1 eig — matches sklearn linalg.pinvh, robust on rank-deficient/near-singular covariance. Reuses existing prim. | ✓ |
| v1 Cholesky inverse (SPD-only) | Reuse Cholesky prim; simpler/faster but raises NotPositiveDefinite on singular covariance (MLE singular when n_samples ≤ n_features). | |

**User's choice:** eig/SVD pinvh (singular-safe)
**Notes:** Matches sklearn covariance semantics; reuses v1 eig prim; safe on the EmpiricalCovariance MLE rank-deficient case.

---

## sklearn param fidelity scope (multi-select)

| Option | Description | Selected |
|--------|-------------|----------|
| whiten (IncrementalPCA) | Whitened transform output (components scaled by 1/sqrt(explained_variance)). | ✓ |
| assume_centered (cov) | EmpiricalCovariance + LedoitWolf skip mean subtraction (location_ = 0). | ✓ |
| store_precision / precision_ | Compute + expose precision_ accessor (default True). Required by COV-01 regardless. | ✓ |
| batch_size (IncrementalPCA) | Explicit batch_size for partial_fit batching (None → 5·n_features). | ✓ |

**User's choice:** All four — full sklearn fidelity.
**Notes:** Aligns with the project's sklearn-compatible-API core value.

---

## RandomProjection property-gate bar

| Option | Description | Selected |
|--------|-------------|----------|
| Flake-resistant bands | Loose, well-margined thresholds; moments to ~2-3 sig figs over many trials. Never flakes. | |
| Strict bands | Tight thresholds close to theoretical limits — higher sensitivity, risk of seed-dependent flakiness across cpu/rocm. | ✓ |

**User's choice:** Strict bands
**Notes:** User explicitly wants the tight bar. Mitigation recorded in CONTEXT D-11: deterministic SplitMix64 seeding + averaging distortion/moment stats over many trials keeps strict thresholds reproducible across cpu/rocm rather than seed-fragile. Researcher/planner pins exact numbers + trial count. Gate stays structural (JL distortion, distribution stats, seed-reproducibility, transform self-consistency), NOT a 1e-5 value oracle; johnson_lindenstrauss_min_dim is value-matched.

---

## Claude's Discretion

- Exact f32-on-rocm tolerance bands for LedoitWolf / IncrementalPCA (follow v1 per-family documented-band precedent).
- Whether EmpiricalCovariance/LedoitWolf expose error_norm/mahalanobis helpers — only if cheap and within COV-01/02 surface.

## Deferred Ideas

- Device RNG kernel — only if the [v2-P1] spike shows host-generate-then-upload is a bottleneck (default: not needed).
- Dedicated incremental rank-update SVD kernel — only if full-Jacobi-per-batch is unstable on f32/rocm per the spike.
- Sparse device kernels for SparseRandomProjection — out of v2 scope; densify at ingress, components_ stored dense.
