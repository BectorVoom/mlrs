# Phase 4: Closed-Form Estimators - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 4-Closed-Form Estimators
**Areas discussed:** PCA solver path, Ridge solver, Fitted-state representation, Estimator API shape

---

## PCA solver path

| Option | Description | Selected |
|--------|-------------|----------|
| SVD of centered X | Match sklearn's actual `svd_solver='full'` arithmetic: center X, run Phase-3 thin SVD, derive explained_variance_ = S²/(n−1), components_ = Vᵀ (svd_flipped). More numerically faithful, robust near rank-deficiency. Uses SVD primitive. | ✓ |
| Eig of covariance | The path Phase 3 D-01/D-06 anticipated: covariance(ddof=1) → symmetric eig. Uses the eig primitive built for exactly this. Slightly less robust for clustered/near-zero eigenvalues. | |
| Defer to researcher | Surface as an open question for the researcher. | |

**User's choice:** SVD of centered X
**Notes:** Consequence — the Phase-3 eig primitive is NOT consumed by PCA in v1; it stays a validated standalone asset. SVD becomes the PCA workhorse. Estimator applies svd_flip via `sign_flip::align_rows`.

---

## Ridge solver

| Option | Description | Selected |
|--------|-------------|----------|
| SVD-based (α-filtered) | Reuse Phase-3 SVD: coef = V·diag(σ/(σ²+α))·Uᵀ·y. No new primitive, shares LinearRegression path, matches sklearn `solver='svd'`. | |
| Cholesky normal-equations | sklearn dense `solver='auto'`→Cholesky default: solve (XᵀX + αI)·coef = Xᵀy. Reuses Gram primitive but needs a NEW Cholesky/triangular-solve primitive. | ✓ |
| Defer to researcher | Let the researcher pick based on cost vs 1e-5 fidelity. | |

**User's choice:** Cholesky normal-equations
**Notes:** Accepted knowing it requires a NEW linear-solve primitive (Cholesky factorization + triangular solve) not present in Phase 2/3. Captured in CONTEXT.md D-02 as an explicit in-phase sub-deliverable + the highest Phase-4 implementation risk, to be validated standalone before Ridge consumes it. LinearRegression stays SVD-based and separate (pinned by LINEAR-01).

---

## Fitted-state representation

| Option | Description | Selected |
|--------|-------------|----------|
| Device-resident (DeviceArray) | Fitted state stays on-device; predict/transform device-side; lazy host materialize on accessor. Memory-efficient, device-resident pipeline, Phase-6 zero-copy setup. | ✓ |
| Host-materialized (Vec) at fit | Read fitted arrays to host Vec at fit; re-upload for predict/transform. Simpler structs but extra copies, breaks device-residency, fights the memory gate. | |
| Defer to researcher | Let the researcher choose. | |

**User's choice:** Device-resident (DeviceArray)
**Notes:** The Phase-2/3 build-failing memory gate extends to fit→predict/transform pipelines.

---

## Estimator API shape

| Option | Description | Selected |
|--------|-------------|----------|
| Shared traits (Fit/Transform/Predict) | Common traits in mlrs-algos; each estimator implements relevant ones. Uniform surface, generic Phase-6 wrapping, mirrors sklearn mixins. | ✓ |
| Standalone structs | Independent structs with inherent methods, no shared trait. Less abstraction; revisit traits in Phase 5. | |
| Defer to researcher | Let the researcher propose the organization. | |

**User's choice:** Shared traits (Fit/Transform/Predict)
**Notes:** `fit` returns self (sklearn convention). Chosen partly because Phase 5 adds 7 more estimators — the shared surface pays off.

---

## Claude's Discretion

- Module/file layout within `mlrs-algos`; exact trait method signatures.
- New Cholesky/solve primitive internal design (blocked vs unblocked, in-place, triangular-solve structure) — subject to the memory gate + tolerance policy.
- LinearRegression small-singular-value (rcond) cutoff constant to match sklearn lstsq.
- Random shapes/seeds for the estimator oracle sweep; which cases get committed sklearn fixtures vs algebraic invariants.
- New estimator/primitive error variant names.

## Deferred Ideas

- `n_components` as float (variance-ratio) / `'mle'` / `None`=all — v1 is int only.
- Additional sklearn constructor knobs (copy_X, tol, whiten, randomized SVD + random_state, positive/normalize).
- Ridge alternative solvers (svd, lsqr, sag, saga, sparse_cg) — v1 ships Cholesky only.
- PCA via eig-of-covariance as a selectable alternate solver — eig primitive exists but unused in v1.
- Reusing the new Cholesky/solve primitive for future GLM/GP/Mahalanobis paths.
