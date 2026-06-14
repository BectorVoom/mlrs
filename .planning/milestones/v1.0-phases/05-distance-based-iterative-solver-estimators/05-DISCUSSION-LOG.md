# Phase 5: Distance-Based & Iterative-Solver Estimators - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 5-Distance-Based & Iterative-Solver Estimators
**Areas discussed:** New-primitive boundary, Output API surface, Stochastic oracle, Iterative solver model

---

## New-primitive boundary

### Primitive-promotion bar
| Option | Description | Selected |
|--------|-------------|----------|
| Shared OR invariant-testable | Promote compute ≥2 estimators share OR with a clean algebraic invariant; estimator-specific orchestration stays in mlrs-algos | |
| Aggressive — every kernel | Even single-consumer device compute becomes its own gated mlrs-backend primitive | ✓ |

**User's choice:** Aggressive — every device-compute kernel becomes its own validated standalone primitive.

### KNN k-nearest selection
| Option | Description | Selected |
|--------|-------------|----------|
| New top-k selection primitive | Partial-select-k kernel, k indices + distances, lowest-index tie-break | ✓ |
| Reuse argmin_rows k times | Iterative masked argmin, no new kernel | |

**User's choice:** New top-k selection primitive.

### CD vs L-BFGS abstraction
| Option | Description | Selected |
|--------|-------------|----------|
| Two separate solvers | Shared CD for Lasso/EN; independent L-BFGS for LogReg | ✓ |
| One generic optimizer | Shared iterative-optimization loop both plug into | |

**User's choice:** Two separate solvers.

### DBSCAN cluster expansion
| Option | Description | Selected |
|--------|-------------|----------|
| Host-side BFS/union-find | Device does distance+eps-mask+core-point; host walks the graph | ✓ |
| Device label propagation | Iterative connected-components on device | |

**User's choice:** Host-side BFS/union-find.

---

## Output API surface

### Integer label representation
| Option | Description | Selected |
|--------|-------------|----------|
| New label-returning trait(s) | Integer DeviceArray trait; keep Predict<F> for regressors | ✓ |
| Encode labels as F | Reuse Predict<F>, cast labels to/from float | |

**User's choice:** New label-returning trait(s).

### Label element type
| Option | Description | Selected |
|--------|-------------|----------|
| i32 everywhere | Signed for DBSCAN -1; one label type across the surface | ✓ |
| Split usize / i32 | usize for KMeans/KNN, i32 only for DBSCAN | |

**User's choice:** i32 everywhere.

### Multi-output / estimator-specific methods
| Option | Description | Selected |
|--------|-------------|----------|
| Inherent methods, not traits | kneighbors → tuple, predict_proba an inherent method | |
| Formalize as traits | KNeighbors trait (dist+idx) + PredictProba trait | ✓ |

**User's choice:** Formalize as traits.

### fit_predict / API shape
| Option | Description | Selected |
|--------|-------------|----------|
| Match sklearn shape | Fit + fit_predict; KMeans Predict, DBSCAN none; device-resident attrs | ✓ |
| Force uniform Predict on both | Give DBSCAN a predict | |

**User's choice:** Match sklearn shape.

---

## Stochastic oracle

### KMeans oracle determinism
| Option | Description | Selected |
|--------|-------------|----------|
| Inject fixed init centers | Both run Lloyd from identical init; compare up to permutation within 1e-5 | ✓ |
| Quality-bound comparison | inertia ≤ sklearn·(1+ε) + labels up to permutation | |

**User's choice:** Inject fixed init centers.

### k-means++ implementation
| Option | Description | Selected |
|--------|-------------|----------|
| Implement it, test Lloyd via injected init | Build D²-sampling primitive; deterministic oracle from injected init | ✓ |
| Ship plain random init | Skip k-means++ (violates CLUSTER-01) | |

**User's choice:** Implement it, test Lloyd via injected init.

### n_init policy
| Option | Description | Selected |
|--------|-------------|----------|
| n_init=1 (sklearn 'auto') | Current sklearn k-means++ default; deterministic with injected init | ✓ |
| n_init=10, keep best | Legacy default; more robust, more compute | |

**User's choice:** n_init=1.

### k-means++ RNG location
| Option | Description | Selected |
|--------|-------------|----------|
| Host-side seeded RNG | Host picks next center from device D² weights; backend-independent | ✓ |
| Device-side RNG | On-device, backend-divergent streams | |

**User's choice:** Host-side seeded RNG.

---

## Iterative solver model

### Iteration loop structure
| Option | Description | Selected |
|--------|-------------|----------|
| Host-driven loop over device kernels | Host loop, device kernels, one scalar readback/iter (gate exception) | ✓ |
| In-kernel iteration (Jacobi precedent) | Full solver loop inside one #[cube] kernel | |

**User's choice:** Host-driven loop over device kernels.

### Convergence criteria
| Option | Description | Selected |
|--------|-------------|----------|
| Match sklearn's exact criteria | Duality gap (CD); grad-norm/pgtol (L-BFGS); exact tol/max_iter | ✓ |
| Simple change-based tol | Stop on max coef change < tol | |

**User's choice:** Match sklearn's exact criteria.

### LogReg multiclass formulation
| Option | Description | Selected |
|--------|-------------|----------|
| Multinomial softmax | Stable softmax + cross-entropy, sklearn lbfgs default; binary = 2-class | ✓ |
| One-vs-rest | k independent binary models | |

**User's choice:** Multinomial softmax.

### Penalty / objective scaling
| Option | Description | Selected |
|--------|-------------|----------|
| Match sklearn objectives exactly | (1/2n)‖y−Xw‖²+α·penalty; LogReg L2 with C; intercepts unpenalized | ✓ |
| You decide / pin in research | Defer entirely without locking the directive | |

**User's choice:** Match sklearn objectives exactly.

---

## Claude's Discretion

- Exact set/granularity of the new primitives (D-01) — Lloyd-update split, CD-step shape, etc.
- Module/file layout in mlrs-algos and exact trait names/method signatures (D-05/D-07).
- L-BFGS history size `m` and line-search details (D-03/D-11).
- Random shapes/seeds for oracle sweeps; fixture-vs-invariant case selection.
- New error-variant naming (extend the thiserror enums).
- KNN distance-metric scope and `weights='uniform'` default — confirmed during planning.

## Deferred Ideas

- KMeans `n_init=10` / multi-restart.
- KNN `weights='distance'`, non-Euclidean metrics, spatial indices (kd-tree/ball-tree).
- DBSCAN device-side label propagation.
- Additional sklearn constructor knobs (neighbors algorithm/leaf_size/p; CD selection/positive/warm_start; LogReg penalty/solver/class_weight/multi_class=ovr; DBSCAN metrics).
- Generalizing the new optimizer primitives (top-k for ANN, L-BFGS for other GLMs, CD for other sparse models).
