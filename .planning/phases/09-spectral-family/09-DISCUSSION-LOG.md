# Phase 9: Spectral Family - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 9-spectral-family
**Areas discussed:** Affinity scope & gamma, Eigensolver & size cap, Embedding recovery, Exact-label via KMeans

---

## Affinity scope & gamma

### Affinity surface
| Option | Description | Selected |
|--------|-------------|----------|
| RBF-only | Only affinity='rbf' (kernel_matrix(Rbf)); oracle constructs sklearn with affinity='rbf' explicitly | |
| RBF + precomputed | Also accept a precomputed n×n affinity matrix | |
| RBF + nearest_neighbors | Also build a symmetrized kNN connectivity graph (sklearn SpectralEmbedding default) | ✓ |

### Per-estimator defaults
| Option | Description | Selected |
|--------|-------------|----------|
| Mirror sklearn per-estimator | SE default = nearest_neighbors, SC default = rbf; oracle uses each default constructor | ✓ |
| One shared default (rbf) | Both default to rbf; SE diverges from sklearn default | |

### kNN graph construction
| Option | Description | Selected |
|--------|-------------|----------|
| sklearn-exact connectivity | Binary connectivity (0/1), symmetrized 0.5·(A+Aᵀ), n_neighbors=10 | ✓ |
| Reuse v1 kNN distances | Distance-weighted graph; diverges from sklearn connectivity | |

### Gamma
| Option | Description | Selected |
|--------|-------------|----------|
| Mirror each exactly | SE: None→1/n_features; SC: default 1.0 literal; pin from sklearn source | ✓ |
| Defer exact value to planner | Lock the principle, leave the formula for planner | |

**User's choice:** RBF + nearest_neighbors; mirror sklearn per-estimator defaults; sklearn-exact binary connectivity graph; gamma mirrors each estimator exactly.
**Notes:** Chose the broadest affinity surface to match sklearn SpectralEmbedding's actual default behavior. This pulls in a small new kNN-connectivity-graph builder (binary weights, symmetrized) beyond the bare kernel_matrix(Rbf) seam — accepted deliberately for parity. gamma is a double parity fork (None→1/n_features for SE vs literal 1.0 for SC).

---

## Eigensolver & size cap ([v2-P3] research flag)

| Option | Description | Selected |
|--------|-------------|----------|
| Full-spectrum + n≤64 cap | v1 eig full spectrum, slice smallest; hard-reject n_samples>64 as typed error; no Lanczos | ✓ |
| Full-spectrum, soft cap | Same, but 64 is a soft/warn limit | |
| Investigate Lanczos | Research spike on shift-invert/Lanczos to lift the cap | |

**User's choice:** Full-spectrum-then-slice with a hard n_samples ≤ 64 cap.
**Notes:** Codebase scout found v1 eig caps n ≤ MAX_DIM = 64 and the Laplacian is n_samples×n_samples, so n_samples ≤ 64 is forced. Dense Jacobi is exact and already validated at that size — Lanczos/shift-invert is pointless. [v2-P3] is effectively pre-answered; the research spike confirms-and-documents the cap rather than investigating a new solver.

---

## Embedding recovery

### D^-1/2 recovery scaling
| Option | Description | Selected |
|--------|-------------|----------|
| Reproduce dd-recovery exactly | Divide eigenvectors by sqrt(degree) before sign flip; required for value match | ✓ |
| Raw eigenvectors only | No /dd recovery; only subspace-matches, not the value gate | |

### n_components & trivial drop
| Option | Description | Selected |
|--------|-------------|----------|
| Mirror sklearn exactly | Default n_components=2; compute n_components+1, drop trivial ≈0 | ✓ |
| Expose n_components, fixed drop | Same drop, but require explicit n_components (no default) | |

**User's choice:** Reproduce the D^-1/2 (/sqrt(degree)) recovery exactly; mirror sklearn n_components default (2) and drop_first.
**Notes:** The /dd recovery is the make-or-break step for embedding_ value-matching sklearn; easy to omit. Exact operation order (slice ascending → /dd → deterministic sign flip → drop trivial) to be pinned from sklearn _spectral_embedding.py during planning.

---

## Exact-label via KMeans

| Option | Description | Selected |
|--------|-------------|----------|
| Well-separated fixture | Oracle data well-separated → unique partition → init-invariant labels through v1 KMeans | ✓ |
| Inject KMeans init | Phase-5 init-injection pattern; but sklearn SpectralClustering hides its inner KMeans init | |
| Both: separated + seeded | Well-separated fixture AND fixed random_state/n_init | |

### SpectralClustering surface
| Option | Description | Selected |
|--------|-------------|----------|
| n_components=n_clusters, kmeans-only | Embedding dim defaults to n_clusters; assign_labels='kmeans' only | ✓ |
| Defer to planner | Lock kmeans-only, leave n_components default for planner | |

**User's choice:** Well-separated oracle fixture (init-invariant labels); n_components=n_clusters with kmeans-only label assignment.
**Notes:** The exact-labels hard gate is met by FIXTURE DESIGN (unique partition), not by matching sklearn's RNG/init — the v2 spectral analogue of the Phase-5 tuned DBSCAN fixture. Init-injection rejected because sklearn SpectralClustering doesn't expose its inner KMeans init, which would force comparing against a hand-built pipeline instead of the actual estimator. discretize/cluster_qr deferred.

---

## Claude's Discretion

- Exact f32-on-rocm tolerance band for SpectralEmbedding embedding_ (band + sign, or subspace test for degenerate spectra); f64 stays strict, gated by skip_f64_with_log. Exact labels is the hard gate for SpectralClustering (no band).
- The precise laplacian.rs degree-reduction kernel shape (single-owner row reduction; GATHER not scatter; typed-zero guard, no F::INFINITY, no atomics, SharedMemory-free).
- Whether the estimators need a new trait or compose on existing Fit + Transform / PredictLabels (likely no new trait).
- Exact sklearn-source-pinned formulas: gamma None→value per estimator, the recovery/drop_first slice order, the n_components default.

## Deferred Ideas

- affinity='precomputed' / 'precomputed_nearest_neighbors' — not selected; cheap future add.
- assign_labels='discretize' / 'cluster_qr' — out of scope (kmeans-only).
- Lanczos / shift-invert smallest-eigenpair solver — would lift the n≤64 cap; revisit only if a future milestone raises the problem-size ceiling.
