# Phase 3: SVD / Eigendecomposition Primitive (Hard Gate) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 3-SVD / Eigendecomposition Primitive (Hard Gate)
**Areas discussed:** Primitive surface & outputs, Input-shape coverage, Gate backend (rocm), Oracle & tolerance policy, Memory gate scope

---

## Primitive surface & outputs

### SVD ↔ eig relationship
| Option | Description | Selected |
|--------|-------------|----------|
| One Jacobi core, eig derived | One one-sided Jacobi SVD; derive symmetric eig (eigenvectors=V, eigenvalues=S², U=V for PSD) | |
| Two distinct routines | General one-sided SVD AND a separate classic two-sided Jacobi symmetric-eig; eig returns true signed eigenvalues, reusable | ✓ |

### U/V extent
| Option | Description | Selected |
|--------|-------------|----------|
| Thin / economy (k=min(m,n)) | U(m×k), S(k), Vᵀ(k×n) — numpy full_matrices=False; covers all consumers | ✓ |
| Full U(m×m), V(n×n) | Full square factors; extra columns unused, more memory | |

### Sign convention
| Option | Description | Selected |
|--------|-------------|----------|
| Raw output; align at comparison | Kernel pure; mlrs-core sign_flip aligns only at oracle time | ✓ |
| Bake svd_flip into the primitive | Device-side flip kernel; bakes estimator convention into the primitive | |

### Ordering
| Option | Description | Selected |
|--------|-------------|----------|
| Descending, LAPACK/numpy | Singular values & eigenpairs descending; eig sorts (np.eigh is ascending) | ✓ |
| Unsorted, caller sorts | Pushes sklearn-convention sorting into every consumer | |

**User's choice:** Two distinct routines; thin/economy; raw output aligned at comparison; descending order.
**Notes:** Covariance is PSD so eig≡SVD, but user wants the dedicated eig path for true signed eigenvalues + reuse.

---

## Input-shape coverage

### SVD shapes
| Option | Description | Selected |
|--------|-------------|----------|
| Tall + wide (both) | Handle m≥n and m<n (run on Aᵀ, swap U↔V); shape-agnostic | ✓ |
| Tall-only (m≥n) | Only n_samples≥n_features; wide errors/deferred | |

### Eig input
| Option | Description | Selected |
|--------|-------------|----------|
| Square symmetric, caller guarantees | Validate squareness, trust symmetry; only feeder is covariance primitive | ✓ |
| Symmetrize defensively | (A+Aᵀ)/2 before decomposing; unnecessary given feeder | |

### Validation sweep coverage
| Option | Description | Selected |
|--------|-------------|----------|
| Random sweep + degenerate cases | Random tall/wide/square + rank-deficient/repeated/near-identity/clustered-eigenvalue | ✓ |
| Random well-conditioned only | Distinct singular values only; leaves degenerate cases unvalidated | |

### Largest size
| Option | Description | Selected |
|--------|-------------|----------|
| Small + one moderate case | Mostly small + one ~256×64 to exercise convergence loop on GPU | ✓ |
| Small only | ≤8×8; won't surface convergence/stability issues | |

**User's choice:** Tall+wide; square-symmetric trusted; random + degenerate; small + one moderate.

---

## Gate backend (rocm)

> User injected this mid-discussion: "Please test rocm. no wgpu."

| Option | Description | Selected |
|--------|-------------|----------|
| This phase only (cpu + rocm) | rocm replaces wgpu for Phase 3 only; project gate unchanged | |
| Project-wide (cpu + rocm henceforth) | rocm becomes the GPU gate from Phase 3 on; wgpu opportunistic; needs PROJECT/ROADMAP update | ✓ |
| Add rocm, keep wgpu (cpu+wgpu+rocm) | Validate on all three | |

**User's choice:** Project-wide — cpu + rocm supersedes wgpu from Phase 3 onward.
**Notes:** Feasibility verified live — ROCm 7.1.1, hipcc, AMD gfx1100 (RDNA3), /dev/kfd + /dev/dri present; gfx1100 supports f64 natively so f64 runs (not skips) on rocm. rocm has been compile-only/never-executed through Phase 2 → ROCm bring-up is the first task. ROADMAP/PROJECT still document cpu+wgpu and need reconciling.

---

## Oracle & tolerance policy

### Reference source
| Option | Description | Selected |
|--------|-------------|----------|
| numpy fixtures + reference-free invariants | np.linalg.svd/eigh fixtures + reconstruction/orthonormality/eig-residual invariants | ✓ |
| Host Rust SVD reference (like P2 D-12) | Naive host Jacobi as primary; itself iterative/error-prone, needs its own validation | |

### f32 tolerance
| Option | Description | Selected |
|--------|-------------|----------|
| Hold 1e-5; family table only if forced | Keep global 1e-5; document looser bound only if a real case forces it, record which | ✓ |
| Pre-document looser SVD/eig f32 bound | Proactively loosen; risks masking real errors | |

**User's choice:** numpy fixtures + reference-free invariants; hold 1e-5, no pre-loosening.

---

## Memory gate scope

### Gate extension
| Option | Description | Selected |
|--------|-------------|----------|
| Extend the build-failing gate | HARD: bounded Jacobi scratch, eig reuses covariance/GEMM buffer, no host round-trip between sweeps | ✓ |
| Log-only for this phase | Defer hard SVD/eig assertions; breaks per-phase memory discipline | |

### Convergence policy
| Option | Description | Selected |
|--------|-------------|----------|
| Fixed internal constants | Threshold + max-sweeps as internal constants; primitive has no hyperparameters | ✓ |
| Surface as tunable parameters | Expose tol/max-sweeps; adds surface, invites estimator concerns into a primitive | |

**User's choice:** Extend the build-failing gate; convergence constants internal.

---

## Claude's Discretion

- Jacobi rotation-kernel design (one-sided vs two-sided mechanics, parallel rotation-pair ordering, plane vs shared-memory path) — deferred to the researcher (phase is NEEDS DEEPER RESEARCH).
- Module/file layout in `mlrs-kernels` / `mlrs-backend/prims/`; internal convergence constants; sweep ordering; block/tile sizes; random shapes/seeds; which cases get committed fixtures vs invariant-only checks; new error-variant naming.

## Deferred Ideas

- Per-estimator-family tolerance tables (activate only if a real case can't hold 1e-5).
- Unified single Jacobi core (rejected for two distinct routines).
- Full U(m×m)/V(n×n) SVD (thin only for v1).
- Device-side svd_flip kernel (align at comparison instead).
- Defensive eig symmetrization (trust the covariance feeder).
- wgpu as a gate (dropped to opportunistic project-wide).
