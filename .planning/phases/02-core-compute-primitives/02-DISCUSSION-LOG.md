# Phase 2: Core Compute Primitives - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 2-Core Compute Primitives
**Areas discussed:** Reduction axis scope, Primitive API contract, Distance strategy, Memory gate, Oracle source

---

## Reduction axis scope

### Q1 — Axis support for reduction primitives
| Option | Description | Selected |
|--------|-------------|----------|
| Full + row + column | Full-array AND axis-wise (column + row) reductions | ✓ |
| Full-array only | 1D total reductions only; axis added later | |
| Full + only what's needed | Full-array + only the specific axis ops each estimator needs | |

**User's choice:** Full + row + column

### Q2 — argmin/argmax scope
| Option | Description | Selected |
|--------|-------------|----------|
| Full + per-row argmin | Full-array argmin/argmax + per-row argmin (KMeans assignment), lowest-index tie-break | ✓ |
| Full-array argmin only | 1D argmin only; per-row assignment built inside KMeans | |
| You decide | Claude chooses during planning | |

**User's choice:** Full + per-row argmin

---

## Primitive API contract

### Q1 — Matrix shape representation
| Option | Description | Selected |
|--------|-------------|----------|
| Explicit dims per call | DeviceArray stays flat 1D; pass (rows, cols) per call | ✓ |
| Extend DeviceArray to 2D | Add shape state into DeviceArray | |
| You decide | Claude chooses during planning | |

**User's choice:** Explicit dims per call

### Q2 — Primitive I/O contract
| Option | Description | Selected |
|--------|-------------|----------|
| DeviceArray in/out | Device-resident; no host round-trips; host helpers in tests only | ✓ |
| Host-slice convenience | Host slices in/out, upload/read-back per call | |
| Both layers | Device core + host convenience wrapper | |

**User's choice:** DeviceArray in/out

### Q3 — GEMM transpose handling
| Option | Description | Selected |
|--------|-------------|----------|
| BLAS-style flags | transa/transb flags; XᵀX reuses GEMM directly | ✓ |
| Row-major, caller transposes | GEMM row-major only; materialize transpose for XᵀX | |
| You decide | Claude chooses during research/planning | |

**User's choice:** BLAS-style flags (transpose-kernel fallback only if cubecl-matmul lacks transposed operands)

---

## Distance strategy

### Q1 — Computation method
| Option | Description | Selected |
|--------|-------------|----------|
| GEMM-expansion + clamp | ‖x‖²+‖y‖²−2·XYᵀ, reuses GEMM + reduction, max(d²,0) clamp | ✓ |
| Direct difference accumulation | Σ(x−y)² per pair; no cancellation but no GEMM reuse | |
| Expansion default + direct fallback | Expansion primary, direct fallback for f32 edge cases | |

**User's choice:** GEMM-expansion + clamp

### Q2 — Output form
| Option | Description | Selected |
|--------|-------------|----------|
| Squared core + sqrt flag | Squared distance core + optional sqrt at boundary for KNN | ✓ |
| Squared only | Squared only; estimators apply sqrt themselves | |
| You decide | Claude chooses during planning | |

**User's choice:** Squared core + sqrt flag

---

## Memory gate

### Q1 — Hard assertions for Phase 2
| Option | Description | Selected |
|--------|-------------|----------|
| Reuse + no-round-trip + Gram reuse | Three hard assertions (reuse>0, no mid-pipeline host round-trip, Gram reuses GEMM buffer) | ✓ |
| Reuse-rate only | Just assert reuse counter increments + bounded peak bytes | |
| Counters logged, soft check | Keep logging; no build-failing threshold | |

**User's choice:** Reuse + no-round-trip + Gram reuse (activates Phase 1 D-05 deferred gate)

### Q2 — Buffer handling
| Option | Description | Selected |
|--------|-------------|----------|
| Optional out-param + pool scratch | Optional caller-provided output + pooled scratch; fresh array when absent | ✓ |
| Always allocate-and-return | Always allocate fresh; rely on pool free-list | |
| You decide | Claude chooses during planning | |

**User's choice:** Optional out-param + pool scratch

---

## Oracle source

### Q1 — Reference for primitive validation
| Option | Description | Selected |
|--------|-------------|----------|
| Host reference + convention fixtures | Live Rust host reference (primary) + small numpy .npz fixtures for conventions | ✓ |
| Live Rust host reference only | Naive CPU loops in-test only; no convention cross-check | |
| Committed numpy .npz fixtures | Phase 1 pattern exactly; fixed shapes, venv regen | |

**User's choice:** Host reference + convention fixtures (cov ddof, distance squared, GEMM pinned by fixtures)

---

## Claude's Discretion
- Module/file layout in mlrs-kernels and mlrs-backend (subject to AGENTS.md source/test separation)
- Kernel tiling/block sizes, shared-memory tile dims (subject to PLANE_DIM / no-hardcoded-width)
- Exact cubecl-matmul API surface + transpose-flag plumbing (subject to research)
- New primitive error variant names
- Random shapes/seeds for the host-reference sweep; which exact cases get committed convention fixtures

## Deferred Ideas
- Direct difference-accumulation distance kernel (fallback only if f32 expansion fails 1e-5)
- Extending DeviceArray to carry 2D shape
- GEMM transpose-kernel fallback (only if cubecl-matmul lacks transposed operands)
- Per-estimator-family tolerance tables (still deferred to Phase 3/4/5)
