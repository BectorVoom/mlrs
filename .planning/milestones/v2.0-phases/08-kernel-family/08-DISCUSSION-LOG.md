# Phase 8: Kernel Family - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 8-kernel-family
**Areas discussed:** kernel_matrix prim API, KernelRidge fidelity, KernelDensity kernel scope, KD oracle & numerics

---

## kernel_matrix prim API

### Kernel param representation

| Option | Description | Selected |
|--------|-------------|----------|
| Typed enum w/ per-variant params | `Kernel<F>` enum: Linear, Rbf{gamma}, Poly{gamma,degree,coef0}, Sigmoid{gamma,coef0}. Type-safe, cleanest seam. | ✓ |
| Flat fn + tag + all params | All params always passed even when unused; less type-safe. | |
| One prim fn per kernel | Separate linear/rbf/poly/sigmoid fns, no dispatch. | |

### Symmetric vs general

| Option | Description | Selected |
|--------|-------------|----------|
| Always full general K(X,Y) | One branch-free path; K(X,X) passes Y=X; ~2× redundant on symmetric case OK at v2 sizes. | ✓ |
| Exploit symmetry for K(X,X) | Upper triangle + mirror; saves ~half but adds branch + mirror. | |

### Base-op composition

| Option | Description | Selected |
|--------|-------------|----------|
| Self-contained: prim dispatches base op | Internally calls v1 distance (RBF) / GEMM (linear/poly/sigmoid), then map. One call site. | ✓ |
| Map-only: caller passes precomputed base | Prim applies only the map; callers orchestrate the base op. | |

**User's choice:** Typed enum + always-general + self-contained dispatch.
**Notes:** All three recommended options taken — prioritizes a clean, type-safe, self-contained seam for Phase 9 / kernel-SVM reuse.

---

## KernelRidge fidelity

### Multi-target

| Option | Description | Selected |
|--------|-------------|----------|
| Support multi-target | y as n×t; (K+αI)⁻¹Y via multi-RHS Cholesky — near-free, full parity. | ✓ |
| Single-target only | n_targets=1; defer multi-output. | |

### gamma default

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror sklearn exactly | gamma=None → 1/n_features; explicit used as-is; oracle pins both. | ✓ |
| Require explicit gamma | No None default. | |

### HP surface

| Option | Description | Selected |
|--------|-------------|----------|
| Scalar alpha + 4 kernels only | No precomputed, no per-target alpha array, no intercept. | ✓ |
| Add precomputed + per-target alpha | More surface, more oracle cases. | |

**User's choice:** Multi-target + sklearn-faithful gamma + tight scalar-alpha 4-kernel surface.
**Notes:** Full numeric parity where it's nearly free (multi-target via existing multi-RHS Cholesky), tight surface elsewhere.

---

## KernelDensity kernel scope

### KD kernels

| Option | Description | Selected |
|--------|-------------|----------|
| All 6 (full parity) | gaussian/tophat/epanechnikov/exponential/linear/cosine; -inf handled host/linear-domain. | ✓ |
| gaussian + exponential | Two smooth infinite-support kernels only. | |
| gaussian only | sklearn default only. | |

### KD base

| Option | Description | Selected |
|--------|-------------|----------|
| v1 distance prim directly | KD = distance + density-map + normalize + LSE; NOT via kernel_matrix prim. | ✓ |
| Force gaussian through kernel_matrix(Rbf) | Split path for marginal reuse. | |

### bandwidth

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric bandwidth only | float > 0; defer scott/silverman. | |
| Numeric + scott/silverman | Also the two auto-bandwidth string rules. | ✓ |

**User's choice:** All 6 kernels + distance-prim base + numeric AND scott/silverman bandwidth.
**Notes:** Maximum sklearn parity for KernelDensity. Clarified that KD is a distinct kernel family from the kernel_matrix prim (functions of raw distance, not dot product).

---

## KD oracle & numerics

### KD oracle

| Option | Description | Selected |
|--------|-------------|----------|
| sklearn w/ rtol=0, atol=0 | Force sklearn KD exact; faithful to oracle=sklearn rule. | ✓ |
| NumPy brute-force reference | Hand-rolled reference, spot-check vs sklearn. | |

### LSE placement

| Option | Description | Selected |
|--------|-------------|----------|
| Host-side LSE | Read back m×n, reduce host-side; sidesteps F::INFINITY landmine. | |
| Device-side LSE | Reduce on-device via v1 reduce prim, device-resident. | ✓ |

**User's choice:** sklearn-exact oracle + device-side LSE.
**Notes:** Device-side LSE chosen (memory-efficiency / device-residency). Implies the LSE MUST operate in the linear (non-log) kernel domain so out-of-support zeros never become `F::INFINITY` (cpu-MLIR landmine); optional reduce-max rescale for stability; single log at the end. Captured as D-11 in CONTEXT.md.

---

## Claude's Discretion

- Exact f32-on-rocm tolerance bands for KernelRidge predictions and KernelDensity log-density — follow v1 documented-band precedent.
- Whether the D-11 reduce-max rescale is needed vs a plain linear reduce-sum — decide from numerical testing.
- Exact `'scott'`/`'silverman'` formulas — pin from sklearn source.
- Single parameterized map kernel vs one kernel per variant — planner's call (SharedMemory/atomics-free either way).

## Deferred Ideas

- `kernel='precomputed'` + per-target `alpha` array (KernelRidge).
- Tree-based KD acceleration (BallTree/KDTree).
- Kernel SVC/SVR (SMO) — v3 backlog.
- Bespoke fused kernel-matrix-then-reduce device kernel — only if compose-over-distance/GEMM proves a perf/memory problem.
