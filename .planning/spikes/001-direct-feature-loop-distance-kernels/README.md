---
spike: 001
name: direct-feature-loop-distance-kernels
type: standard
validates: "Given X (n×d), when a #[cube(launch)] kernel loops over the feature dim computing Manhattan (Σ|Δ|), Chebyshev (max|Δ|), and Minkowski-p ((Σ|Δ|^p)^(1/p) via in-kernel F::powf), then it launches under --features cpu and matches a host reference (f64 ≤1e-6, f32 ≤1e-3)"
verdict: VALIDATED
related: [002]
tags: [knn, cpu-mlir, kernel, minkowski, distance, powf]
---

# Spike 001: Direct Feature-Loop Distance Kernels

## What This Validates

**Given** X `(n×d)`, **when** a single `#[cube(launch)]` kernel loops over the feature
dimension (`while kk < cols`) computing Manhattan (`Σ|Δ|`), Chebyshev (`max|Δ|`), and
parameterized Minkowski-p (`(Σ|Δ|^p)^(1/p)` via **in-kernel `F::powf`**), **then** it
launches under `--features cpu` (cpu-MLIR) and matches a host reference (f64 ≤1e-6, f32 ≤1e-3).

This is the named cpu-MLIR keystone unknown from Phase 13 decision **D-06**: the v1
GEMM-expansion is Euclidean-specific and cannot back L1/L∞/Lp, so the multi-metric scope
expansion (D-05) requires NEW direct pairwise distance kernels — and **Minkowski-p needs
in-kernel `pow`**, which the discussion flagged as the open question for this phase. If this
fails, the metric-scope expansion is infeasible as specified.

## Research

Docs/precedent checked before coding (no new external deps — pure CubeCL kernel question):

| Op needed | Proven-elsewhere precedent | cpu-MLIR-safe? |
|-----------|----------------------------|----------------|
| `F::powf(base, exp)` (static) | `poly_map` — `crates/mlrs-kernels/src/elementwise.rs:140`, tested via `kernel_matrix_test`/`kernel_ridge_test` under cpu | ✓ (proven) |
| `.abs()` (instance) | `jacobi_eig` / `jacobi_svd` rotation kernels | ✓ (proven) |
| running max via statement-form `if` | `kde_epanechnikov_map` compact-support guard | ✓ (proven) |
| runtime `while c < cols { … c += 1 }` loop | `top_k` selection-by-rank (one unit, no SharedMemory) | ✓ (proven) |
| **feature-loop accumulator + the above in ONE direct pairwise kernel** | **none — this is the new combination** | **UNKNOWN → this spike** |

**Chosen approach:** three separate `#[cube(launch)]` kernels (`manhattan_dist`,
`chebyshev_dist`, `minkowski_dist`), one unit per output element `(i,j)`, GATHER over the
feature dim with `F`/`u32` accumulators and `if` guards only — no SharedMemory, no mutable
bool, no `F::INFINITY`, no descending-shift loop (the cpu-MLIR landmine set from project
memory). Separate kernels isolate which (if any) op fails to lower. `Y = X` (self-pairwise),
the actual KNN-graph case.

## How to Run

```bash
# from repo root — TEMPORARY run vehicle lives in the backend test dir
cargo test -p mlrs-backend --features cpu --test knn_spike_001_test -- --nocapture
```

(`kernels_and_harness.rs` here is the verbatim copy of
`crates/mlrs-backend/tests/knn_spike_001_test.rs`. The test file is the run vehicle and is
deleted once findings are recorded — it is NOT the real prim. The real kernels land in
`mlrs-kernels` during Phase 13 execution.)

## What to Expect

3 tests, all pass:
- `spike001_f64_direct_distance_kernels` — Manhattan/Chebyshev/Minkowski-3 match host (≤1e-6)
- `spike001_f32_direct_distance_kernels` — same, ≤1e-3
- `spike001_minkowski_subsumes_l1_l2_and_handles_fractional_p` — depth probe (below)

## Investigation Trail

1. **First compile error (API, not lowering):** used the 3-arg `ArrayArg::from_raw_parts::<F>(&h, len, vec)` form; repo's cubecl 0.10 uses the **2-arg by-value** `from_raw_parts(handle, len)` (consumes the handle) — confirmed against `spike_test.rs` / `distance.rs` / `eig.rs`. Output handles come from `client.empty(size_bytes)`. Fixed.
2. **First real run: all three kernels lowered and launched under cpu-MLIR on the first try** — no MLIR panic on the feature loop, the `F::powf`, the `.abs()`, or the running-max statement. Both f64 (the cpu gate precision) and f32 matched.
3. **Duplicate-point dist-0 edge:** rows 0 and 4 are identical; `d(0,4)==0` exactly for every metric (Minkowski `powf(0,p)=0`, no NaN). Asserted.
4. **Depth probe added** to answer the named Claude's-discretion decision (special-case p∈{1,2}?): does the general Minkowski-p kernel SUBSUME the specials, and does a genuinely fractional exponent work? Confirmed `Minkowski(1)==Manhattan`, `Minkowski(2)==true Euclidean (sqrt of L2²)`, and `Minkowski(1.5)` correct — all ≤1e-9.

## Results

**VERDICT: VALIDATED ✓**

The single named cpu-MLIR feasibility risk for Phase 13's metric expansion is cleared:

- A direct pairwise distance kernel with a **runtime feature-dim loop** lowers and launches
  under cpu-MLIR. The "shift-loop" landmine in project memory is specific to SharedMemory
  descending-shift patterns; a plain bounded `while kk < cols` accumulator is fine.
- **In-kernel `F::powf` works** — both as the per-term `Σ|Δ|^p` and as the final `^(1/p)`
  root, for integer AND fractional `p`. The Minkowski-p `pow` unknown is resolved.
- `.abs()` (instance), running-max (statement-form `if`), and `F::from_int(0)` /
  `F::new(1.0)` accumulator seeds all lower cleanly — matching the existing-kernel precedents.
- f64 (cpu gate) and f32 both correct; f32 within 1e-3 of the f64-faithful host ref.

**Signal for the build:**
- Land Manhattan/Chebyshev/Minkowski-p as new direct `#[cube(launch)]` kernels in
  `mlrs-kernels` following this exact GATHER idiom; Euclidean/Cosine keep the v1 GEMM path.
- **Special-casing p∈{1,2} to the fast paths is an optimization, NOT a correctness need** —
  one general Minkowski kernel reproduces L1 and L2 exactly. Recommend: route Euclidean (and
  Cosine on normalized rows) through the launch-proven GEMM-expansion for speed; route
  Manhattan/Chebyshev/Minkowski-p through the general direct kernel. Validate `p ≥ 1` at the
  prim boundary (host-side) — the kernel itself does not guard it.
- Memory gate (R-6): this spike materializes a full `n×n` for n=5 only. The real prim must
  tile the query axis so the `n×n` distance block is never fully resident — unchanged by this
  result; the kernel is per-output-element and tiles trivially over the query (i) axis.

## Surprises

- **Zero lowering friction.** Given the project's history of cpu-MLIR landmines (SharedMemory,
  `F::INFINITY`, mutable bool, shift-loops all panic at launch), the expectation was at least
  one failed lowering to work around. The feature-loop + `powf` combination lowered first-try.
  The accumulated precedent (poly/jacobi/topk) was an accurate predictor: stay inside the
  proven op-set and cpu-MLIR cooperates.
