//! Direct pairwise GATHER distance kernels for the KNN-graph primitive
//! (PRIM-11, Phase 13) — the empty-but-registered home plan 13-02 fills.
//!
//! The v1 GEMM-expansion `distance` prim only covers (squared) Euclidean; the
//! L1 / L-infinity / general-exponent metrics cannot be expressed as a GEMM and
//! need direct per-output-element feature-loop kernels. Plan 13-02 lands those
//! `#[cube(launch)]` kernels here (one unit per output pair `(i,j)`, a runtime
//! `while kk < cols` loop over the feature dim) plus the per-row `self_drop`
//! GATHER kernel; this file is the Wave-1 scaffold so `pub mod distance;`
//! compiles today (mirrors the Phase-8/9 Wave-0 stub-registration precedent).
//!
//! ## cpu-MLIR authoring contract (the kernels plan 13-02 adds MUST follow)
//!
//! These kernels are validated under `cubecl-cpu` (the MLIR backend, the f64
//! correctness gate). cpu-MLIR fails LOUDLY outside its proven op-set OR — worse
//! — SILENTLY miscompiles. The contract, distilled from the validated spikes:
//!
//! - **STATIC transcendentals only.** Use the associated form `F::powf(diff, p)`
//!   / `F::powf(acc, inv_p)` for the general-exponent metric — a bounded
//!   feature-loop accumulator plus `F::powf` lowers fine. NEVER the instance
//!   `x.powf()` form (it can mis-lower in the `#[cube]` IR). `.abs()` is the one
//!   instance form that is allowed (jacobi-proven).
//! - **STATEMENT-form running comparison.** The L-infinity running maximum is a
//!   mutable-variable `if` guard (`let mut acc = …; if diff > acc { acc = diff; }`),
//!   NEVER an `if`-expression in value position. Diffs are non-negative so the
//!   `F::from_int(0i64)` seed is correct.
//! - **Per-element 2D launch** for the pairwise kernels: `ABSOLUTE_POS_X` /
//!   `ABSOLUTE_POS_Y` (`u32`) with `CubeDim {x:16, y:16}` and ceiling-div counts,
//!   guarded `if i < rows_x { if j < rows_y { … } }`.
//! - **Per-row GATHER launch** for the self-drop kernel: `CUBE_POS_X` /
//!   `UNIT_POS_X == 0u32` with `CubeCount::Static(n, 1, 1)`, `CubeDim {x:1,y:1,z:1}`
//!   — NEVER a bare 1D `ABSOLUTE_POS` launch (that is a loud MLIR pass failure;
//!   the kernel never runs and reads back zeros).
//! - **No cross-sibling-loop accumulator.** A flag/counter written in one `while`
//!   and read in a SEPARATE sibling `while` SILENTLY miscompiles. Recompute any
//!   per-row positional value with a self-contained nested count inside the
//!   consuming loop (the self-shift `src = s + #self-cols-at-cols-<=-s` idiom).
//! - **`F` / `u32` accumulators only** — no mutable-bool scans. **Banned
//!   entirely** (panic at launch): `SharedMemory`, `Atomic`, the infinity
//!   constant, and descending-shift loops.
//! - Scalar kernel params (dims, the general-exponent value) pass **by value**
//!   in cubecl 0.10 (no `ScalarArg` wrapper).
//!
//! Plan 13-02 adds the kernel bodies AND their `pub use distance::{…}` re-export
//! line in lib.rs as part of that plan's edit (file-disjoint, single-owner).

use cubecl::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Direct pairwise distance kernels (cpu-MLIR-safe; VALIDATED spike-001 shapes).
//
// One unit per output element `(i, j)`; a runtime `while kk < cols` loop over the
// feature dim; only `F`/`u32` accumulators + `if` guards. No SharedMemory, no
// Atomic, no infinity constant, no mutable-bool scan, no descending-shift loop.
// `.abs()` is the jacobi-proven instance form (the one allowed instance form);
// the general-exponent power MUST be the STATIC `F::powf` associated form (the
// instance `x.powf()` can mis-lower in the `#[cube]` IR). Output is row-major
// (`rows_x × rows_y`): `out[i * rows_y + j]`.
// ─────────────────────────────────────────────────────────────────────────────

/// Manhattan (L1) pairwise distance: `out[i*rows_y+j] = sum_k |x_ik - y_jk|`.
///
/// cpu-MLIR contract: per-element 2D launch (`ABSOLUTE_POS_{X,Y}`), bounded
/// feature loop, `F`/`u32` accumulators only; the per-term absolute difference
/// uses the allowed instance `.abs()` form, no root applied.
#[cube(launch)]
pub fn manhattan_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += diff;
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Chebyshev (L-infinity) pairwise distance: `out[i*rows_y+j] = max_k |x_ik - y_jk|`.
///
/// cpu-MLIR contract: the running maximum is a mutable-variable STATEMENT-form
/// `if` guard (`if diff > acc { acc = diff; }`), NEVER an `if`-expression in value
/// position. Per-term differences are non-negative so the `F::from_int(0i64)` seed
/// is correct.
#[cube(launch)]
pub fn chebyshev_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                if diff > acc {
                    acc = diff;
                }
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Minkowski-`p` pairwise distance: `out[i*rows_y+j] = (sum_k |x_ik - y_jk|^p)^(1/p)`.
///
/// # Precondition (caller obligation, WR-02)
/// `p >= 1` is a HARD caller precondition: this kernel computes `inv_p = 1/p` with
/// NO in-kernel positive-`p` guard (an in-kernel branch would risk a cpu-MLIR
/// mis-lower and the host already validates `p` typed). A `p == 0` launch divides
/// by zero (→ inf) and then `F::powf(acc, inv_p)` yields inf/NaN distances rather
/// than a typed error. The ONLY supported launch path is through the validated
/// `knn_graph` entry (`validate_geometry` rejects `p < 1` BEFORE any launch); do
/// not launch this kernel directly with unchecked `p`.
///
/// The named cpu-MLIR feasibility unknown for this phase (VALIDATED spike 001):
/// an in-kernel general-exponent power inside the feature-loop accumulator, then a
/// final `^(1/p)` root. cpu-MLIR contract: BOTH powers use the STATIC associated
/// `F::powf(base, exp)` form (the instance form can mis-lower); `p` passes by value
/// (cubecl 0.10 has no `ScalarArg` wrapper). Subsumes L1 (`p=1`) and L2 (`p=2`) per
/// the spike depth probe; fast-path special-casing is an optimization, not a
/// correctness need.
#[cube(launch)]
pub fn minkowski_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    p: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += F::powf(diff, p);
                kk += 1u32;
            }
            let inv_p = F::new(1.0) / p;
            out[(i * rows_y + j) as usize] = F::powf(acc, inv_p);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-drop-by-index-identity GATHER kernel (cpu-MLIR-safe; VALIDATED spike-002).
//
// Input is the `top_k(k+1)` result (ascending `(val, idx)` per row); output is the
// `k` true neighbours with the self column (the slot whose index == the query row)
// removed — the `include_self=false` UMAP path (D-02: drop by INDEX IDENTITY, not
// first-zero-distance, so a duplicate point at distance 0 is handled correctly).
// ─────────────────────────────────────────────────────────────────────────────

/// Per-row self-drop GATHER: removes the index-identity self column from a
/// `top_k(k+1)` result, emitting the `k` true neighbours per row.
///
/// cpu-MLIR contract (two VALIDATED landmines this kernel must NOT trip):
/// - **002-A (loud):** launch via `CUBE_POS_X` / `UNIT_POS_X == 0u32` (one cube per
///   query row, one selecting unit) — NEVER a bare 1D `ABSOLUTE_POS` launch, which
///   is a loud MLIR pass failure (the kernel never runs and reads back zeros).
/// - **002-B (silent):** the per-output-slot shift is recomputed LOCALLY via a
///   nested count inside the consuming `while` (`src = s + #self-cols-at-cols-<=-s`)
///   — NEVER a flag/counter written in one `while` and read in a separate sibling
///   `while` (that silently miscompiles under the cube macro).
///
/// Fallback (R-3): if self is absent from the top-`(k+1)` (shouldn't happen for
/// X-vs-X), `bump` stays 0 for every `s` so `src = s`, dropping the last column `k`.
/// Uses only `u32`/`F` accumulators and STATEMENT-form `if`; no mutable bool, no
/// SharedMemory, no infinity constant.
#[cube(launch)]
pub fn self_drop_gather<F: Float + CubeElement>(
    in_val: &Array<F>,
    in_idx: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    k: u32,
    k1: u32, // k + 1
) {
    let row = CUBE_POS_X;
    if row < rows {
        if UNIT_POS_X == 0u32 {
            let ibase = row * k1;
            let obase = row * k;
            let mut s = 0u32;
            while s < k {
                let mut bump = 0u32;
                let mut c = 0u32;
                while c < s + 1u32 {
                    if in_idx[(ibase + c) as usize] == row {
                        bump += 1u32;
                    }
                    c += 1u32;
                }
                let src = s + bump;
                out_val[(obase + s) as usize] = in_val[(ibase + src) as usize];
                out_idx[(obase + s) as usize] = in_idx[(ibase + src) as usize];
                s += 1u32;
            }
        }
    }
}
