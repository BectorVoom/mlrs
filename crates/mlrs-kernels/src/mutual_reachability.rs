//! Mutual-reachability GATHER kernel for the HDBSCAN feature-metric device
//! front-end (HDBS-01, Phase 15, plan 15-05) — the ONLY new device kernel of the
//! phase.
//!
//! Given a dense pairwise distance block `d` (`rows_x × rows_y`, row-major) and
//! per-row core distances `core`, the kernel computes the dense
//! mutual-reachability matrix
//!
//! ```text
//! out[i*rows_y + j] = max(core[i], core[j], d[i*rows_y + j] / alpha)
//! ```
//!
//! for every guarded output element `(i, j)`. It is structurally identical to
//! `distance.rs::chebyshev_dist`'s STATEMENT-form running-max idiom — a per-element
//! 2D GATHER with NO cross-thread state, NO loop, and only `F` accumulators.
//!
//! ## cpu-MLIR authoring contract (VALIDATED, the same rules `distance.rs` obeys)
//!
//! This kernel is the project's f64 correctness gate under `cubecl-cpu` (the MLIR
//! backend). It honours every landmine from
//! `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md`:
//!
//! - **Per-element 2D launch.** `ABSOLUTE_POS_X` / `ABSOLUTE_POS_Y` (`u32`) with
//!   `CubeDim {x:16, y:16}` and ceiling-div counts, guarded
//!   `if i < rows_x { if j < rows_y { … } }` — NEVER a bare 1D `ABSOLUTE_POS`
//!   launch (FINDING 002-A, a loud MLIR pass failure → reads back zeros).
//! - **STATEMENT-form running max.** The three-way `max(core_i, core_j, d/alpha)`
//!   is a mutable-`F` `if`-guard (`let mut acc = …; if ci > acc { acc = ci; }`),
//!   NEVER an `if`-expression in value position.
//! - **No cross-sibling-loop accumulator** (FINDING 002-B, a SILENT miscompile):
//!   inert here — the kernel has NO loop, just a per-element three-way max.
//! - **Banned entirely** (panic at launch): `SharedMemory`, `Atomic`,
//!   `F::INFINITY`, mutable-`bool` scans, descending-shift loops. The kernel is
//!   SharedMemory-free BY CONSTRUCTION (a per-element GATHER carries no
//!   cross-thread state).
//! - Scalars (`alpha`) pass **by value** in cubecl 0.10 (no `ScalarArg` wrapper).
//!
//! The host-launch wrapper lives in `mlrs-backend`
//! (`prims/mutual_reachability.rs`, which owns the concrete `ActiveRuntime`); the
//! VALUE oracle (incl. a duplicate-point row, R-9) is asserted there under a real
//! runtime. This crate stays backend-feature-free (Criterion 1).

use cubecl::prelude::*;

/// Dense mutual-reachability GATHER:
/// `out[i*rows_y + j] = max(core[i], core[j], d[i*rows_y + j] / alpha)`.
///
/// One unit per output element `(i, j)`; `i` on `ABSOLUTE_POS_X`,
/// `j` on `ABSOLUTE_POS_Y`. The three-way running max is a STATEMENT-form mutable
/// `if` guard (cpu-MLIR-safe — never an `if`-expression in value position).
///
/// `alpha` is the robust-single-linkage scaling (Variant-A placement on a dense
/// distance block: `d_ij / alpha` BEFORE the core-distance max). It passes by
/// value (cubecl 0.10 has no `ScalarArg` wrapper); the host validates `alpha > 0`
/// at `HdbscanBuilder::build` BEFORE any launch (a zero `alpha` would divide by
/// zero → inf, not a typed error — so the build-time guard is load-bearing).
#[cube(launch)]
pub fn mutual_reachability<F: Float + CubeElement>(
    d: &Array<F>,
    core: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    alpha: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            // STATEMENT-form three-way running max (chebyshev_dist:117-123
            // precedent); NEVER an if-expression in value position.
            let mut acc = d[(i * rows_y + j) as usize] / alpha;
            let ci = core[i as usize];
            let cj = core[j as usize];
            if ci > acc {
                acc = ci;
            }
            if cj > acc {
                acc = cj;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}
