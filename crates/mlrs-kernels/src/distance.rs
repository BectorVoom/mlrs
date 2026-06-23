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
