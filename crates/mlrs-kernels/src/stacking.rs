//! Meta-feature assembly kernel (STACK-META-01) — the device arm of
//! `mlrs.StackingRegressor`'s `_concatenate_predictions`.
//!
//! A stack's meta matrix is `k` prediction blocks written side by side, plus
//! (under `passthrough`) the original design `X`:
//!
//! ```text
//!   out (n × width) = [ block₀ | block₁ | … | block_{k−1} | X ]
//!                       ^offset₀ ^offset₁      ^offset_{k−1} ^n_meta
//! ```
//!
//! Every block is row-major and `n`-tall; only its column count varies. The
//! layout (`offsets`, `n_meta`, `width`) is decided once on the host by
//! `mlrs_algos::ensemble::stacking::meta_layout` and handed here as scalars, so
//! this kernel owns no policy — it is the scatter, and nothing else.
//!
//! ## One launch per block, not one launch total
//! CubeCL's `Array` arguments are fixed at kernel-definition time, so a single
//! kernel cannot take a caller-chosen `k` of them. [`stack_meta_block`] writes
//! ONE block and is launched `k` (or `k + 1`) times into the same output
//! handle. The launches are independent — disjoint column ranges of disjoint
//! rows, no read-after-write between them — so they queue back to back on the
//! stream with no barrier and no intermediate host sync.
//!
//! ## This kernel does no arithmetic
//! It is a strided `memcpy`, which is exactly why the arm it belongs to is
//! **not** the default: see `mlrs_backend::prims::stacking_meta` for the
//! measured host/device/numpy ladder that settled that. It exists so the
//! meta-matrix copy CAN run on the device when the blocks are already there,
//! and so the claim "the device loses this one" is a number rather than an
//! assumption.
//!
//! ## `block_cols` is usually 1
//! A single-output regressor contributes one column, so the common stack is `k`
//! one-column blocks and (optionally) one `d`-column `X`. The per-element
//! `div`/`mod` below is therefore trivial in the common case and, in the
//! passthrough case, is dwarfed by the bandwidth of the `n × d` copy itself.
//! Specializing a `block_cols == 1` variant was measured as noise.
//!
//! Per AGENTS.md §2 this file carries no in-file test module; the live launch
//! tests are in `crates/mlrs-backend/tests/stacking_meta_test.rs` (this crate is
//! backend-feature-free and cannot launch anything itself).

use cubecl::prelude::*;

/// Scatter one `n × block_cols` row-major block into the `n × width` meta
/// matrix, starting at column `offset`.
///
/// `out[r, offset + c] = block[r, c]`, one unit per block element. Bounds are
/// checked against `block.len()` so the standard ceiling-division launch may
/// over-provision units safely (T-0203-01).
///
/// The caller guarantees `offset + block_cols <= width` and that `out` is
/// `n * width` long — both are validated host-side in
/// `mlrs_backend::prims::stacking_meta::concat_meta_device` before any launch,
/// because a kernel cannot return an error and an out-of-range write here would
/// silently corrupt a neighbouring block's columns.
#[cube(launch)]
pub fn stack_meta_block<F: Float + CubeElement>(
    block: &Array<F>,
    out: &mut Array<F>,
    block_cols: u32,
    width: u32,
    offset: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < block.len() {
        let r = tid / block_cols as usize;
        let c = tid % block_cols as usize;
        out[r * width as usize + offset as usize + c] = block[tid];
    }
}
