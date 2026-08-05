//! `feature_select` — the column-gather kernel every feature selector's
//! `transform` is (FSEL-01).
//!
//! Feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`,
//! composed by `mlrs_backend::prims::feature_score`.
//!
//! ## Why this is the ONLY device kernel in the feature-selection stack
//! `sklearn.feature_selection`'s selectors are, mechanically, a support MASK
//! plus `X[:, mask]`. The mask's computation (the scores, their p-values, the
//! k-best / percentile / FDR thresholds) is `O(n_features)` scalar work in
//! `f64` — see `prims::feature_score`'s module docs for why it accumulates on
//! the host — but the `X[:, mask]` gather is `O(n_samples · n_selected)` pure
//! data movement with NO accumulation, so it is exact in any float width, has
//! no `f64` question to answer, and is the one piece with real device
//! parallelism to exploit. It runs here.
//!
//! ## cubecl-cpu MLIR safety
//! Both kernels touch only `F` loads/stores and `u32` index arithmetic — no
//! `SharedMemory`, no atomics, no transcendentals, no infinity constant — so
//! unlike `gram.rs`'s shared-memory kernels there is no cpu gate to apply and
//! the host prim launches these on every backend.

use cubecl::prelude::*;

/// Gather selected columns of a row-major `rows × cols_in` matrix into a
/// row-major `rows × cols_out` result: `output[r, j] = x[r, idx[j]]`.
///
/// One unit handles one OUTPUT element at `ABSOLUTE_POS` (so the launch is
/// sized by `rows · cols_out`, not by the input), bounds-checked on
/// `tid < output.len()` so the ceiling-division launch may over-provision
/// safely — and so the X/Y-folded grid
/// `mlrs_backend::prims::launch_dims_1d_folded` produces, which can overshoot,
/// is safe without a separate guard.
///
/// The output row/column split is derived from `cols_out` (`r = tid / cols_out`,
/// `j = tid % cols_out`), and the input offset from `cols_in`, so the two
/// strides are genuinely independent — the caller passes both rather than
/// letting the kernel infer one, because a selector's whole point is that
/// `cols_out < cols_in`.
///
/// Writes are perfectly coalesced (consecutive units write consecutive
/// `output` slots); reads are strided by whatever gaps the selected columns
/// leave, which is inherent to a gather and is why there is no tiled variant
/// here.
///
/// `idx` values MUST be `< cols_in`; the host prim
/// (`prims::feature_score::gather_columns`) validates that before launching, so
/// this kernel does not re-check and does not clamp — a clamp would silently
/// duplicate a column into the result rather than surfacing the caller's bug.
#[cube(launch)]
pub fn gather_columns<F: Float + CubeElement>(
    x: &Array<F>,
    idx: &Array<u32>,
    output: &mut Array<F>,
    cols_in: u32,
    cols_out: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < output.len() {
        let r = tid / cols_out as usize;
        let j = tid % cols_out as usize;
        output[tid] = x[r * cols_in as usize + idx[j] as usize];
    }
}

/// Scatter selected columns back into a zero-filled row-major
/// `rows × cols_out` frame: `output[r, idx[j]] = z[r, j]`, every unselected
/// column left at zero.
///
/// This is `SelectorMixin.inverse_transform`, which sklearn defines as exactly
/// that zero-fill (a selector discards information, so the inverse can only
/// restore the geometry, not the values). The kernel is indexed by the INPUT
/// element (`tid < z.len()`) rather than the output, because the output is
/// larger and mostly untouched — the caller zeroes it once and this writes only
/// the `rows · cols_in` slots that carry data.
///
/// Note the reversed naming relative to [`gather_columns`]: here `cols_in` is
/// the width of `z` (the SELECTED count) and `cols_out` the width of the
/// restored frame, so both kernels read "in = my operand, out = my result".
#[cube(launch)]
pub fn scatter_columns<F: Float + CubeElement>(
    z: &Array<F>,
    idx: &Array<u32>,
    output: &mut Array<F>,
    cols_in: u32,
    cols_out: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < z.len() {
        let r = tid / cols_in as usize;
        let j = tid % cols_in as usize;
        // `idx[j]` is hoisted into a local rather than written inline inside the
        // `output[..]` subscript: the `#[cube]` macro lowers the whole index
        // expression of an ASSIGNMENT target as a mutable place, so an
        // immutable `&Array<u32>` read nested in it fails to compile
        // ("cannot borrow `*idx` as mutable"). Reading it first sidesteps that
        // without changing the generated indexing.
        let col = idx[j] as usize;
        output[r * cols_out as usize + col] = z[tid];
    }
}
