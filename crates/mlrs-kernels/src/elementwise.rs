//! Per-element map kernels (PRIM-03 / PRIM-04) — `clamp_nonneg`, `sqrt_elem`,
//! `scale`, and the `dist_combine_clamp` distance-combine kernel.
//!
//! All four are `#[cube(launch)]` functions generic over `<F: Float +
//! CubeElement>`, following the `smoke.rs` `saxpy_kernel` shape exactly: one
//! unit handles one element at `ABSOLUTE_POS` (or `(ABSOLUTE_POS_X,
//! ABSOLUTE_POS_Y)` for the 2D combine), bounds-checked so the standard
//! ceiling-division launch may over-provision threads safely (T-0203-01).
//!
//! ## Clamp is a STATEMENT, never an expression (D-07)
//! The non-negative clamp `max(d, 0)` is written as a mutable-variable `if`
//! guard (`let zero = F::from_int(0i64); if d < zero { d = zero; }`), NOT as an
//! `if`-expression or a `max(..)` call. The CubeCL conditionals manual
//! (`Cubecl_conditionals.md`) documents that `if`-expressions can mis-lower in
//! the IR; the statement form is the robust pattern. This clamp is the reason
//! the distance pipeline produces NO negative squared distances under f32
//! catastrophic cancellation (Criterion 3 / Pitfall 5 / T-0203-03).
//!
//! ## `CubeElement` bound (D-13)
//! `CubeElement` is mandatory on `F`: the scalar `factor: F` arg of [`scale`]
//! must implement `LaunchArg` for the generated `launch` fn (same rationale as
//! `saxpy_kernel`'s `a: F`). This crate stays backend-feature-free; a concrete
//! runtime is chosen in `mlrs-backend`.
//!
//! Per AGENTS.md §2, this source file carries NO in-file test module — the live
//! launch tests are in `crates/mlrs-backend/tests/distance_test.rs` (which owns
//! a concrete runtime feature; this crate is feature-free).

use cubecl::prelude::*;

/// Non-negative clamp `out[i] = max(in[i], 0)` (D-07), per element.
///
/// Written in the STATEMENT form (`if d < zero { d = zero; }`) per
/// `Cubecl_conditionals.md` — an `if`-expression or a `max` call could
/// mis-lower. This is the clamp that guarantees no negative distances escape
/// the distance pipeline under f32 cancellation (Criterion 3).
#[cube(launch)]
pub fn clamp_nonneg<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let mut d = input[tid];
        let zero = F::from_int(0i64);
        if d < zero {
            d = zero;
        }
        output[tid] = d;
    }
}

/// Element-wise square root `out[i] = sqrt(in[i])` (D-08, the optional Euclidean
/// boundary for KNN). The distance host API clamps to `>= 0` BEFORE this, so the
/// argument is always non-negative (no `sqrt`-of-negative NaN — T-0203-03).
#[cube(launch)]
pub fn sqrt_elem<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::sqrt(input[tid]);
    }
}

/// Element-wise scale `out[i] = in[i] * factor` (Plan 04 covariance consumes
/// this for the `1/(n-ddof)` normalisation). `factor` is a scalar `F` passed by
/// value (A6 — like `saxpy_kernel`'s `a`).
#[cube(launch)]
pub fn scale<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, factor: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = input[tid] * factor;
    }
}

/// Column centring (PRIM-04 covariance): subtract the per-column mean from each
/// element of a `rows × cols` row-major matrix, `out[r, c] = a[r, c] − mean[c]`.
///
/// One unit handles one element at `ABSOLUTE_POS`; the column index is
/// `tid % cols` so the broadcast mean is read from the length-`cols` `mean`
/// array. Bounds-checked on `tid < a.len()` (over-provisioned threads are
/// no-ops, T-0204-01). `cols` is a scalar `u32` passed by value (cubecl 0.10,
/// like `dist_combine_clamp`'s `rows`/`cols`). Keeps the two-pass covariance
/// centring device-resident (no host round-trip in `covariance.rs`, D-05).
#[cube(launch)]
pub fn center_columns<F: Float + CubeElement>(
    a: &Array<F>,
    mean: &Array<F>,
    output: &mut Array<F>,
    cols: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < a.len() {
        let c = tid % cols as usize;
        output[tid] = a[tid] - mean[c];
    }
}

/// Distance combine + clamp (PRIM-03): for the `rows × cols` output element
/// `(i, j)`, compute `‖x_i‖² + ‖y_j‖² − 2·XYᵀ[i,j]` then clamp to `max(d², 0)`
/// (the GEMM-expansion of the squared Euclidean distance, D-07).
///
/// - `xy` is the row-major `rows × cols` `XYᵀ` matrix (from GEMM).
/// - `xnorm` is the length-`rows` per-row squared norm `‖x_i‖²`.
/// - `ynorm` is the length-`cols` per-row squared norm `‖y_j‖²`.
/// - `out` is the `rows × cols` clamped squared distance.
///
/// The clamp is the STATEMENT form (`if d < zero { d = zero; }`) so no negative
/// squared distance can escape under f32 cancellation (Criterion 3 / Pitfall 5).
/// Bounds-checked on `(i < rows && j < cols)` (T-0203-01).
#[cube(launch)]
pub fn dist_combine_clamp<F: Float + CubeElement>(
    xy: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out: &mut Array<F>,
    rows: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows && j < cols {
        let idx = (i * cols + j) as usize;
        let mut d = xnorm[i as usize] + ynorm[j as usize] - F::new(2.0) * xy[idx];
        let zero = F::from_int(0i64);
        if d < zero {
            d = zero;
        }
        out[idx] = d;
    }
}
