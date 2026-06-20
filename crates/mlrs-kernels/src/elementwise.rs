//! Per-element map kernels (PRIM-03 / PRIM-04 / PRIM-08) — `clamp_nonneg`,
//! `sqrt_elem`, `scale`, `center_columns`, the `dist_combine_clamp`
//! distance-combine kernel, and the kernel-family maps `rbf_map` / `poly_map` /
//! `sigmoid_map` (PRIM-08, the `kernel_matrix` in-place maps over the
//! distance/GEMM base), and the six KernelDensity density-value maps
//! `kde_gaussian_map` / `kde_epanechnikov_map` / `kde_tophat_map` /
//! `kde_exponential_map` / `kde_linear_map` / `kde_cosine_map` plus the
//! `div_by_row` log-sum-exp rescale helper (KERNEL-02, the linear-domain density
//! maps over the v1 distance base — compact-support kernels yield exact 0 out of
//! support via STATEMENT-form guards, never the infinity constant, D-11).
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

/// RBF (Gaussian) kernel map (PRIM-08): `out[i] = exp(-γ·in[i])`, where `in` is
/// the already-computed squared-Euclidean distance `‖xᵢ − yⱼ‖²` (the
/// `distance(.., sqrt=false)` base, D-03). One unit per element at
/// `ABSOLUTE_POS`, bounds-checked so the ceiling-division launch may
/// over-provision safely (T-08-02-01).
///
/// `gamma` is a scalar `F` passed by value (A6 — like [`scale`]'s `factor`),
/// hence the `CubeElement` bound. The transcendental is the STATIC associated fn
/// `F::exp(..)`, NEVER the `x.exp()` instance form (Pitfall 7 — the instance form
/// can mis-lower in the `#[cube]` IR). Shared-memory-free, atomics-free, and free
/// of the infinity constant (cpu-MLIR-safe — module doc precedent).
#[cube(launch)]
pub fn rbf_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, gamma: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::exp(-gamma * input[tid]);
    }
}

/// Polynomial kernel map (PRIM-08): `out[i] = (γ·in[i] + coef0)^degree`, where
/// `in` is the `XYᵀ` Gram entry `⟨xᵢ, yⱼ⟩` (the `gemm(.., transb=true)` base,
/// D-03). One unit per element at `ABSOLUTE_POS`, bounds-checked (T-08-02-01).
///
/// `gamma` / `coef0` / `degree` are scalar `F` by value (A6). `degree` is a REAL
/// `F` (not an integer) and the exponent is the STATIC `F::powf(base, degree)` —
/// the sklearn-faithful real-exponent form (A3 / Pitfall 7), never the `x.powf()`
/// instance form. Shared-memory-free, atomics-free, no infinity constant.
#[cube(launch)]
pub fn poly_map<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    gamma: F,
    coef0: F,
    degree: F,
) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::powf(gamma * input[tid] + coef0, degree);
    }
}

/// Sigmoid kernel map (PRIM-08): `out[i] = tanh(γ·in[i] + coef0)`, where `in` is
/// the `XYᵀ` Gram entry `⟨xᵢ, yⱼ⟩` (the `gemm(.., transb=true)` base, D-03). One
/// unit per element at `ABSOLUTE_POS`, bounds-checked (T-08-02-01).
///
/// `gamma` / `coef0` are scalar `F` by value (A6). The transcendental is the
/// STATIC `F::tanh(..)`, never the `x.tanh()` instance form (Pitfall 7).
/// Shared-memory-free, atomics-free, no infinity constant — `tanh` is a bounded
/// finite transcendental over the finite Gram base (T-08-02-03).
#[cube(launch)]
pub fn sigmoid_map<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    gamma: F,
    coef0: F,
) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::tanh(gamma * input[tid] + coef0);
    }
}

/// KernelDensity gaussian map (PRIM-08 / KERNEL-02, D-07): `out[i] =
/// exp(−0.5·sqdist/h²)`, where `in` is the already-computed SQUARED-euclidean
/// distance `‖qᵢ − xⱼ‖²` (the `distance(.., sqrt=false)` base, Pitfall 4 — gaussian
/// uses SQUARED distance). One unit per element at `ABSOLUTE_POS`, bounds-checked.
///
/// `h` is the bandwidth passed by value (A6). The transcendental is the STATIC
/// associated fn `F::exp(..)`, never the instance form (Pitfall 7). Gaussian has
/// NO compact support — it is finite and positive everywhere. Shared-memory-free,
/// atomics-free, no infinity constant (cpu-MLIR-safe, D-11).
#[cube(launch)]
pub fn kde_gaussian_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let inv_h2 = F::new(1.0) / (h * h);
        output[tid] = F::exp(F::new(-0.5) * input[tid] * inv_h2);
    }
}

/// KernelDensity epanechnikov map (PRIM-08 / KERNEL-02, D-07/D-11): `out[i] =
/// 1 − sqdist/h²` inside support, exact `0` outside, where `in` is the SQUARED
/// distance (Pitfall 4). Compact support: the value is zero when `dist ≥ h`, i.e.
/// `sqdist ≥ h²`. One unit per element at `ABSOLUTE_POS`, bounds-checked.
///
/// The compact-support guard is the STATEMENT form (`let mut val = …; if sqdist
/// >= h² { val = zero; }`) per `Cubecl_conditionals.md`, mirroring the
/// `clamp_nonneg` / `dist_combine_clamp` idiom — NEVER an if-expression, NEVER
/// the infinity constant / `−∞` (D-11 / Pitfall 3 — out-of-support yields EXACT 0 in the
/// linear domain, the `log` is applied once at the very end host/estimator-side).
/// Shared-memory-free, atomics-free, no infinity constant.
#[cube(launch)]
pub fn kde_epanechnikov_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let h2 = h * h;
        let sqdist = input[tid];
        let zero = F::from_int(0i64);
        let mut val = F::new(1.0) - sqdist / h2;
        if sqdist >= h2 {
            val = zero;
        }
        output[tid] = val;
    }
}

/// KernelDensity tophat map (PRIM-08 / KERNEL-02, D-07/D-11): `out[i] = 1` inside
/// support (`dist < h`), exact `0` outside, where `in` is the RAW euclidean
/// distance (Pitfall 4 — the `distance(.., sqrt=true)` base). One unit per element
/// at `ABSOLUTE_POS`, bounds-checked.
///
/// STATEMENT-form compact-support guard (D-11 / Pitfall 3 — exact 0 outside, never
/// the infinity constant). Shared-memory-free, atomics-free, no infinity constant.
#[cube(launch)]
pub fn kde_tophat_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let dist = input[tid];
        let zero = F::from_int(0i64);
        let mut val = F::new(1.0);
        if dist >= h {
            val = zero;
        }
        output[tid] = val;
    }
}

/// KernelDensity exponential map (PRIM-08 / KERNEL-02, D-07): `out[i] =
/// exp(−dist/h)`, where `in` is the RAW euclidean distance (Pitfall 4 — the
/// `distance(.., sqrt=true)` base). One unit per element at `ABSOLUTE_POS`,
/// bounds-checked.
///
/// The exponential kernel has NO compact support — it is finite and positive
/// everywhere. Transcendental via the STATIC `F::exp(..)` (Pitfall 7).
/// Shared-memory-free, atomics-free, no infinity constant (D-11).
#[cube(launch)]
pub fn kde_exponential_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = F::exp(-input[tid] / h);
    }
}

/// KernelDensity linear map (PRIM-08 / KERNEL-02, D-07/D-11): `out[i] = 1 −
/// dist/h` inside support (`dist < h`), exact `0` outside, where `in` is the RAW
/// euclidean distance (Pitfall 4). One unit per element at `ABSOLUTE_POS`,
/// bounds-checked.
///
/// STATEMENT-form compact-support guard (D-11 / Pitfall 3 — exact 0 outside, never
/// the infinity constant). Shared-memory-free, atomics-free, no infinity constant.
#[cube(launch)]
pub fn kde_linear_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let dist = input[tid];
        let zero = F::from_int(0i64);
        let mut val = F::new(1.0) - dist / h;
        if dist >= h {
            val = zero;
        }
        output[tid] = val;
    }
}

/// KernelDensity cosine map (PRIM-08 / KERNEL-02, D-07/D-11): `out[i] =
/// cos(0.5·π·dist/h)` inside support (`dist < h`), exact `0` outside, where `in`
/// is the RAW euclidean distance (Pitfall 4). One unit per element at
/// `ABSOLUTE_POS`, bounds-checked.
///
/// The half-π constant is `π/2 ≈ 1.5707963267948966`. Transcendental via the
/// STATIC `F::cos(..)` (Pitfall 7). STATEMENT-form compact-support guard (D-11 /
/// Pitfall 3 — exact 0 outside, never the infinity constant). Shared-memory-free,
/// atomics-free, no infinity constant.
#[cube(launch)]
pub fn kde_cosine_map<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, h: F) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let dist = input[tid];
        let zero = F::from_int(0i64);
        let half_pi = F::new(1.570_796_326_794_896_6);
        let mut val = F::cos(half_pi * dist / h);
        if dist >= h {
            val = zero;
        }
        output[tid] = val;
    }
}

/// Element-wise reciprocal-scale map (KERNEL-02 log-sum-exp rescale helper, D-11):
/// `out[i] = in[i] / divisor[row(i)]`, dividing each element by the per-row scalar
/// in `divisor` (length `rows`; `cols` columns per row, broadcast across the row).
/// One unit per element at `ABSOLUTE_POS`, bounds-checked.
///
/// This is the OPTIONAL reduce-max rescale step of the linear-domain log-sum-exp
/// (divide each row's kernel values by that row's max before summing, then add
/// `log(max)` back once at the end). `cols` is a scalar `u32` by value (cubecl
/// 0.10). Shared-memory-free, atomics-free, no infinity constant — division by a
/// strictly-positive per-row max never produces `±∞`.
#[cube(launch)]
pub fn div_by_row<F: Float + CubeElement>(
    input: &Array<F>,
    divisor: &Array<F>,
    output: &mut Array<F>,
    cols: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let r = tid / cols as usize;
        output[tid] = input[tid] / divisor[r];
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
