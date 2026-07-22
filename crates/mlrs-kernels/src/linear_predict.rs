//! `linear_predict` — fused on-device linear-model inference kernel
//! (LINEAR-01/02 predict perf lever).
//!
//! Feature-free `#[cube]` kernel generic over `<F: Float + CubeElement>`,
//! launched by `mlrs_backend::prims::linear_predict`.
//!
//! ## Why this exists (the predict-side host-sync pathology)
//! Every dense linear regressor (`LinearRegression`, `Ridge`, `Lasso`,
//! `ElasticNet`) shared ONE predict body: form `raw = X·coef` via the generic
//! tiled `gemm` (a skinny `m×1` output), then broadcast-add the scalar
//! intercept. That broadcast was done on the HOST — `intercept.to_host()`
//! (a blocking scalar readback), then `raw.to_host()` (an `m`-length device→host
//! copy), then an element-wise host loop, then `DeviceArray::from_host()` (an
//! `m`-length host→device copy BACK). The PyO3 boundary then reads the result
//! to host ONE more time. On a discrete GPU across PCIe those round-trips — not
//! the arithmetic — dominate `predict`, exactly like `center`'s per-column
//! readback pathology (see `mlrs_kernels::colmean` module docs) and `gram`'s
//! skinny-GEMM starvation (see `mlrs_kernels::gram`).
//!
//! [`linear_predict_bias`] collapses the whole predict into a SINGLE launch
//! that stays device-resident end-to-end: one unit per output row computes
//! `y[r] = Σ_c X[r,c]·coef[c] + bias`, reading the intercept straight from its
//! length-1 device buffer (no scalar readback) and writing the length-`m`
//! result the caller materializes with its one unavoidable readback. The
//! feature axis these models fit is small and capped
//! (`GRAM_EIG_MAX_FEATURES = 64`), so the per-row dot loop is short and the
//! row-major (mildly uncoalesced) column stride is absorbed by L2 — the win is
//! the eliminated PCIe round-trips and the fused bias, not the FLOPs.
//!
//! ## cubecl-cpu MLIR safety
//! GATHER-only: no `SharedMemory`, no atomics, no mutable `bool` — an ascending
//! `while` scan over `F` accumulators. Safe on EVERY backend (cpu included), so
//! `prims::linear_predict` needs no cpu fallback (unlike the `SharedMemory`
//! `gram`/`colmean` perf kernels).

use cubecl::prelude::*;

/// Fused linear-model inference: `out[r] = Σ_c x[r,c]·coef[c] + bias[0]`.
///
/// - `x` is the `m × n` row-major test matrix, `coef` the length-`n` fitted
///   coefficients, `bias` a length-1 device buffer holding the intercept
///   (`0` for the fit-intercept-`false` case — the caller always supplies a
///   real length-1 buffer, so there is no branch here).
/// - One unit per output row (`r < m`); the slack lanes of the final block are
///   masked by the `r < m` guard. The dot product accumulates in `F`, matching
///   the precision of the `gemm` path it replaces (the fitted feature count is
///   small and capped, so a sequential `F` sum stays within the 1e-5 oracle
///   contract).
#[cube(launch)]
pub fn linear_predict_bias<F: Float + CubeElement>(
    x: &Array<F>,
    coef: &Array<F>,
    bias: &Array<F>,
    out: &mut Array<F>,
    m: u32,
    n: u32,
) {
    let r = ABSOLUTE_POS;
    if r < m as usize {
        let base = r * n as usize;
        let mut acc = F::new(0.0_f32);
        let mut c = 0u32;
        while c < n {
            acc += x[base + c as usize] * coef[c as usize];
            c += 1u32;
        }
        out[r] = acc + bias[0];
    }
}
