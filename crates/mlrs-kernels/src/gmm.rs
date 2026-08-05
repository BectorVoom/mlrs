//! `gmm` — the bulk `O(n·k·d)` / `O(n·k·d²)` `#[cube]` kernels behind
//! [`GaussianMixture`](../../mlrs_algos/mixture/gaussian_mixture/struct.GaussianMixture.html)'s
//! DEVICE EM engine (`mlrs_backend::prims::gmm_device::GmmDevice`).
//!
//! `gmm_host`'s module docs (`mlrs-backend/src/prims/gmm_host.rs`) give three
//! reasons the WHOLE EM loop is host-resident on every backend: launch overhead
//! dominates a `max_iter × n_init` loop of tiny passes, the reduction needs
//! `f64` (and cuda's `supports_type(F64)` under-reports), and the per-iteration
//! tail (Cholesky, triangular inverse) is `O(k·d³)`, serial and branchy. None of
//! that changes here — this file does NOT reimplement the tail. It moves ONLY
//! the two passes whose cost actually scales with `n` off the host: the E-step
//! weighted-log-prob / responsibility-normalize / `nk`+`means` reduction, and
//! the M-step covariance reduction. Everything `O(k·d)`-or-smaller (the
//! Cholesky, the log-determinant, the bias fold, the final divide) stays host
//! arithmetic in `gmm_device.rs`, exactly as it does in `gmm_host.rs`.
//!
//! ## Kernel set
//! - [`gmm_wlp_direct`] — one thread per `(row, component)`: the weighted
//!   log-probability `bias[c] − ½·maha(xᵢ, μ_c, P_c)`, branching on
//!   `covariance_type` (`ct: u32`, `0=full 1=tied 2=diag 3=spherical`). This is
//!   the DENSE twin of `gmm_host::maha_full`/`maha_tied_row`: the host walks
//!   only the stored upper triangle of `P_c`, this kernel walks all `d²`
//!   entries (the below-diagonal ones are zero by construction, so the result
//!   is identical) — a GPU thread doing `O(d²)` redundant work per output is
//!   the right trade here, not a triangular loop nest that would serialize
//!   unevenly across threads.
//! - [`gmm_resp_normalize_rows`] — one thread per row: the row-wise
//!   `logsumexp` + `resp[i,c] = exp(wlp[i,c] − lse_i)`, plus the per-row `lse_i`
//!   written out so the caller can fold it to `mean_lpn` via
//!   `prims::kmeans::sum_device`'s existing row-blocked partial-sum (never a
//!   second bespoke reducer).
//! - [`gmm_soft_sumcount_blocked`] — the ROW-BLOCKED soft-weight generalization
//!   of `kmeans`'s [`crate::kmeans::centroid_sumcount_blocked`]: one-hot
//!   `labels[i] == c` becomes a dense `resp[i,c]` weight, so every block
//!   accumulates `Σ resp[i,c]·x[i,j]` (→ `means`, before the divide) and
//!   `Σ resp[i,c]` (→ `nk`, before `+= NK_EPS`) for every block.
//! - [`gmm_cov_diag_blocked`] — row-blocked partial `Σ resp[i,c]·x[i,j]²`, the
//!   `diag`/`spherical` M-step term (finished on host as `s/nk − μ² +
//!   reg_covar`, `spherical` additionally averaging over `d`).
//! - [`gmm_cov_full_blocked`] — row-blocked partial DENSE outer product
//!   `Σ resp[i,c]·(xᵢ−μ_c)⊗(xᵢ−μ_c)`, redundantly computing both symmetric
//!   halves (the `gmm_wlp_direct` trade again — simpler indexing beats a
//!   packed triangle on a GPU).
//! - [`gmm_xtx_blocked`] — row-blocked partial DENSE `Σ xᵢ⊗xᵢ`, computed ONCE
//!   per fit for the `tied` M-step's loop-invariant `XᵀX` (`gmm_host`'s win #2,
//!   ported to device: sklearn and the host engine both avoid recomputing this
//!   every iteration, and so does this arm).
//! - [`gmm_fold_partials`] — the generic stage-2 fold shared by EVERY blocked
//!   kernel above: sums `nblocks` length-`len` partials down to one length-`len`
//!   output. One kernel body serves `nk` (`len = k`), `means`/`cov_diag`
//!   (`len = k·d`), `cov_full` (`len = k·d²`) and `xtx` (`len = d²`) — mirroring
//!   `kmeans::centroid_reduce_partials`'s role but generalized past the
//!   `(F sums, u32 counts)` pair `centroid_reduce_partials` is specialized to
//!   (`nk` here is a soft `f64` sum, not an integer count).
//!
//! ## cubecl-cpu MLIR safety
//! Every kernel here uses ONLY `F`/`u32` accumulators, ascending `while` loops
//! guarded by `if`, and GATHER-shaped output (`ABSOLUTE_POS`-indexed, each unit
//! writes exactly the slots it owns) — no `SharedMemory`, no atomics, no
//! `F::INFINITY` sentinel, no descending shift (the `kmeans.rs` / `lbfgs.rs`
//! precedent this file follows verbatim). [`gmm_resp_normalize_rows`] evaluates
//! `.exp()` / `.ln()` at whatever width `F` is instantiated at — the caller
//! (`gmm_device.rs`) pins that to `f64` and gates the whole arm off backends
//! without `f64` transcendentals (`capability::f64_transcendental_supported`),
//! mirroring `prims::lbfgs::softmax_loss_grad`'s own f64-transcendental gate.
//!
//! All kernels carry NO backend feature (D-13). Tests live in
//! `crates/mlrs-algos/tests/gaussian_mixture_device_test.rs` (AGENTS.md §2).

use cubecl::prelude::*;

/// `wlp[i,c] = bias[c] − ½·maha(xᵢ, μ_c, P_c)` for every `(row, component)`
/// pair, `ct` selecting the Mahalanobis form (`0=full 1=tied 2=diag
/// 3=spherical`).
///
/// - `x` is the row-major `n × d` design.
/// - `means` is `k × d` row-major.
/// - `prec` is the precision-Cholesky buffer in whatever layout `ct` implies:
///   `full` → `k × d × d` (one dense block per component, upper triangular with
///   zeros below the diagonal — read densely, see the module docs); `tied` →
///   ONE shared `d × d` block; `diag` → `k × d`; `spherical` → length `k`.
/// - `bias` is length `k`: `ln(weight_c) + log_det_cholesky_c − ½·d·ln(2π)`,
///   folded by the host caller so this kernel adds a single number.
/// - `wlp` is the `n × k` output.
///
/// One unit per `(i, c)` (`ABSOLUTE_POS = i·k + c`); GATHER-only, no
/// cross-unit state.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gmm_wlp_direct<F: Float + CubeElement>(
    x: &Array<F>,
    means: &Array<F>,
    prec: &Array<F>,
    bias: &Array<F>,
    wlp: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    ct: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let xbase = i * d;
        let mbase = c * d;
        let half = F::new(0.5_f32);
        let mut acc = F::new(0.0_f32);
        if ct == 0u32 {
            // full: dense k blocks of d x d (upper-triangular by construction;
            // read every entry, the zeros below the diagonal contribute 0).
            let pbase = c * d * d;
            let mut j = 0u32;
            while j < d {
                let mut yj = F::new(0.0_f32);
                let mut a = 0u32;
                while a < d {
                    let da = x[(xbase + a) as usize] - means[(mbase + a) as usize];
                    yj += da * prec[(pbase + a * d + j) as usize];
                    a += 1u32;
                }
                acc += yj * yj;
                j += 1u32;
            }
        } else if ct == 1u32 {
            // tied: ONE shared d x d block for every component; only `means`
            // varies with `c`.
            let mut j = 0u32;
            while j < d {
                let mut yj = F::new(0.0_f32);
                let mut a = 0u32;
                while a < d {
                    let da = x[(xbase + a) as usize] - means[(mbase + a) as usize];
                    yj += da * prec[(a * d + j) as usize];
                    a += 1u32;
                }
                acc += yj * yj;
                j += 1u32;
            }
        } else if ct == 2u32 {
            // diag: prec is k x d.
            let mut a = 0u32;
            while a < d {
                let e = (x[(xbase + a) as usize] - means[(mbase + a) as usize])
                    * prec[(c * d + a) as usize];
                acc += e * e;
                a += 1u32;
            }
        } else {
            // spherical: prec is length k (one scalar per component).
            let pc = prec[c as usize];
            let mut s = F::new(0.0_f32);
            let mut a = 0u32;
            while a < d {
                let e = x[(xbase + a) as usize] - means[(mbase + a) as usize];
                s += e * e;
                a += 1u32;
            }
            acc = s * pc * pc;
        }
        wlp[tid] = bias[c as usize] - half * acc;
    }
}

/// Row-wise `logsumexp` + responsibility normalize: `resp[i,c] = exp(wlp[i,c] −
/// lse_i)`, `lse_i = max_c wlp[i,c] + ln Σ_c exp(wlp[i,c] − max_c wlp[i,c])`.
///
/// `lse_out[i] = lse_i` is written so the caller can fold it to `mean_lpn`
/// through `prims::kmeans::sum_device`'s existing row-blocked partial-sum
/// reduction, rather than a second bespoke reducer here.
///
/// One unit per row (`ABSOLUTE_POS = i`); GATHER-only.
#[cube(launch)]
pub fn gmm_resp_normalize_rows<F: Float + CubeElement>(
    wlp: &Array<F>,
    resp: &mut Array<F>,
    lse_out: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = (i as u32) * k;
        let mut mx = wlp[base as usize];
        let mut c = 1u32;
        while c < k {
            let v = wlp[(base + c) as usize];
            if v > mx {
                mx = v;
            }
            c += 1u32;
        }
        let mut se = F::new(0.0_f32);
        let mut c2 = 0u32;
        while c2 < k {
            se += (wlp[(base + c2) as usize] - mx).exp();
            c2 += 1u32;
        }
        let lse = mx + se.ln();
        lse_out[i] = lse;
        let mut c3 = 0u32;
        while c3 < k {
            resp[(base + c3) as usize] = (wlp[(base + c3) as usize] - lse).exp();
            c3 += 1u32;
        }
    }
}

/// ROW-BLOCKED soft-weight `nk`/`means` partial — the dense-`resp` twin of
/// `kmeans::centroid_sumcount_blocked`.
///
/// One unit per `(block b, component c, feature j)`
/// (`ABSOLUTE_POS = b·k·d + c·d + j`): scan only block `b`'s rows and
/// accumulate `Σ resp[i,c]·x[i,j]` into `psums[b·k·d + c·d + j]`; the `j == 0`
/// unit additionally accumulates `Σ resp[i,c]` into `pnk[b·k + c]`. Fold with
/// [`gmm_fold_partials`] (`len = k·d` for `psums`, `len = k` for `pnk`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gmm_soft_sumcount_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    resp: &Array<F>,
    psums: &mut Array<F>,
    pnk: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * k * d;
    if tid < total as usize {
        let b = (tid as u32) / (k * d);
        let rem = (tid as u32) % (k * d);
        let c = rem / d;
        let j = rem % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut cnt = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            let r = resp[(i * k + c) as usize];
            acc += r * x[(i * d + j) as usize];
            if j == 0u32 {
                cnt += r;
            }
            i += 1u32;
        }
        psums[tid] = acc;
        if j == 0u32 {
            pnk[(b * k + c) as usize] = cnt;
        }
    }
}

/// ROW-BLOCKED `diag`/`spherical` M-step partial: `Σ_{i in block}
/// resp[i,c]·x[i,j]²` into `psums[b·k·d + c·d + j]`.
///
/// Fold with [`gmm_fold_partials`] (`len = k·d`); the host finishes as
/// `s/nk_c − μ_cj² + reg_covar` (`diag`), additionally averaged over `d` for
/// `spherical` (mirrors `gmm_host::cov_diag`'s tail, minus the packed-triangle
/// concern that has no diag/spherical analogue).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gmm_cov_diag_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    resp: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * k * d;
    if tid < total as usize {
        let b = (tid as u32) / (k * d);
        let rem = (tid as u32) % (k * d);
        let c = rem / d;
        let j = rem % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            let r = resp[(i * k + c) as usize];
            let v = x[(i * d + j) as usize];
            acc += r * v * v;
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// ROW-BLOCKED `full` M-step partial: the DENSE (both symmetric halves,
/// redundantly — see module docs) weighted outer product `Σ_{i in block}
/// resp[i,c]·(x[i,a] − μ[c,a])·(x[i,jj] − μ[c,jj])` into
/// `psums[b·k·d² + c·d² + a·d + jj]`.
///
/// Fold with [`gmm_fold_partials`] (`len = k·d²`); the host finishes as
/// `s/nk_c` with `reg_covar` added on the `a == jj` diagonal (mirrors
/// `gmm_host::cov_full`'s tail, minus its packed-triangle mirroring — this
/// buffer is already dense in both halves).
///
/// The caller must keep `nblocks` small enough that `nblocks·k·d²` stays
/// within a reasonable device-memory budget (this is `O(n·k·d²)` WORK spread
/// over relatively few blocks, unlike the `O(n·k·d)` kernels above which can
/// use many more, smaller blocks).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gmm_cov_full_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    resp: &Array<F>,
    means: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let dd = d * d;
    let total = nblocks * k * dd;
    if tid < total as usize {
        let b = (tid as u32) / (k * dd);
        let rem = (tid as u32) % (k * dd);
        let c = rem / dd;
        let rem2 = rem % dd;
        let a = rem2 / d;
        let jj = rem2 % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mbase = c * d;
        let ma = means[(mbase + a) as usize];
        let mj = means[(mbase + jj) as usize];
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            let r = resp[(i * k + c) as usize];
            let xi = x[(i * d + a) as usize] - ma;
            let xj = x[(i * d + jj) as usize] - mj;
            acc += r * xi * xj;
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// ROW-BLOCKED, UNWEIGHTED `Σ xᵢ ⊗ xᵢ` partial (dense, both halves) — the
/// `tied` M-step's loop-invariant `XᵀX`, computed ONCE per fit (`gmm_host`'s
/// win #2, ported to device: neither the host engine nor this one recomputes
/// it every iteration).
///
/// `psums[b·d² + a·d + jj] = Σ_{i in block} x[i,a]·x[i,jj]`. Fold with
/// [`gmm_fold_partials`] (`len = d²`).
#[cube(launch)]
pub fn gmm_xtx_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let dd = d * d;
    let total = nblocks * dd;
    if tid < total as usize {
        let b = (tid as u32) / dd;
        let rem = (tid as u32) % dd;
        let a = rem / d;
        let jj = rem % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            acc += x[(i * d + a) as usize] * x[(i * d + jj) as usize];
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// Generic stage-2 fold: sums `nblocks` length-`len` partials down to one
/// length-`len` output — `out[t] = Σ_b partials[b·len + t]`.
///
/// Shared by every blocked kernel above ([`gmm_soft_sumcount_blocked`]'s
/// `psums`/`pnk`, [`gmm_cov_diag_blocked`], [`gmm_cov_full_blocked`],
/// [`gmm_xtx_blocked`]) rather than one bespoke reducer per shape — generalizes
/// `kmeans::centroid_reduce_partials`, which is specialized to an
/// `(F sums, u32 counts)` pair `centroid_reduce_partials` folds together; here
/// every quantity being folded is already an `F` sum (GMM's `nk` is a soft
/// responsibility total, not an integer count), so one kernel serves all of
/// them at whatever `len` the caller needs.
#[cube(launch)]
pub fn gmm_fold_partials<F: Float + CubeElement>(
    partials: &Array<F>,
    out: &mut Array<F>,
    len: u32,
    nblocks: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < len as usize {
        let mut acc = F::new(0.0_f32);
        let mut b = 0u32;
        while b < nblocks {
            acc += partials[(b * len + tid as u32) as usize];
            b += 1u32;
        }
        out[tid] = acc;
    }
}
